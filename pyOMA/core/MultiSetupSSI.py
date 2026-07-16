# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2015-2025  Simon Marwitz, Volkmar Zabel, Andrei Udrea et al.
"""Multi-setup SSI with pre-estimation re-scaling (PreGER).

This module implements the pre-global-estimation-re-scaling (PreGER) merging
strategy of Döhler, Lam & Mevel (2013, MSSP 36(2), doi:10.1016/j.ymssp.2012.06.009)
for multi-setup operational modal analysis.  In contrast to the post-estimation
re-scaling of :class:`~pyOMA.core.SSICovRef.PogerSSICovRef`, each setup's
observability is normalised to a common modal basis *before* the global system
identification.  The approach is theoretically valid under changing ambient
excitation across setups, requires no per-mode mode-shape gluing, and admits a
rigorous first-order covariance computation (implemented in
:class:`VarPreGERSSI`).

Two classes are provided:

* :class:`PreGERSSI` -- point estimates of the modal parameters.
* :class:`VarPreGERSSI` -- additionally propagates measurement uncertainty to
  first-order covariances of the modal parameters (see the class for status).

The merging and identification chain is kept agnostic to *how* each setup's
subspace matrix is built (covariance-driven block Hankel or, in a later phase,
projection/data-driven), mirroring the ``subspace_method`` switch of
:class:`~pyOMA.core.VarSSIRef.VarSSIRef`.

References
----------
Döhler, M., Lam, X.-B. & Mevel, L. "Uncertainty quantification for modal
parameters from stochastic subspace identification on multi-setup measurements."
Mechanical Systems and Signal Processing 36(2), 2013, pp. 562-581.
"""

import os
import warnings

import numpy as np
import scipy.linalg

from .PreProcessingTools import PreProcessSignals
from .ModalBase import ModalBase
from .SSICovRef import PogerSSICovRef
from .Helpers import validate_array, simplePbar, lq_decomp

import logging

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


class PreGERSSI(ModalBase):
    """Pre-Global Estimation and Re-scaling (PreGER) multi-setup SSI.

    Merges correlation functions from multiple measurement setups by
    normalising each setup's observability matrix to the reference-DOF basis of
    the first setup *before* the global identification, then extracts the state
    and output matrices from the merged observability matrix (Döhler, Lam &
    Mevel 2013).

    The standard workflow is:

    1. For each setup, create a
       :class:`~pyOMA.core.PreProcessingTools.PreProcessSignals` object with
       pre-computed correlation functions and call :meth:`add_setup`.
    2. :meth:`pair_channels` -- match channels across setups and determine the
       common reference DOFs.
    3. :meth:`build_subspace_matrices` -- build and decompose the per-setup
       block-Hankel subspace matrices.
    4. :meth:`compute_modal_params` -- run the multi-order modal identification.

    Notes
    -----
    The reference channels play two roles, both auto-determined from the
    channel-DOF assignments:

    1. *Merging* -- the DOFs common to *all* setups (and chosen as reference
       channels in every setup) define the basis into which each setup's
       observability is normalised.  Their correlation functions form the
       columns of every setup's block-Hankel matrix.
    2. *Row ordering* -- each setup's output channels are physically reordered
       so the common-reference channels come first, which lets the merged
       output matrix ``C`` be assembled by plain block-row selection.

    Unlike PoGER, mode shapes are globally scaled by construction; no
    least-squares re-scaling of partial mode shapes is performed.
    """

    def __init__(self):
        """Initialise an empty PreGER merger; add setups with :meth:`add_setup`."""
        super().__init__()

        #              0         1        2          3        4
        # self.state = [subspace, paired, modal par, setups,  variances]
        self.state = [False, False, False, False, False]

        self.setup_name = 'merged_'

        # add_setup
        self.setups = []
        self.sampling_rate = None
        self.num_ref_channels = None
        self.m_lags = None

        # pair_channels
        self.ssi_ref_channels = None
        self.rescale_ref_channels = None
        self.setup_row_orders = None
        self.merged_chan_dofs = None
        self.merged_accel_channels = None
        self.merged_velo_channels = None
        self.merged_disp_channels = None
        self.merged_num_channels = None
        self.num_analised_channels = None

        # build_subspace_matrices
        self.num_block_columns = None
        self.num_block_rows = None
        self.subspace_method = None
        self.setup_n_l = None
        self.svd_h1 = None
        self.hankels_mov = None
        self.svd_refs = None
        # transient per-setup projection blocks (data-driven variance source),
        # stashed by build_subspace_matrices for a VarPreGERSSI subclass to consume
        self._projection_blocks = None

    # ── properties ────────────────────────────────────────────────────────────

    @property
    def accel_channels(self):
        # merged lists are required by integrate_quantities; using the last
        # setup's per-setup lists (as a plain ModalBase would) is a bug.
        return self.merged_accel_channels

    @property
    def velo_channels(self):
        return self.merged_velo_channels

    # ── add_setup ─────────────────────────────────────────────────────────────

    def add_setup(self, prep_signals):
        """Register one measurement setup.

        Validates consistency with previously added setups (type, channel-DOF
        assignments, sampling rate) and stores a deep copy of the setup's
        correlation functions.  The correlation properties of
        :class:`PreProcessSignals` are mutable, ``_last_meth``-switched caches,
        so a deep copy is mandatory to keep each setup's data stable.

        Parameters
        ----------
        prep_signals : PreProcessSignals
            Pre-processed signal object with pre-computed correlation functions
            (e.g. via ``corr_blackman_tukey(m_lags=...)``) and channel-DOF
            assignments.
        """
        if not isinstance(prep_signals, PreProcessSignals):
            raise TypeError(
                f"Expected PreProcessSignals for 'prep_signals', got {type(prep_signals).__name__!r}.")

        if not prep_signals.chan_dofs:
            raise ValueError("prep_signals.chan_dofs must be set before calling this method.")

        if prep_signals.corr_matrix is None and prep_signals.signals is None:
            raise ValueError(
                "prep_signals must carry either pre-computed correlation functions "
                "(covariance method, e.g. corr_blackman_tukey(m_lags=...)) or raw "
                "signals (projection method).")

        if self.sampling_rate is not None:
            if prep_signals.sampling_rate != self.sampling_rate:
                raise ValueError(
                    f"prep_signals.sampling_rate ({prep_signals.sampling_rate}) "
                    f"does not match self.sampling_rate ({self.sampling_rate}).")
        else:
            self.sampling_rate = prep_signals.sampling_rate

        if self.num_ref_channels is not None:
            if self.num_ref_channels != prep_signals.num_ref_channels:
                warnings.warn(
                    'This setup contains a different number of reference channels ({}), '
                    'than the previous setups ({})!'.format(
                        prep_signals.num_ref_channels, self.num_ref_channels))
                self.num_ref_channels = min(self.num_ref_channels, prep_signals.num_ref_channels)
        else:
            self.num_ref_channels = prep_signals.num_ref_channels

        if prep_signals.m_lags is not None:
            self.m_lags = (prep_signals.m_lags if self.m_lags is None
                           else min(self.m_lags, prep_signals.m_lags))

        self.setup_name += prep_signals.setup_name + '_'

        # deep-copy the correlation cache: the corr_* properties are mutable and
        # switched by prep_signals._last_meth, so a reference would not be stable.
        # signals are kept (reference) for the projection/data-driven method.
        corr = prep_signals.corr_matrix
        self.setups.append({
            'setup_name': prep_signals.setup_name,
            'num_analised_channels': prep_signals.num_analised_channels,
            'chan_dofs': prep_signals.chan_dofs,
            'ref_channels': prep_signals.ref_channels,
            'accel_channels': prep_signals.accel_channels,
            'velo_channels': prep_signals.velo_channels,
            'disp_channels': prep_signals.disp_channels,
            'corr_matrix': None if corr is None else np.array(corr, copy=True),
            'signals': prep_signals.signals,
            'start_time': prep_signals.start_time,
        })

        logger.info('Added setup "{}" with {} channels'.format(
            prep_signals.setup_name, prep_signals.num_analised_channels))

        # keep the last setup for spectra display in stabilization plots
        self.prep_signals = prep_signals

        self.state[3] = True

    # ── pair_channels ─────────────────────────────────────────────────────────

    def pair_channels(self):
        """Pair channels across setups and determine common reference DOFs.

        Re-uses the static channel-pairing helpers of
        :class:`~pyOMA.core.SSICovRef.PogerSSICovRef`, with two PreGER-specific
        additions:

        1. The common reference DOFs are filtered to those chosen as reference
           channels in *every* setup (so all setups contribute the same ``r``
           columns), and their per-setup lengths are checked for equality.
        2. A physical row permutation :attr:`setup_row_orders` is stored for
           *every* setup (common references first, in ``ssi_ref_dofs`` order,
           then the remaining channels in ascending order).  This lets each
           setup's Hankel matrix be split into reference/moving parts by plain
           block-row selection and lets ``C`` land in the merged channel order.
        """
        logger.info('Pairing channels and dofs...')
        setups = self.setups

        # Step 1: per-setup channel-DOF and channel-type lists
        merged_chan_dofs, merged_accel_channels, merged_velo_channels, merged_disp_channels = \
            PogerSSICovRef._extract_setup_channel_lists(setups)

        # Step 2: DOFs common to ALL setups
        ssi_ref_dofs = PogerSSICovRef._find_common_ssi_ref_dofs(merged_chan_dofs)

        # Step 2b (PreGER): keep only DOFs that are reference channels in every setup
        ssi_ref_dofs = self._filter_ssi_ref_dofs(setups, ssi_ref_dofs)

        # Step 3: map common DOFs to per-setup channel indices
        ssi_ref_channels, rescale_ref_channels = PogerSSICovRef._compute_ref_channel_indices(
            setups, merged_chan_dofs, ssi_ref_dofs)

        # PreGER: after filtering every setup must contribute the same r columns
        r = len(ssi_ref_dofs)
        for setup_num, this_ssi in enumerate(ssi_ref_channels):
            if len(this_ssi) != r:
                raise RuntimeError(
                    f"Setup {setup_num} contributes {len(this_ssi)} common reference "
                    f"channels but {r} were expected; channel pairing is inconsistent.")

        # Step 3b (PreGER): physical row order per setup (refs first, then rovings)
        setup_row_orders = []
        for setup_num, setup in enumerate(setups):
            refs = [int(c) for c in rescale_ref_channels[setup_num]]
            rovs = sorted(set(range(setup['num_analised_channels'])).difference(refs))
            setup_row_orders.append(refs + rovs)

        # Step 4: reorder first-setup bookkeeping (refs first, then rovings)
        merged_chan_dofs[0], merged_accel_channels[0], \
            merged_velo_channels[0], merged_disp_channels[0] = \
            PogerSSICovRef._reorder_first_setup_channels(
                merged_chan_dofs[0], merged_accel_channels[0],
                merged_velo_channels[0], merged_disp_channels[0],
                ssi_ref_dofs)

        # Step 5: remove reference-DOF channels from all subsequent setups
        PogerSSICovRef._remove_ref_dofs_from_setups(
            merged_chan_dofs[1:], merged_accel_channels[1:],
            merged_velo_channels[1:], merged_disp_channels[1:],
            ssi_ref_dofs)

        # Step 6: flatten lists and assign global ascending channel numbers
        merged_chan_dofs, merged_accel_channels, \
            merged_velo_channels, merged_disp_channels = \
            PogerSSICovRef._flatten_channel_lists(
                merged_chan_dofs, merged_accel_channels,
                merged_velo_channels, merged_disp_channels)

        self.ssi_ref_channels = ssi_ref_channels
        self.rescale_ref_channels = rescale_ref_channels
        self.setup_row_orders = setup_row_orders
        self.merged_chan_dofs = merged_chan_dofs
        self.merged_accel_channels = merged_accel_channels
        self.merged_velo_channels = merged_velo_channels
        self.merged_disp_channels = merged_disp_channels
        self.merged_num_channels = len(merged_chan_dofs)

        # C has exactly merged_num_channels rows -- no truncation hack needed
        self.num_analised_channels = self.merged_num_channels
        self.num_ref_channels = r
        self.start_time = min(stp['start_time'] for stp in setups)

        self.state[1] = True

        return ssi_ref_channels, merged_chan_dofs

    @staticmethod
    def _filter_ssi_ref_dofs(setups, ssi_ref_dofs):
        """Restrict common DOFs to those chosen as reference channels everywhere.

        Only DOFs that are reference channels in *every* setup can be used as
        Hankel-matrix columns.  A warning is issued for each dropped DOF; an
        empty result raises.
        """
        kept = []
        for rnode, raz, relev, rname in ssi_ref_dofs:
            in_all = True
            for setup in setups:
                is_ref = False
                for chan_dof in setup['chan_dofs']:
                    channel, node, az, elev = chan_dof[:4]
                    name = chan_dof[4] if len(chan_dof) == 5 else ''
                    if node == rnode and az == raz and elev == relev and name == rname:
                        is_ref = channel in setup['ref_channels']
                        break
                if not is_ref:
                    in_all = False
                    break
            if in_all:
                kept.append((rnode, raz, relev, rname))
            else:
                warnings.warn(
                    'Common DOF (node={}, az={}, elev={}, name={!r}) is not a reference '
                    'channel in every setup and is dropped from the merging basis.'.format(
                        rnode, raz, relev, rname))
        if not kept:
            raise RuntimeError(
                'No DOF is common to all setups AND a reference channel in every setup; '
                'PreGER merging is not possible.')
        return kept

    # ── build_subspace_matrices ───────────────────────────────────────────────

    def build_subspace_matrices(self, num_block_columns, num_block_rows=None,
                                subspace_method='covariance', num_blocks=None):
        """Build and decompose the per-setup subspace matrices ``H^(j)``.

        Each setup ``j`` yields a subspace matrix ``H^(j)`` of shape
        ``((p+1)*n_l_j, cols)`` whose ``(p+1)`` output block rows (reordered
        common-references-first) carry the observability structure.  For the
        first setup its SVD is stored; for every subsequent setup the moving-row
        matrix and the SVD of the reference-row matrix are stored -- these feed
        the observability merging in :meth:`_observability_parts`.  The merging
        and downstream identification are agnostic to how ``H^(j)`` is built.

        Two constructions are supported:

        * ``'covariance'`` -- a block-Hankel of output/reference correlations,
          block ``(i, k) = corr[:, :, i+k]`` (``cols = q*r``; see notes).
        * ``'projection'`` -- the data-driven UPC subspace matrix ``H^dat``
          from the two-pass LQ decomposition of the raw data Hankels
          (``cols = p*r``, requires ``q == p``; Döhler paper Appendix A).

        Parameters
        ----------
        num_block_columns : int
            Number of block columns ``q``.  For ``'projection'`` it must equal
            ``num_block_rows`` (it is coerced with a warning otherwise).
        num_block_rows : int, optional
            ``p`` (there are ``p+1`` output block rows).  Defaults to
            *num_block_columns*.
        subspace_method : {'covariance', 'projection'}, optional
        num_blocks : int, optional
            Projection method only: number of blocks the signal is split into
            for the two-pass LQ (defaults to each setup's ``n_segments`` if
            correlations were computed, else 50).

        Notes
        -----
        pyOMA's ``corr_matrix[:, :, k]`` holds the correlation at time lag ``k``
        (0-based), and :class:`~pyOMA.core.SSICovRef.BRSSICovRef` treats that
        very ``k=0`` entry as the paper's ``R_1`` slot.  The covariance method
        follows the same convention, so the merged Hankel spans the same column
        space as the single-setup Toeplitz -- a prerequisite for the ``Ns=1``
        equivalence with ``BRSSICovRef``.
        """
        if not self.state[1]:
            raise RuntimeError('Call pair_channels() before build_subspace_matrices().')

        if subspace_method not in ('covariance', 'projection'):
            raise ValueError(
                f"subspace_method must be 'covariance' or 'projection', got {subspace_method!r}.")

        if not isinstance(num_block_columns, int):
            raise TypeError(
                f"Expected int for 'num_block_columns', got {type(num_block_columns).__name__!r}.")
        if num_block_rows is None:
            num_block_rows = num_block_columns
        if not isinstance(num_block_rows, int):
            raise TypeError(
                f"Expected int for 'num_block_rows', got {type(num_block_rows).__name__!r}.")

        p = num_block_rows
        q = num_block_columns

        if subspace_method == 'covariance':
            # highest lag index used is p+q-1, so p+q <= m_lags is required;
            # require one more to stay clear of the last, noisiest lag.
            if self.m_lags is None:
                raise RuntimeError(
                    "subspace_method='covariance' requires pre-computed correlation "
                    "functions (e.g. corr_blackman_tukey) on every setup.")
            if p + q + 1 > self.m_lags:
                raise RuntimeError(
                    'Correlation functions were pre-computed up to {} time lags, which is '
                    'insufficient for a block Hankel with {} block rows and {} block columns '
                    '(requires num_block_rows + num_block_columns + 1 <= m_lags).'.format(
                        self.m_lags, p + 1, q))
            cols = q * self.num_ref_channels
        else:  # projection: q == p, columns are the p past-reference blocks
            if q != p:
                warnings.warn(
                    "subspace_method='projection' requires num_block_columns == "
                    "num_block_rows; coercing num_block_columns to num_block_rows.")
                q = p
            cols = p * self.num_ref_channels

        r = self.num_ref_channels

        setup_n_l = []
        svd_h1 = None
        hankels_mov = [None] * len(self.setups)
        svd_refs = [None] * len(self.setups)
        projection_blocks = [None] * len(self.setups)

        for j, setup in enumerate(self.setups):
            n_l = setup['num_analised_channels']
            setup_n_l.append(n_l)

            if subspace_method == 'covariance':
                H = self._covariance_hankel(j, p, q)
            else:
                H, projection_blocks[j] = self._projection_subspace(j, p, num_blocks)

            if n_l <= r:
                warnings.warn(
                    f"Setup {j} has {n_l} channels but {r} common references; it "
                    "contributes no moving (roving) rows to the merged model.")

            if j == 0:
                U1, S1, V1_T = scipy.linalg.svd(H, full_matrices=False)
                svd_h1 = (U1, S1, V1_T)
            else:
                mov_idx = self._block_row_selection(p + 1, n_l, r, n_l)
                ref_idx = self._block_row_selection(p + 1, n_l, 0, r)
                hankels_mov[j] = H[mov_idx, :]
                Ur, Sr, Vr_T = scipy.linalg.svd(H[ref_idx, :], full_matrices=False)
                svd_refs[j] = (Ur, Sr, Vr_T)

        self.subspace_method = subspace_method
        self.num_block_rows = p
        self.num_block_columns = q
        self.setup_n_l = setup_n_l
        self.svd_h1 = svd_h1
        self.hankels_mov = hankels_mov
        self.svd_refs = svd_refs
        self._projection_blocks = projection_blocks if subspace_method == 'projection' else None

        # order cap: n <= min((p+1)*r, cols)
        self.max_model_order = min((p + 1) * r, cols)

        self.state[0] = True
        self.state[2] = False
        self.state[4] = False

    def _covariance_hankel(self, j, p, q):
        """Covariance-driven block Hankel ``H^(j)`` for setup ``j``."""
        setup = self.setups[j]
        if setup['corr_matrix'] is None:
            raise RuntimeError(
                f"Setup {j} has no correlation functions for the covariance method.")
        corr = setup['corr_matrix'][self.setup_row_orders[j], :, :]
        corr = corr[:, self.ssi_ref_channels[j], :]
        return self._hankel_from_corr(corr, p, q)

    def _projection_subspace(self, j, p, num_blocks):
        """Data-driven UPC subspace matrix ``H^dat,(j)`` for setup ``j``.

        Builds the past (reference) / future (all-channel) data Hankels from the
        row-reordered signals, splits them into ``num_blocks`` blocks and forms
        the per-block ``H^dat`` via the two-pass LQ scheme (paper Appendix A,
        ref [35]).  Returns the block mean and the per-block stack (the latter
        for the variance subclass).

        Returns
        -------
        H_mean : ((p+1)*n_l_j, p*r) ndarray
        H_blocks : (num_blocks, (p+1)*n_l_j, p*r) ndarray
        """
        setup = self.setups[j]
        if setup['signals'] is None:
            raise RuntimeError(f"Setup {j} has no signals for the projection method.")

        r = self.num_ref_channels
        n_l = setup['num_analised_channels']
        if num_blocks is None:
            num_blocks = setup['n_segments'] if setup.get('n_segments') else 50

        # reorder columns so the first r are the common references (ssi_ref_dofs order)
        sig = np.asarray(setup['signals'])[:, self.setup_row_orders[j]]
        n_tot = sig.shape[0]
        block_length = int((n_tot - 2 * p) // num_blocks)
        if block_length < r * p:
            raise RuntimeError(
                f"Projection block length ({block_length}) is smaller than r*p ({r * p}); "
                f"reduce num_blocks ({num_blocks}) or num_block_rows ({p}).")
        n = block_length * num_blocks

        y_minus = np.zeros((p * r, n))
        for ii in range(p):
            y_minus[(p - ii - 1) * r:(p - ii) * r, :] = sig[ii:ii + n, :r].T
        y_plus = np.zeros(((p + 1) * n_l, n))
        for ii in range(p + 1):
            y_plus[ii * n_l:(ii + 1) * n_l, :] = sig[p + ii:p + ii + n, :].T
        hankel = np.vstack([y_minus, y_plus])

        blocks = np.hsplit(hankel, np.arange(block_length, n, block_length))

        r11_list, h_dat = [], []
        for blk in blocks:
            blk = blk / (np.sqrt(blk.shape[1]) * num_blocks)
            lmat = lq_decomp(blk, mode='r', unique=True)
            r11_list.append(lmat[:p * r, :p * r])
            h_dat.append(lmat[p * r:p * r + (p + 1) * n_l, :p * r])

        _, q_breve = lq_decomp(np.hstack(r11_list), mode='reduced', unique=True)
        q11 = np.hsplit(q_breve, np.arange(p * r, num_blocks * p * r, p * r))
        h_dat = [hd @ q11[k].T for k, hd in enumerate(h_dat)]

        h_blocks = np.stack(h_dat, axis=0)
        return np.mean(h_blocks, axis=0), h_blocks

    @staticmethod
    def _hankel_from_corr(corr, p, q):
        """Assemble a block-Hankel matrix from correlation functions.

        Parameters
        ----------
        corr : numpy.ndarray, shape (n_l, r, m_lags)
            Correlation functions (rows reordered, reference columns selected).
        p, q : int
            ``p+1`` block rows and ``q`` block columns.

        Returns
        -------
        H : numpy.ndarray, shape ((p+1)*n_l, q*r)
            Block ``(i, k) = corr[:, :, i+k]``.
        """
        n_l, r, _ = corr.shape
        H = np.zeros(((p + 1) * n_l, q * r))
        for i in range(p + 1):
            for k in range(q):
                H[i * n_l:(i + 1) * n_l, k * r:(k + 1) * r] = corr[:, :, i + k]
        return H

    @staticmethod
    def _block_row_selection(num_blocks, block_width, start, stop):
        """Row indices selecting columns ``[start:stop]`` within each block row.

        For a matrix stacked as ``num_blocks`` block rows of ``block_width``
        rows each, returns the flat row indices of sub-rows ``start..stop-1``
        within every block.
        """
        blocks = np.arange(num_blocks)[:, None] * block_width
        within = np.arange(start, stop)[None, :]
        return (blocks + within).ravel()

    # ── observability merging ─────────────────────────────────────────────────

    def _observability_parts(self, order):
        """Return the ``Ns+1`` merged observability parts at a given order.

        The parts are ``[O_ref, Ō^(0,mov), Ō^(1,mov), ...]``:

        * ``O_ref`` -- reference-DOF rows of the first setup's observability,
          ``((p+1)*r, n)``.
        * ``Ō^(0,mov)`` -- moving rows of the first setup's observability
          (identity basis change).
        * ``Ō^(j,mov)`` for ``j >= 1`` -- setup ``j``'s moving-row observability
          normalised into the first setup's state basis via the rank-``n``
          truncated pseudo-inverse of its reference-row Hankel (Remark 1,
          Eq. 17): ``H^(j,mov) @ pinv_n(H^(j,ref)) @ O_ref``.

        All parts share the same state matrix ``A`` by construction, so the
        shift invariance used in :meth:`estimate_state` holds across parts.
        """
        n = order
        p = self.num_block_rows
        r = self.num_ref_channels

        U1, S1, _ = self.svd_h1
        n_l0 = self.setup_n_l[0]
        O0 = U1[:, :n] * np.sqrt(S1[:n])[np.newaxis, :]

        ref_idx0 = self._block_row_selection(p + 1, n_l0, 0, r)
        mov_idx0 = self._block_row_selection(p + 1, n_l0, r, n_l0)
        O_ref = O0[ref_idx0, :]

        parts = [O_ref, O0[mov_idx0, :]]

        for j in range(1, len(self.setups)):
            Ur, Sr, Vr_T = self.svd_refs[j]
            H_mov = self.hankels_mov[j]
            # pinv_n(H^(j,ref)) @ O_ref, evaluated right-to-left (never form pinv)
            tmp = Ur[:, :n].T @ O_ref                 # (n, n)
            tmp = (1.0 / Sr[:n])[:, None] * tmp        # (n, n)
            pinv_O = Vr_T[:n, :].T @ tmp               # (q*r, n)
            parts.append(H_mov @ pinv_O)               # ((p+1)*(n_l_j-r), n)

        return parts

    def estimate_state(self, order):
        """Estimate the state matrix ``A`` and output matrix ``C`` at *order*.

        Assembles the shifted-up/shifted-down merged observability matrices from
        :meth:`_observability_parts` and solves ``A = pinv(O_up) @ O_down``.
        ``C`` is the stack of first block rows of every part, already in the
        merged channel order.

        Returns
        -------
        A : numpy.ndarray, shape (order, order)
        C : numpy.ndarray, shape (merged_num_channels, order)
        """
        if order > self.max_model_order:
            raise RuntimeError(
                f'Order cannot be higher than {self.max_model_order}. '
                'Consider using more block_rows/block_columns.')

        p = self.num_block_rows
        parts = self._observability_parts(order)

        up_blocks, down_blocks, c_blocks = [], [], []
        for part in parts:
            rows = part.shape[0]
            if rows == 0:
                continue
            w = rows // (p + 1)
            up_blocks.append(part[:p * w, :])            # block rows 0..p-1
            down_blocks.append(part[w:(p + 1) * w, :])   # block rows 1..p
            c_blocks.append(part[:w, :])                 # block row 0

        O_up = np.vstack(up_blocks)
        O_down = np.vstack(down_blocks)
        C = np.vstack(c_blocks)

        A = np.linalg.pinv(O_up) @ O_down

        return A, C

    def modal_analysis(self, A, C):
        """Extract modal parameters from a merged state-space model.

        Standard SSI modal analysis: eigendecomposition of ``A``, mode shapes
        ``C @ eigvecs``, removal of complex-conjugate/real poles, frequency and
        damping from the discrete eigenvalues, integration to modal
        displacements and phase-rotation normalisation.  No PoGER-style
        re-scaling is applied -- PreGER mode shapes are globally scaled by
        construction.

        Returns
        -------
        modal_frequencies, modal_damping : (order,) numpy.ndarray
        mode_shapes : (merged_num_channels, order) numpy.ndarray, complex
        eigenvalues : (order,) numpy.ndarray, complex
        """
        accel_channels = self.accel_channels
        velo_channels = self.velo_channels
        sampling_rate = self.sampling_rate

        n_l = self.num_analised_channels
        order = A.shape[0]
        if order != A.shape[1]:
            raise RuntimeError(f"Internal error: A must be square, got shape {A.shape}.")

        modal_frequencies = np.full((order), np.nan)
        modal_damping = np.full((order), np.nan)
        mode_shapes = np.full((n_l, order), np.nan, dtype=complex)
        eigenvalues = np.full((order), np.nan, dtype=complex)

        eigvals, eigvecs_r = np.linalg.eig(A)
        Phi = C.dot(eigvecs_r)

        conj_indices = self.remove_conjugates(eigvals, eigvecs_r, inds_only=True)
        for i, ind in enumerate(conj_indices):
            lambda_i = eigvals[ind]
            mode_shape_i = Phi[:, ind]

            a_i = np.abs(np.arctan2(np.imag(lambda_i), np.real(lambda_i)))
            b_i = np.log(np.abs(lambda_i))
            freq_i = np.sqrt(a_i ** 2 + b_i ** 2) * sampling_rate / 2 / np.pi
            damping_i = 100 * np.abs(b_i) / np.sqrt(a_i ** 2 + b_i ** 2)

            mode_shape_i = self.integrate_quantities(
                mode_shape_i, accel_channels, velo_channels, freq_i * 2 * np.pi)
            mode_shape_i = self.rescale_mode_shape(mode_shape_i)

            modal_frequencies[i] = freq_i
            modal_damping[i] = damping_i
            mode_shapes[:mode_shape_i.shape[0], i] = mode_shape_i
            eigenvalues[i] = lambda_i

        argsort = np.argsort(modal_frequencies)

        return (modal_frequencies[argsort], modal_damping[argsort],
                mode_shapes[:, argsort], eigenvalues[argsort])

    def compute_modal_params(self, max_model_order=None):
        """Multi-order modal identification.

        Successively calls :meth:`estimate_state` and :meth:`modal_analysis` at
        ascending model orders and stores the results in the ModalBase array
        conventions.

        Parameters
        ----------
        max_model_order : int, optional
            Maximum model order.  Defaults to (and is capped at) the order cap
            ``min((p+1)*r, q*r)`` determined by :meth:`build_subspace_matrices`.
        """
        if not self.state[0]:
            raise RuntimeError('Call build_subspace_matrices() before compute_modal_params().')

        order_cap = self.max_model_order
        if max_model_order is None:
            max_model_order = order_cap
        elif not isinstance(max_model_order, int):
            raise TypeError(
                f"Expected int for 'max_model_order', got {type(max_model_order).__name__!r}.")
        elif max_model_order > order_cap:
            raise ValueError(f"max_model_order must be <= {order_cap}, got {max_model_order}.")

        n_l = self.num_analised_channels

        logger.info('Computing modal parameters...')

        modal_frequencies = np.zeros((max_model_order, max_model_order))
        modal_damping = np.zeros((max_model_order, max_model_order))
        mode_shapes = np.zeros((n_l, max_model_order, max_model_order), dtype=complex)
        eigenvalues = np.zeros((max_model_order, max_model_order), dtype=complex)

        pbar = simplePbar(max_model_order - 1)
        for order in range(1, max_model_order):
            next(pbar)

            A, C = self.estimate_state(order)
            f, d, phi, lamda = self.modal_analysis(A, C)

            modal_frequencies[order, :order] = f
            modal_damping[order, :order] = d
            mode_shapes[:phi.shape[0], :order, order] = phi
            eigenvalues[order, :order] = lamda

        self.max_model_order = max_model_order
        self.modal_frequencies = modal_frequencies
        self.modal_damping = modal_damping
        self.mode_shapes = mode_shapes
        self.eigenvalues = eigenvalues

        self.state[2] = True

    # ── persistence ───────────────────────────────────────────────────────────

    def save_state(self, fname):
        """Save the current computation state to a compressed NumPy archive."""
        logger.info('Saving results to  {}...'.format(fname))

        dirname, _ = os.path.split(fname)
        if dirname and not os.path.isdir(dirname):
            os.makedirs(dirname)

        out_dict = {'self.state': self.state, 'self.setup_name': self.setup_name}

        if self.state[3]:  # add_setup
            out_dict['self.setups'] = self.setups
            out_dict['self.sampling_rate'] = self.sampling_rate
            out_dict['self.num_ref_channels'] = self.num_ref_channels
            out_dict['self.m_lags'] = self.m_lags
        if self.state[1]:  # pair_channels
            out_dict['self.ssi_ref_channels'] = np.array(self.ssi_ref_channels, dtype=object)
            out_dict['self.rescale_ref_channels'] = np.array(self.rescale_ref_channels, dtype=object)
            out_dict['self.setup_row_orders'] = np.array(self.setup_row_orders, dtype=object)
            out_dict['self.merged_chan_dofs'] = self.merged_chan_dofs
            out_dict['self.merged_accel_channels'] = self.merged_accel_channels
            out_dict['self.merged_velo_channels'] = self.merged_velo_channels
            out_dict['self.merged_disp_channels'] = self.merged_disp_channels
            out_dict['self.merged_num_channels'] = self.merged_num_channels
            out_dict['self.num_analised_channels'] = self.num_analised_channels
            out_dict['self.start_time'] = self.start_time
        if self.state[0]:  # build_subspace_matrices
            out_dict['self.num_block_columns'] = self.num_block_columns
            out_dict['self.num_block_rows'] = self.num_block_rows
            out_dict['self.subspace_method'] = self.subspace_method
            out_dict['self.setup_n_l'] = self.setup_n_l
            out_dict['self.svd_h1'] = np.array(self.svd_h1, dtype=object)
            out_dict['self.hankels_mov'] = np.array(self.hankels_mov, dtype=object)
            out_dict['self.svd_refs'] = np.array(self.svd_refs, dtype=object)
            out_dict['self.max_model_order'] = self.max_model_order
        if self.state[2]:  # compute_modal_params
            out_dict['self.eigenvalues'] = self.eigenvalues
            out_dict['self.modal_damping'] = self.modal_damping
            out_dict['self.modal_frequencies'] = self.modal_frequencies
            out_dict['self.mode_shapes'] = self.mode_shapes

        np.savez_compressed(fname, **out_dict)

    @classmethod
    def load_state(cls, fname):  # pylint: disable=arguments-differ
        """Restore a :class:`PreGERSSI` from an archive written by :meth:`save_state`."""
        logger.info('Loading results from  {}'.format(fname))

        in_dict = np.load(fname, allow_pickle=True)

        if 'self.state' not in in_dict:
            raise RuntimeError('The result file is missing required components (self.state)')
        state = [bool(s) for s in in_dict['self.state']]

        ssi_object = cls()
        ssi_object.setup_name = str(in_dict['self.setup_name'].item())
        ssi_object.state = state

        if state[3]:  # add_setup
            ssi_object.setups = list(validate_array(in_dict['self.setups']))
            ssi_object.sampling_rate = validate_array(in_dict['self.sampling_rate'])
            ssi_object.num_ref_channels = validate_array(in_dict['self.num_ref_channels'])
            ssi_object.m_lags = validate_array(in_dict['self.m_lags'])
        if state[1]:  # pair_channels
            ssi_object.ssi_ref_channels = [list(x) for x in in_dict['self.ssi_ref_channels']]
            ssi_object.rescale_ref_channels = [list(x) for x in in_dict['self.rescale_ref_channels']]
            ssi_object.setup_row_orders = [list(x) for x in in_dict['self.setup_row_orders']]
            ssi_object.merged_chan_dofs = [[int(float(cd[0])), str(cd[1]), float(cd[2]), float(
                cd[3]), str(cd[4] if len(cd) == 5 else '')] for cd in in_dict['self.merged_chan_dofs']]
            ssi_object.merged_accel_channels = list(validate_array(in_dict['self.merged_accel_channels']))
            ssi_object.merged_velo_channels = list(validate_array(in_dict['self.merged_velo_channels']))
            ssi_object.merged_disp_channels = list(validate_array(in_dict['self.merged_disp_channels']))
            ssi_object.merged_num_channels = int(validate_array(in_dict['self.merged_num_channels']))
            ssi_object.num_analised_channels = int(validate_array(in_dict['self.num_analised_channels']))
            ssi_object.start_time = validate_array(in_dict['self.start_time'])
        if state[0]:  # build_subspace_matrices
            ssi_object.num_block_columns = int(validate_array(in_dict['self.num_block_columns']))
            ssi_object.num_block_rows = int(validate_array(in_dict['self.num_block_rows']))
            ssi_object.subspace_method = str(in_dict['self.subspace_method'].item())
            ssi_object.setup_n_l = list(validate_array(in_dict['self.setup_n_l']))
            ssi_object.svd_h1 = tuple(in_dict['self.svd_h1'])
            ssi_object.hankels_mov = list(in_dict['self.hankels_mov'])
            ssi_object.svd_refs = list(in_dict['self.svd_refs'])
            ssi_object.max_model_order = int(validate_array(in_dict['self.max_model_order']))
        if state[2]:  # compute_modal_params
            ssi_object.eigenvalues = validate_array(in_dict['self.eigenvalues'])
            ssi_object.modal_damping = validate_array(in_dict['self.modal_damping'])
            ssi_object.modal_frequencies = validate_array(in_dict['self.modal_frequencies'])
            ssi_object.mode_shapes = validate_array(in_dict['self.mode_shapes'])

        return ssi_object


class VarPreGERSSI(PreGERSSI):
    """PreGER multi-setup SSI with first-order modal-parameter uncertainties.

    Extends :class:`PreGERSSI` by propagating the (co)variance of the estimated
    correlation functions to first-order standard deviations of the modal
    frequencies, damping ratios and mode shapes.

    Method (factored perturbation propagation, Döhler, Lam & Mevel 2013).  Each
    setup ``j`` is split into ``n_b_j`` blocks; the per-block Hankel deviations
    ``ΔH^(j)_k = (H^(j)_k - H̄^(j)) / sqrt(n_b (n_b - 1))`` are treated as
    concrete perturbation directions (columns).  Every column is propagated
    linearly through the whole identification chain -- SVD perturbation of the
    subspace matrix (Lemma 6), observability merging (Lemma 2/3), the
    least-squares state matrix, and the per-mode eigenvalue/eigenvector
    sensitivities -- to a modal-parameter perturbation ``Δ(·)_k``.  Because the
    setups (and blocks) are independent, the covariance is the sum of outer
    products of the columns, i.e. the variance of any modal quantity is the sum
    of squares of its column perturbations (Theorem 4).

    Two propagation branches arise:

    * A **setup-0** column perturbs ``H^(0)`` and hence ``O_ref``; it ripples
      into every setup's normalised observability
      (``ΔŌ^(j,mov) = H^(j,mov) pinv_n(H^(j,ref)) ΔO_ref``).
    * A **setup-``j>=1``** column perturbs only ``H^(j)``, giving
      ``ΔŌ^(j,mov) = ΔH^(j,mov) (pinv_n O_ref) + H^(j,mov) (Δpinv_n O_ref)``
      via the truncated-pseudo-inverse perturbation (Eq. C.5).

    Grouping all three product-rule terms this way sidesteps the ambiguous
    ``j=1`` term of the paper's Eq. (25)/Algorithm 1.

    Notes
    -----
    Like the point-estimate classes, the mode shapes are integrated from modal
    accelerations/velocities to modal displacements
    (:meth:`~ModalBase.integrate_quantities`); the ``omega``-dependence of that
    integration is propagated into the mode-shape standard deviations via the
    ``Delta f -> Delta omega`` coupling.  Normalisation is then the max-amplitude
    phase rotation (Prop. 5), as in :class:`~pyOMA.core.VarSSIRef.VarSSIRef`,
    rather than the unit re-scaling of :meth:`PreGERSSI.modal_analysis`; the two
    therefore differ by a per-mode complex scale, while frequencies and damping
    ratios are identical.
    """

    def __init__(self):
        super().__init__()
        # add_setup / build_subspace_matrices (variance)
        self.setup_n_b = None
        self.hankel_devs = None
        # effective block count for StabilCalc std-threshold t-factor (conservative:
        # the smallest setup's block count drives the confidence-interval dof)
        self.num_blocks = None
        # compute_modal_params (variance)
        self.std_frequencies = None
        self.std_damping = None
        self.std_mode_shapes = None
        # lazy perturbation-factor cache (not persisted)
        self._pert_factors = None

    # ── add_setup ─────────────────────────────────────────────────────────────

    def add_setup(self, prep_signals):
        """Register a setup and store its block-wise data for the variance.

        For the covariance method the setup must carry block-wise correlation
        estimates (``corr_matrices`` with ``n_segments >= 2``, e.g. from
        ``corr_blackman_tukey(m_lags, n_segments=...)``).  For the projection
        method the raw signals kept by :meth:`PreGERSSI.add_setup` suffice; the
        per-block data Hankels are formed at build time.  Whichever is present
        is stored; the applicable requirement is enforced in
        :meth:`build_subspace_matrices`.
        """
        super().add_setup(prep_signals)

        corr_matrices = prep_signals.corr_matrices
        n_segments = prep_signals.n_segments
        if corr_matrices is not None and n_segments is not None and int(n_segments) >= 2:
            mean_corr = np.mean(corr_matrices, axis=0)
            if not np.allclose(mean_corr, prep_signals.corr_matrix, rtol=1e-6, atol=1e-12):
                warnings.warn(
                    "mean of corr_matrices does not match corr_matrix; the variance "
                    "estimate assumes corr_matrix is the block mean.")
            self.setups[-1]['corr_matrices'] = np.array(corr_matrices, copy=True)
            self.setups[-1]['n_segments'] = int(n_segments)

    # ── build_subspace_matrices ───────────────────────────────────────────────

    def build_subspace_matrices(self, num_block_columns, num_block_rows=None,
                                subspace_method='covariance', num_blocks=None):
        """Build the subspace matrices and the per-block deviation factors.

        The per-block Hankel deviations ``ΔH^(j)_k = (H^(j)_k - H̄^(j)) /
        sqrt(n_b (n_b - 1))`` are formed from block-wise correlations
        (covariance method) or from the per-block data Hankels of the two-pass
        LQ (projection method); the perturbation chain is identical thereafter.
        """
        super().build_subspace_matrices(num_block_columns, num_block_rows,
                                        subspace_method, num_blocks)

        p = self.num_block_rows
        q = self.num_block_columns

        setup_n_b = []
        hankel_devs = []
        for j, setup in enumerate(self.setups):
            if self.subspace_method == 'covariance':
                if 'corr_matrices' not in setup:
                    raise ValueError(
                        f"Setup {j} has no block-wise correlations; call "
                        "corr_blackman_tukey(m_lags, n_segments>=2) before add_setup() "
                        "for covariance-method variances.")
                devs, n_b = self._covariance_deviations(j, p, q)
            else:
                devs, n_b = self._projection_deviations(j)
            if n_b < 2:
                raise ValueError(
                    f"Setup {j} has only {n_b} block(s); >= 2 are required for a "
                    "variance estimate.")
            setup_n_b.append(n_b)
            hankel_devs.append(devs)

        self.setup_n_b = setup_n_b
        self.hankel_devs = hankel_devs
        self.num_blocks = int(min(setup_n_b))
        self._projection_blocks = None  # consumed; free memory
        self._pert_factors = None
        self.state[4] = False

    def _covariance_deviations(self, j, p, q):
        """Per-block covariance-Hankel deviation factors ``(rows, cols, n_b)``."""
        setup = self.setups[j]
        n_b = setup['n_segments']
        row_order = self.setup_row_orders[j]
        refcols = self.ssi_ref_channels[j]

        corr_mean = setup['corr_matrix'][row_order, :, :][:, refcols, :]
        h_mean = self._hankel_from_corr(corr_mean, p, q)
        scale = 1.0 / np.sqrt(n_b * (n_b - 1))
        devs = np.empty((h_mean.shape[0], h_mean.shape[1], n_b))
        for k in range(n_b):
            corr_k = setup['corr_matrices'][k][row_order, :, :][:, refcols, :]
            devs[:, :, k] = (self._hankel_from_corr(corr_k, p, q) - h_mean) * scale
        return devs, n_b

    def _projection_deviations(self, j):
        """Per-block projection ``H^dat`` deviation factors ``(rows, cols, n_b)``."""
        h_blocks = self._projection_blocks[j]      # (n_b, rows, cols)
        n_b = h_blocks.shape[0]
        h_mean = np.mean(h_blocks, axis=0)
        scale = 1.0 / np.sqrt(n_b * (n_b - 1))
        devs = np.moveaxis(h_blocks - h_mean, 0, -1) * scale
        return devs, n_b

    # ── perturbation factors (Lemma 6, order-independent) ─────────────────────

    @staticmethod
    def _svd_triplet_perturbations(H, U, S, Vt, devs, n_max):
        """First-order SVD triplet perturbations for every block deviation.

        Solves the (singular) Lemma-6 system ``B_j [Δu_j; Δv_j] = rhs`` once per
        triplet (factorised via the minimal-norm pseudo-inverse, which supplies
        the paper's normalisation) and applies it to all ``n_b`` block columns.

        Returns
        -------
        dU : (n_max, m, n_b) ndarray
        dV : (n_max, ncol, n_b) ndarray
        dS : (n_max, n_b) ndarray
        """
        m, ncol = H.shape
        n_b = devs.shape[2]
        dU = np.empty((n_max, m, n_b))
        dV = np.empty((n_max, ncol, n_b))
        dS = np.empty((n_max, n_b))
        for j in range(n_max):
            u_j = U[:, j]
            v_j = Vt[j]
            s_j = S[j]
            ds = np.einsum('m,mck,c->k', u_j, devs, v_j)          # (n_b,)
            Dv = np.einsum('mck,c->mk', devs, v_j)                # (m, n_b)
            DTu = np.einsum('mck,m->ck', devs, u_j)               # (ncol, n_b)
            rhs = np.vstack([(Dv - u_j[:, None] * ds[None, :]) / s_j,
                             (DTu - v_j[:, None] * ds[None, :]) / s_j])
            B_j = np.block([[np.eye(m), -H / s_j],
                            [-H.T / s_j, np.eye(ncol)]])
            duv = np.linalg.pinv(B_j) @ rhs                       # (m+ncol, n_b)
            dU[j] = duv[:m]
            dV[j] = duv[m:]
            dS[j] = ds
        return dU, dV, dS

    def _prepare_perturbation_factors(self, n_max):
        """Lazily compute the order-independent Lemma-6 triplet perturbations.

        Computes, once up to ``n_max`` triplets, the SVD perturbations of the
        first setup's full Hankel and of every subsequent setup's reference-row
        Hankel, batched over all block deviations.
        """
        if (self._pert_factors is not None
                and self._pert_factors['n_max'] >= n_max):
            return self._pert_factors

        p = self.num_block_rows
        r = self.num_ref_channels

        # setup 0: full Hankel
        U1, S1, V1_T = self.svd_h1
        H0 = (U1 * S1) @ V1_T
        setup0 = self._svd_triplet_perturbations(H0, U1, S1, V1_T,
                                                 self.hankel_devs[0], n_max)

        # setups j >= 1: reference-row Hankel
        setup_refs = [None] * len(self.setups)
        for j in range(1, len(self.setups)):
            n_l = self.setup_n_l[j]
            ref_idx = self._block_row_selection(p + 1, n_l, 0, r)
            devs_ref = self.hankel_devs[j][ref_idx, :, :]
            Ur, Sr, Vr_T = self.svd_refs[j]
            Href = (Ur * Sr) @ Vr_T
            setup_refs[j] = self._svd_triplet_perturbations(
                Href, Ur, Sr, Vr_T, devs_ref, n_max)

        self._pert_factors = {'n_max': n_max, 'setup0': setup0, 'setup_refs': setup_refs}
        return self._pert_factors

    # ── per-order point-estimate context and perturbations ────────────────────

    def _var_context(self, order):
        """Point-estimate intermediates required by the perturbation chain."""
        n = order
        p = self.num_block_rows
        r = self.num_ref_channels

        U1, S1, _ = self.svd_h1
        n_l0 = self.setup_n_l[0]
        O0 = U1[:, :n] * np.sqrt(S1[:n])[np.newaxis, :]
        ref0 = self._block_row_selection(p + 1, n_l0, 0, r)
        mov0 = self._block_row_selection(p + 1, n_l0, r, n_l0)
        O_ref = O0[ref0, :]

        parts = [O_ref, O0[mov0, :]]
        M = [None] * len(self.setups)          # H^(j,mov) pinv_n(H^(j,ref))
        pinvn_Oref = [None] * len(self.setups)  # pinv_n(H^(j,ref)) O_ref
        Hmov = [None] * len(self.setups)
        refsvd = [None] * len(self.setups)
        for j in range(1, len(self.setups)):
            Ur, Sr, Vr_T = self.svd_refs[j]
            H_mov = self.hankels_mov[j]
            pinv_n = Vr_T[:n, :].T @ ((1.0 / Sr[:n])[:, None] * Ur[:, :n].T)
            M[j] = H_mov @ pinv_n
            pinvn_Oref[j] = pinv_n @ O_ref
            Hmov[j] = H_mov
            refsvd[j] = (Ur, Sr, Vr_T)
            parts.append(H_mov @ pinvn_Oref[j])

        O_up, O_down, C = self._assemble_up_down_c(parts, p)
        A = np.linalg.pinv(O_up) @ O_down

        return {'U1': U1, 'S1': S1, 'O_ref': O_ref, 'parts': parts, 'M': M,
                'pinvn_Oref': pinvn_Oref, 'Hmov': Hmov, 'refsvd': refsvd,
                'O_up': O_up, 'O_down': O_down, 'C': C, 'A': A}

    @staticmethod
    def _assemble_up_down_c(parts, p):
        """Stack the shifted-up/shifted-down/first-block-row parts.

        Works for both 2-D observability parts ``(rows, n)`` and 3-D
        perturbation tensors ``(rows, n, K)`` (stacking is along axis 0).
        """
        up, down, cb = [], [], []
        for part in parts:
            rows = part.shape[0]
            w = rows // (p + 1)
            up.append(part[:p * w])
            down.append(part[w:(p + 1) * w])
            cb.append(part[:w])
        return np.vstack(up), np.vstack(down), np.vstack(cb)

    def _observability_perturbations(self, order, ctx, factors):
        """Perturbation tensors ``ΔO_up, ΔO_down, ΔC`` over all K columns.

        Returns arrays of shape ``(rows, order, K)`` where ``K = sum_j n_b_j``.
        """
        n = order
        p = self.num_block_rows
        r = self.num_ref_channels
        U1, S1 = ctx['U1'], ctx['S1']
        O_ref = ctx['O_ref']
        sqrt_s = np.sqrt(S1[:n])

        up_groups, down_groups, c_groups = [], [], []

        # ---- setup-0 columns: perturb every part ----
        dU0, _, dS0 = factors['setup0']
        n_l0 = self.setup_n_l[0]
        ref0 = self._block_row_selection(p + 1, n_l0, 0, r)
        mov0 = self._block_row_selection(p + 1, n_l0, r, n_l0)
        # ΔO^(0)[:, j, k] = dU0[j,:,k] sqrt(σ_j) + u_j dσ_j,k / (2 sqrt(σ_j))
        dO0 = (np.transpose(dU0[:n], (1, 0, 2)) * sqrt_s[None, :, None]
               + U1[:, :n][:, :, None] * (dS0[:n] / (2 * sqrt_s[:, None]))[None, :, :])
        dO_ref = dO0[ref0, :, :]
        dparts = [dO_ref, dO0[mov0, :, :]]
        for j in range(1, len(self.setups)):
            dparts.append(np.einsum('ab,bnk->ank', ctx['M'][j], dO_ref))
        up0, down0, c0 = self._assemble_up_down_c(dparts, p)
        up_groups.append(up0)
        down_groups.append(down0)
        c_groups.append(c0)

        # ---- setup-j>=1 columns: perturb only part j ----
        for s in range(1, len(self.setups)):
            n_b = self.setup_n_b[s]
            n_l = self.setup_n_l[s]
            mov_idx = self._block_row_selection(p + 1, n_l, r, n_l)
            devs_mov = self.hankel_devs[s][mov_idx, :, :]        # (mov, ncol, n_b)
            Ur, Sr, Vr_T = ctx['refsvd'][s]
            dUr, dVr, dSr = factors['setup_refs'][s]

            # Δ(pinv_n) O_ref  (sum over triplets)  -> (q*r, n, n_b)
            dpinv_Oref = np.zeros((Vr_T.shape[1], n, n_b))
            for jj in range(n):
                u_j = Ur[:, jj]
                v_j = Vr_T[jj]
                s_j = Sr[jj]
                uO = u_j @ O_ref                                 # (n,)
                duO = np.einsum('mk,mn->nk', dUr[jj], O_ref)     # (n, n_b)
                dpinv_Oref += (
                    -(dSr[jj] / s_j ** 2)[None, None, :] * np.outer(v_j, uO)[:, :, None]
                    + (1.0 / s_j) * np.einsum('ck,n->cnk', dVr[jj], uO)
                    + (1.0 / s_j) * np.einsum('c,nk->cnk', v_j, duO))

            dO_smov = (np.einsum('mck,cn->mnk', devs_mov, ctx['pinvn_Oref'][s])
                       + np.einsum('mc,cnk->mnk', ctx['Hmov'][s], dpinv_Oref))

            dparts = []
            for pi, part in enumerate(ctx['parts']):
                if pi == s + 1:
                    dparts.append(dO_smov)
                else:
                    dparts.append(np.zeros((part.shape[0], n, n_b)))
            up_s, down_s, c_s = self._assemble_up_down_c(dparts, p)
            up_groups.append(up_s)
            down_groups.append(down_s)
            c_groups.append(c_s)

        dO_up = np.concatenate(up_groups, axis=2)
        dO_down = np.concatenate(down_groups, axis=2)
        dC = np.concatenate(c_groups, axis=2)
        return dO_up, dO_down, dC

    # ── per-mode sensitivities ────────────────────────────────────────────────

    @staticmethod
    def _jacobian_freq_damp(lambda_i, sampling_rate):
        """2x2 Jacobian mapping ``[Re Δλ, Im Δλ]`` to ``[Δf, Δξ]`` (Eq. 5)."""
        a_i = np.abs(np.arctan2(lambda_i.imag, lambda_i.real))
        b_i = np.log(np.abs(lambda_i))
        tlambda = (b_i + 1j * a_i) * sampling_rate
        return (sampling_rate / ((np.abs(lambda_i) ** 2) * np.abs(tlambda))
                * np.array([[1 / (2 * np.pi), 0],
                            [0, 100 / (np.abs(tlambda) ** 2)]])
                @ np.array([[tlambda.real, tlambda.imag],
                            [-(tlambda.imag ** 2), tlambda.real * tlambda.imag]])
                @ np.array([[lambda_i.real, lambda_i.imag],
                            [-lambda_i.imag, lambda_i.real]]))

    # ── mode-shape integration + normalisation ────────────────────────────────

    def _integration_factors(self, omega):
        """Diagonal integration factor ``D(omega)`` and its omega-derivative.

        ``D`` maps modal accelerations/velocities to modal displacements
        (accel: ``-1/omega**2``, velo: ``1j/omega``, disp: ``1``);
        ``Ddash = dD/domega`` (accel: ``2/omega**3``, velo: ``-1j/omega**2``).
        """
        n_l = self.num_analised_channels
        accel = self.accel_channels
        velo = self.velo_channels
        d_diag = np.ones(n_l, dtype=complex)
        d_diag[accel] = -1.0 / omega ** 2
        d_diag[velo] = 1j / omega
        ddash = np.zeros(n_l, dtype=complex)
        ddash[accel] = 2.0 / omega ** 3
        ddash[velo] = -1j / omega ** 2
        return d_diag, ddash

    def _normalize_mode_shape(self, raw, freq):
        """Integrated (displacement) mode shape, rotation-only normalised.

        Applies :meth:`~ModalBase.integrate_quantities` (as the point-estimate
        classes do), then the max-amplitude phase rotation (Prop. 5) rather than
        the unit re-scaling, keeping the variance-loop convention of
        :class:`~pyOMA.core.VarSSIRef.VarSSIRef`.
        """
        d_diag, _ = self._integration_factors(2 * np.pi * freq)
        psi_int = d_diag * raw
        kmax = np.argmax(np.abs(psi_int))
        return psi_int * np.exp(-1j * np.angle(psi_int[kmax]))

    def _mode_shape_perturbation(self, raw, dpsi_raw, freq, dfreq):
        """First-order perturbation of the integrated, rotation-only mode shape.

        Integration is ``omega``-dependent, so the perturbation carries the
        ``Delta f -> Delta omega`` coupling:
        ``d(psi_int) = D(omega) d(psi_raw) + (dD/domega . raw) . domega`` with
        ``domega = 2*pi*df``; the Prop.-5 phase-rotation derivative then follows.

        Parameters
        ----------
        raw : (n_l,) ndarray -- un-integrated mode shape ``C @ phi``.
        dpsi_raw : (n_l, K) ndarray -- its per-column perturbation.
        freq : float -- modal frequency [Hz].
        dfreq : (K,) ndarray -- per-column frequency perturbation.

        Returns
        -------
        (n_l, K) ndarray -- perturbation of the normalised displacement shape.
        """
        omega = 2 * np.pi * freq
        d_diag, ddash = self._integration_factors(omega)
        domega = 2 * np.pi * dfreq
        dpsi_int = d_diag[:, None] * dpsi_raw + (ddash * raw)[:, None] * domega[None, :]
        psi_int = d_diag * raw
        kmax = np.argmax(np.abs(psi_int))
        s = psi_int[kmax]
        t = np.abs(s)
        corr = np.imag(np.conj(s) * dpsi_int[kmax, :])
        return np.exp(-1j * np.angle(s)) * (
            dpsi_int + (-1j * t ** -2) * psi_int[:, None] * corr[None, :])

    # ── compute_modal_params ──────────────────────────────────────────────────

    def compute_modal_params(self, max_model_order=None, orders=None):  # pylint: disable=arguments-differ
        """Multi-order modal identification with first-order uncertainties.

        Parameters
        ----------
        max_model_order : int, optional
            Maximum model order (capped at ``min((p+1)*r, q*r)``).
        orders : iterable of int, optional
            Explicit model orders to evaluate; defaults to ``1 .. max_model_order-1``.
        """
        if not self.state[0]:
            raise RuntimeError('Call build_subspace_matrices() before compute_modal_params().')

        order_cap = self.max_model_order
        if max_model_order is None:
            max_model_order = order_cap
        elif not isinstance(max_model_order, int):
            raise TypeError(
                f"Expected int for 'max_model_order', got {type(max_model_order).__name__!r}.")
        elif max_model_order > order_cap:
            raise ValueError(f"max_model_order must be <= {order_cap}, got {max_model_order}.")

        if orders is None:
            orders = range(1, max_model_order)
        else:
            orders = [int(o) for o in orders]
            if max(orders) >= max_model_order:
                raise ValueError(f"orders must be < {max_model_order}.")

        n_l = self.num_analised_channels
        fs = self.sampling_rate

        factors = self._prepare_perturbation_factors(max_model_order)

        modal_frequencies = np.zeros((max_model_order, max_model_order))
        modal_damping = np.zeros((max_model_order, max_model_order))
        eigenvalues = np.zeros((max_model_order, max_model_order), dtype=complex)
        mode_shapes = np.zeros((n_l, max_model_order, max_model_order), dtype=complex)
        std_frequencies = np.zeros((max_model_order, max_model_order))
        std_damping = np.zeros((max_model_order, max_model_order))
        std_mode_shapes = np.zeros((n_l, max_model_order, max_model_order), dtype=complex)

        logger.info('Computing modal parameters and variances...')
        pbar = simplePbar(len(list(orders)))
        for order in orders:
            next(pbar)
            self._compute_order_variance(
                order, factors, fs,
                modal_frequencies, modal_damping, eigenvalues, mode_shapes,
                std_frequencies, std_damping, std_mode_shapes)

        self.max_model_order = max_model_order
        self.modal_frequencies = modal_frequencies
        self.modal_damping = modal_damping
        self.eigenvalues = eigenvalues
        self.mode_shapes = mode_shapes
        self.std_frequencies = std_frequencies
        self.std_damping = std_damping
        self.std_mode_shapes = std_mode_shapes

        self.state[2] = True
        self.state[4] = True

    def _compute_order_variance(self, order, factors, fs, modal_frequencies,
                                modal_damping, eigenvalues, mode_shapes,
                                std_frequencies, std_damping, std_mode_shapes):
        """Fill one model order's point estimates and standard deviations."""
        ctx = self._var_context(order)
        A, C = ctx['A'], ctx['C']

        # left+right eigenvectors; index physical modes directly to avoid the
        # eigvec_l/eigvec_r swap ambiguity of remove_conjugates' vector return.
        w, vl, vr = scipy.linalg.eig(A, left=True, right=True)
        phys = self.remove_conjugates(w, vr, vl, inds_only=True)

        n = order
        n_l = self.num_analised_channels
        eye_n = np.eye(n)
        # per-order arrays with nan padding for non-physical slots, mirroring
        # PreGERSSI.modal_analysis so the arrays match element-wise.
        f_row = np.full(order, np.nan)
        d_row = np.full(order, np.nan)
        e_row = np.full(order, np.nan, dtype=complex)
        sf_row = np.full(order, np.nan)
        sd_row = np.full(order, np.nan)
        psi_row = np.full((n_l, order), np.nan, dtype=complex)
        spsi_row = np.full((n_l, order), np.nan, dtype=complex)

        if phys:
            dO_up, dO_down, dC = self._observability_perturbations(order, ctx, factors)
            # ΔA_k = Kinv [ ΔO_up^T (O_down - O_up A) + O_up^T (ΔO_down - ΔO_up A) ]
            Kinv = np.linalg.inv(ctx['O_up'].T @ ctx['O_up'])
            resid = ctx['O_down'] - ctx['O_up'] @ A
            term_a = np.einsum('rnK,rm->nmK', dO_up, resid)
            inner = dO_down - np.einsum('rnK,nm->rmK', dO_up, A)
            term_b = np.einsum('rn,rmK->nmK', ctx['O_up'], inner)
            dA = np.einsum('ij,jmK->imK', Kinv, term_a + term_b)  # (n, n, K)

        for idx, i in enumerate(phys):
            lam = w[i]
            phi = vr[:, i]
            chi = vl[:, i]
            a_i = np.abs(np.arctan2(lam.imag, lam.real))
            b_i = np.log(np.abs(lam))
            freq_i = np.sqrt(a_i ** 2 + b_i ** 2) * fs / 2 / np.pi
            f_row[idx] = freq_i
            d_row[idx] = 100 * np.abs(b_i) / np.sqrt(a_i ** 2 + b_i ** 2)
            e_row[idx] = lam

            raw = C @ phi
            psi_row[:raw.shape[0], idx] = self._normalize_mode_shape(raw, freq_i)

            denom = np.dot(chi.conj(), phi)
            dAphi = np.einsum('imK,m->iK', dA, phi)              # (n, K)
            dlam = (chi.conj() @ dAphi) / denom                 # (K,)
            dfd = self._jacobian_freq_damp(lam, fs) @ np.vstack([dlam.real, dlam.imag])
            sf_row[idx] = np.sqrt(np.sum(dfd[0] ** 2))
            sd_row[idx] = np.sqrt(np.sum(dfd[1] ** 2))

            proj = eye_n - np.outer(phi, chi.conj()) / denom
            pinv_la = np.linalg.pinv(lam * eye_n - A)
            dphi = pinv_la @ (proj @ dAphi)                     # (n, K)
            dpsi_raw = C @ dphi + np.einsum('rmK,m->rK', dC, phi)
            # integrate to displacement (Delta f -> Delta omega coupling) then normalise
            dpsi = self._mode_shape_perturbation(raw, dpsi_raw, freq_i, dfd[0])
            var_pr = np.sum(dpsi.real ** 2, axis=1)
            var_pi = np.sum(dpsi.imag ** 2, axis=1)
            spsi_row[:var_pr.shape[0], idx] = np.sqrt(var_pr) + 1j * np.sqrt(var_pi)

        argsort = np.argsort(f_row)
        modal_frequencies[order, :order] = f_row[argsort]
        modal_damping[order, :order] = d_row[argsort]
        eigenvalues[order, :order] = e_row[argsort]
        std_frequencies[order, :order] = sf_row[argsort]
        std_damping[order, :order] = sd_row[argsort]
        mode_shapes[:, :order, order] = psi_row[:, argsort]
        std_mode_shapes[:, :order, order] = spsi_row[:, argsort]

    # ── persistence ───────────────────────────────────────────────────────────

    def save_state(self, fname):
        """Save state including standard deviations; perturbation caches are not persisted."""
        super().save_state(fname)
        if self.state[4]:
            path = fname if fname.endswith('.npz') else fname + '.npz'
            in_dict = dict(np.load(path, allow_pickle=True))
            in_dict['self.std_frequencies'] = self.std_frequencies
            in_dict['self.std_damping'] = self.std_damping
            in_dict['self.std_mode_shapes'] = self.std_mode_shapes
            in_dict['self.num_blocks'] = self.num_blocks
            np.savez_compressed(path, **in_dict)

    @classmethod
    def load_state(cls, fname):  # pylint: disable=arguments-differ
        """Restore a :class:`VarPreGERSSI` (view only; caches are not recomputed)."""
        ssi_object = super().load_state(fname)
        in_dict = np.load(fname, allow_pickle=True)
        if ssi_object.state[4]:
            ssi_object.std_frequencies = validate_array(in_dict['self.std_frequencies'])
            ssi_object.std_damping = validate_array(in_dict['self.std_damping'])
            ssi_object.std_mode_shapes = validate_array(in_dict['self.std_mode_shapes'])
            ssi_object.num_blocks = int(validate_array(in_dict['self.num_blocks']))
        return ssi_object
