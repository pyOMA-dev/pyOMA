# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2015-2025  Simon Marwitz, Volkmar Zabel, Andrei Udrea et al.
"""Variance estimation for the pLSCF method.

Propagates the uncertainty of the positive half-spectra onto the identified modal
parameters by the delta method, following Steffensen, Döhler, Tcherniak & Thomsen
(MSSP 223, 2025), "Variance estimation of modal parameters from the poly-reference
least-squares complex frequency-domain algorithm".

The paper enters the propagation with the covariance of measured FRFs, obtained
from the H1 estimator and the multiple coherence (their Eq. 26-27). pyOMA is
output-only, so no input spectrum and hence no coherence exists, and that entry
point has no counterpart here. Instead the half-spectra are re-estimated block-wise
from the block correlation functions that :meth:`~pyOMA.core.PreProcessingTools.PreProcessSignals.corr_blackman_tukey`
already provides, and their sample covariance over blocks is used. This is the one
deliberate deviation from the paper's derivation; all Jacobians downstream are the
paper's.

The covariance is carried as a *factor* rather than a matrix: the block deviations
are scaled such that ``cov(h) = B B^T`` for the re/im-stacked slice ``B`` of
:attr:`VarPLSCF.spec_block_factor`, at the scale of the covariance *of the estimate*
(the role the paper's ``1/N_avg`` plays). Every downstream perturbation is then
evaluated per block and variances follow as ``einsum('ij,ij->i', U, U)``, mirroring
:mod:`~pyOMA.core.VarSSIRef`. A consequence worth noting: this retains the
correlations across frequency lines and output channels that the paper's Eq. 40
discards for tractability -- with an empirical covariance, imposing that
independence would be extra work rather than less.
"""

import numpy as np
import logging
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

from .PLSCF import PLSCF


class VarPLSCF(PLSCF):
    """pLSCF identification with uncertainty quantification.

    Extends :class:`~pyOMA.core.PLSCF.PLSCF` with standard deviations of the
    identified modal parameters.  The workflow matches the base class, except that
    :meth:`build_half_spectra` must be given *num_blocks*:

    1. :meth:`build_half_spectra` — half-spectra plus their block-covariance factor.
    2. :meth:`compute_modal_params` — modal parameters and their standard deviations.

    Attributes
    ----------
    spec_block_factor : np.ndarray or None
        Covariance factor of the positive half-spectra, complex, shape
        ``(n_l, n_r, num_omega, n_b)`` with the block index last.  Column *j* holds
        the deviation of block *j* from the block mean, scaled by
        ``1 / sqrt(n_b * (n_b - 1))``, so that contracting the re/im-stacked slice
        with itself yields the covariance of the mean half-spectrum.

        Not persisted by :meth:`save_state`: it is reconstructible from
        ``prep_signals`` and would dominate the archive size.
    """

    def __init__(self, *args, **kwargs):
        """
        Parameters
        ----------
        *args, **kwargs
            Passed to :class:`~pyOMA.core.PLSCF.PLSCF`.
        """
        super().__init__(*args, **kwargs)
        self.spec_block_factor = None

    def build_half_spectra(self, nperseg=None,
                           begin_frequency=None, end_frequency=None,
                           window_decay=0.001, num_blocks=None, training_blocks=None, **kwargs):
        '''
        Construct the positive half-spectra and their block-covariance factor.

        Behaves as :meth:`~pyOMA.core.PLSCF.PLSCF.build_half_spectra`, but
        *num_blocks* is required: the variance of the half-spectra is estimated
        from the scatter of the individual blocks about their mean, and that mean
        is the half-spectrum the model is identified from.

        Parameters
        ----------
            nperseg: integer, optional
                Number of (positive) frequency lines to consider (rfft)

            begin_frequency, end_frequency: float, optional
                Frequency range to restrict the identified system.

            window_decay: float, (0,1)
                Final value of the exponential window, that is applied to the
                correlation functions.

            num_blocks: integer, required
                The number of blocks to split the signal into. At least two
                blocks are needed for a variance estimate; the estimate becomes
                meaningful only for substantially more.

            training_blocks: list, optional
                The selected blocks to use for system identification
                (=training). Defaults to all blocks. The covariance factor is
                built from these blocks only, i.e. from the blocks that entered
                the point estimate.

        Other Parameters
        ----------------
            kwargs :
                Additional kwargs are passed to prep_signals.correlation
        '''
        if num_blocks is None:
            raise ValueError(
                'Argument num_blocks is required for variance estimation: the '
                'half-spectrum covariance is estimated from the scatter of the '
                'individual blocks. Use PLSCF for an identification without it.'
            )

        super().build_half_spectra(
            nperseg, begin_frequency, end_frequency, window_decay,
            num_blocks, training_blocks, **kwargs)

        self.spec_block_factor = self._build_spec_block_factor()

    def _build_spec_block_factor(self):
        """Build the covariance factor of the positive half-spectra.

        Returns
        -------
            spec_block_factor: (n_l, n_r, num_omega, n_b) numpy.ndarray
                Scaled block deviations; see :attr:`VarPLSCF.spec_block_factor`.
        """
        training_blocks = self.training_blocks
        n_b = training_blocks.shape[0]
        if n_b < 2:
            raise ValueError(
                f'At least two training blocks are needed to estimate a variance, got {n_b}.')
        if n_b < 10:
            logger.warning(
                f'Estimating the half-spectrum covariance from only {n_b} blocks; '
                f'the resulting standard deviations are themselves highly uncertain.')

        corr_blocks = self.prep_signals.corr_matrices_bt[training_blocks, ..., :self.nperseg]

        # the windowed rFFT is linear, so the mean of the block spectra is exactly
        # the half-spectrum the model was identified from
        _, block_spectra, _ = self._windowed_half_spectrum(
            corr_blocks, self.nperseg, self.window_decay,
            self.begin_frequency, self.end_frequency)  # (n_b, n_l, n_r, num_omega)

        deviations = block_spectra - np.mean(block_spectra, axis=0, keepdims=True)
        deviations /= np.sqrt(n_b * (n_b - 1))

        return np.moveaxis(deviations, 0, -1)
