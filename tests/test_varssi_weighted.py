# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the weighted / externally-fed subspace estimation in VarSSIRef.

The weighted estimator generalizes the block averaging of the
covariance-driven method: ``weights=None`` must reproduce the classical
unweighted results, non-uniform weights switch mean and Hankel covariance to
their importance-weighted counterparts (Kish effective sample size), and
``corr_matrices`` feeds standalone per-block correlation estimates in place
of the attached signals.
"""

import numpy as np
import pytest

from pyOMA.core.VarSSIRef import VarSSIRef

NUM_BLOCK_COLUMNS = 20
NUM_BLOCKS = 4


def _build(prep_signals, **kwargs):
    obj = VarSSIRef(prep_signals)
    obj.build_subspace_mat(
        num_block_columns=kwargs.pop('num_block_columns', NUM_BLOCK_COLUMNS),
        subspace_method='covariance',
        **kwargs)
    return obj


def _synthetic_corr(prep_signals, block_values, m_lags):
    """Constant-valued correlation estimates: block n is all ``block_values[n]``."""
    n_l = prep_signals.num_analised_channels
    n_r = prep_signals.num_ref_channels
    return np.array(
        [np.full((n_l, n_r, m_lags), val) for val in block_values])


class TestWeightValidation:

    def test_none_is_uniform(self):
        weights, n_eff = VarSSIRef._validate_weights(None, 5)
        assert weights is None
        assert n_eff == 5.0

    def test_renormalization_and_neff(self):
        weights, n_eff = VarSSIRef._validate_weights([2.0, 1.0, 1.0], 3)
        assert np.allclose(weights, [0.5, 0.25, 0.25])
        assert np.isclose(n_eff, 1.0 / (0.25 + 0.0625 + 0.0625))

    def test_negative_raises(self):
        with pytest.raises(ValueError):
            VarSSIRef._validate_weights([-0.1, 1.1], 2)

    def test_all_zero_raises(self):
        with pytest.raises(ValueError):
            VarSSIRef._validate_weights([0.0, 0.0], 2)

    def test_wrong_length_raises(self):
        with pytest.raises(ValueError):
            VarSSIRef._validate_weights([1.0, 1.0, 1.0], 2)

    def test_wrong_length_via_build(self, prep_signals_with_corr):
        with pytest.raises(ValueError):
            _build(prep_signals_with_corr, num_blocks=NUM_BLOCKS,
                   weights=np.ones(NUM_BLOCKS + 1))

    def test_projection_method_accepts_weights_by_default(self, prep_signals_with_corr):
        """Since the weighted-mean projection build (TestWeightedProjection),
        'weights' with subspace_method='projection' no longer requires the
        experimental flag -- it now selects the (variance-capable) weighted
        mean of unweighted per-block projections."""
        obj = VarSSIRef(prep_signals_with_corr)
        obj.build_subspace_mat(
            num_block_columns=10, num_blocks=NUM_BLOCKS,
            subspace_method='projection', weights=np.ones(NUM_BLOCKS))
        assert obj.weights is not None
        assert not obj.experimental_weighted_projection


class TestToyWeightedMoments:
    """Hand-computable checks on synthetic constant-valued correlation blocks."""

    M_LAGS_TOY = 7  # num_block_rows + 1 + num_block_columns with 3 block columns
    NBC_TOY = 3

    def _toy(self, prep_signals, block_values, weights):
        corr = _synthetic_corr(prep_signals, block_values, self.M_LAGS_TOY)
        obj = VarSSIRef(prep_signals)
        obj.build_subspace_mat(
            num_block_columns=self.NBC_TOY, subspace_method='covariance',
            weights=weights, corr_matrices=corr)
        return obj

    def test_weighted_mean(self, prep_signals_with_corr):
        block_values = [1.0, 2.0, 4.0]
        weights = [0.5, 0.3, 0.2]
        obj = self._toy(prep_signals_with_corr, block_values, weights)
        expected = np.dot(weights, block_values)
        assert np.allclose(obj.subspace_matrix, expected)
        assert np.isclose(obj.n_eff, 1.0 / np.sum(np.square(weights)))

    def test_weighted_hankel_covariance(self, prep_signals_with_corr):
        block_values = np.array([1.0, 2.0, 4.0])
        weights = np.array([0.5, 0.3, 0.2])
        obj = self._toy(prep_signals_with_corr, block_values, weights)
        n_l = prep_signals_with_corr.num_analised_channels
        n_r = prep_signals_with_corr.num_ref_channels
        T = obj._compute_hankel_cov_matrix(
            self.NBC_TOY, self.NBC_TOY, n_l, n_r, len(block_values))

        # Independent computation from the derived formula: every entry of
        # block n equals block_values[n], so each vectorized deviation is a
        # constant vector and the covariance is scalar * ones.
        mean = np.dot(weights, block_values)
        n_eff = 1.0 / np.sum(weights ** 2)
        s_w = np.sum(weights * (block_values - mean) ** 2)
        expected_cov_entry = s_w / (n_eff * (n_eff - 1))
        assert np.allclose(T @ T.T, expected_cov_entry)

    def test_zero_weight_elimination(self, prep_signals_with_corr):
        """Zero-weighted blocks must drop out exactly (Imprecision elimination)."""
        obj4 = self._toy(prep_signals_with_corr, [1.0, 3.0, 100.0, -50.0],
                         [0.5, 0.5, 0.0, 0.0])
        obj2 = self._toy(prep_signals_with_corr, [1.0, 3.0], [0.5, 0.5])
        assert np.allclose(obj4.subspace_matrix, obj2.subspace_matrix)
        assert np.isclose(obj4.n_eff, obj2.n_eff)

        n_l = prep_signals_with_corr.num_analised_channels
        n_r = prep_signals_with_corr.num_ref_channels
        T4 = obj4._compute_hankel_cov_matrix(self.NBC_TOY, self.NBC_TOY, n_l, n_r, 4)
        T2 = obj2._compute_hankel_cov_matrix(self.NBC_TOY, self.NBC_TOY, n_l, n_r, 2)
        assert np.allclose(T4 @ T4.T, T2 @ T2.T)

    def test_neff_guard_zeroes_covariance(self, prep_signals_with_corr, caplog):
        obj = self._toy(prep_signals_with_corr, [1.0, 2.0, 4.0], [1.0, 0.0, 0.0])
        assert obj.n_eff == 1.0
        n_l = prep_signals_with_corr.num_analised_channels
        n_r = prep_signals_with_corr.num_ref_channels
        with caplog.at_level('WARNING', logger='pyOMA.core.VarSSIRef'):
            T = obj._compute_hankel_cov_matrix(
                self.NBC_TOY, self.NBC_TOY, n_l, n_r, 3)
        assert np.all(T == 0)
        assert any('n_eff' in rec.message for rec in caplog.records)

    def test_corr_matrices_shape_validation(self, prep_signals_with_corr):
        n_l = prep_signals_with_corr.num_analised_channels
        n_r = prep_signals_with_corr.num_ref_channels
        obj = VarSSIRef(prep_signals_with_corr)
        with pytest.raises(ValueError):  # too few lags for the dimensions
            obj.build_subspace_mat(
                num_block_columns=10, subspace_method='covariance',
                corr_matrices=np.zeros((3, n_l, n_r, 5)))
        with pytest.raises(ValueError):  # contradicting num_blocks
            obj.build_subspace_mat(
                num_block_columns=self.NBC_TOY, num_blocks=5,
                subspace_method='covariance',
                corr_matrices=np.zeros((3, n_l, n_r, self.M_LAGS_TOY)))


@pytest.mark.slow
class TestWeightedPipeline:
    """Full-pipeline equivalence and external-corr runs on synthetic signals."""

    MAX_ORDER = 12

    def _pipeline(self, prep_signals, **build_kwargs):
        obj = _build(prep_signals, **build_kwargs)
        obj.compute_state_matrices(max_model_order=self.MAX_ORDER, lsq_method='pinv')
        obj.prepare_sensitivities(variance_algo='fast')
        obj.compute_modal_params()
        return obj

    def test_uniform_weights_reduce_to_unweighted(self, prep_signals_with_corr):
        ref = self._pipeline(prep_signals_with_corr, num_blocks=NUM_BLOCKS)
        uni = self._pipeline(
            prep_signals_with_corr, num_blocks=NUM_BLOCKS,
            weights=np.full(NUM_BLOCKS, 1.0 / NUM_BLOCKS))
        assert np.allclose(uni.subspace_matrix, ref.subspace_matrix, rtol=1e-10)
        assert np.allclose(uni.hankel_cov_matrix, ref.hankel_cov_matrix, rtol=1e-8)
        assert np.allclose(uni.modal_frequencies, ref.modal_frequencies,
                           rtol=1e-8, atol=1e-10)
        assert np.allclose(uni.std_frequencies, ref.std_frequencies,
                           rtol=1e-6, atol=1e-12)
        assert np.allclose(uni.std_damping, ref.std_damping,
                           rtol=1e-6, atol=1e-12)

    def test_external_corr_bypasses_prep_signals(
            self, prep_signals_with_corr, monkeypatch):
        corr = np.copy(prep_signals_with_corr.corr_matrices)

        def _forbidden(*args, **kwargs):
            raise AssertionError('prep_signals.correlation must not be called')

        monkeypatch.setattr(prep_signals_with_corr, 'correlation', _forbidden)
        obj = self._pipeline(
            prep_signals_with_corr, corr_matrices=corr,
            weights=np.linspace(1.0, 2.0, corr.shape[0]))
        assert obj.external_corr
        assert obj.modal_frequencies.shape == (self.MAX_ORDER, self.MAX_ORDER)
        std_f = obj.std_frequencies[~np.isnan(obj.std_frequencies)]
        assert np.all(std_f >= 0)

    def test_slow_algo_rejected_for_weighted(self, prep_signals_with_corr):
        obj = _build(prep_signals_with_corr, num_blocks=NUM_BLOCKS,
                     weights=np.array([0.4, 0.3, 0.2, 0.1]))
        obj.compute_state_matrices(max_model_order=self.MAX_ORDER, lsq_method='pinv')
        with pytest.raises(NotImplementedError):
            obj.prepare_sensitivities(variance_algo='slow')

    def test_orders_parameter(self, prep_signals_with_corr):
        obj = _build(prep_signals_with_corr, num_blocks=NUM_BLOCKS)
        obj.compute_state_matrices(max_model_order=self.MAX_ORDER, lsq_method='pinv')
        obj.prepare_sensitivities(variance_algo='fast')
        obj.compute_modal_params(orders=[8])
        assert np.any(obj.modal_frequencies[8, :] > 0)
        untouched = np.delete(obj.modal_frequencies, 8, axis=0)
        assert np.all(untouched == 0)
        with pytest.raises(ValueError):
            obj.compute_modal_params(orders=[self.MAX_ORDER])

    def test_save_load_round_trip_with_weights(
            self, prep_signals_with_corr, tmp_path):
        weights = np.array([0.4, 0.3, 0.2, 0.1])
        obj = _build(prep_signals_with_corr, num_blocks=NUM_BLOCKS, weights=weights)
        fname = tmp_path / 'weighted.npz'
        obj.save_state(str(fname))
        loaded = VarSSIRef.load_state(str(fname), prep_signals_with_corr)
        assert np.allclose(loaded.weights, weights)
        assert np.isclose(loaded.n_eff, obj.n_eff)
        assert loaded.external_corr == obj.external_corr

    def test_load_legacy_archive_without_weights(
            self, prep_signals_with_corr, tmp_path):
        obj = _build(prep_signals_with_corr, num_blocks=NUM_BLOCKS)
        fname = tmp_path / 'legacy.npz'
        obj.save_state(str(fname))
        # An unweighted save has no 'self.weights' key, exactly like archives
        # predating the feature.
        with np.load(str(fname), allow_pickle=True) as arch:
            assert 'self.weights' not in arch
        loaded = VarSSIRef.load_state(str(fname), prep_signals_with_corr)
        assert loaded.weights is None
        assert loaded.n_eff == NUM_BLOCKS


WP_NBC = 15
WP_NB = 6
WP_MO = 8
WP_WEIGHTS = np.array([0.4, 0.25, 0.15, 0.1, 0.06, 0.04])


def _build_proj(prep_signals, weights=None, flag=False, num_blocks=None):
    obj = VarSSIRef(prep_signals)
    obj.build_subspace_mat(
        num_block_columns=WP_NBC, num_blocks=num_blocks or WP_NB,
        subspace_method='projection', weights=weights,
        experimental_weighted_projection=flag)
    return obj


def _modal_proj(obj, variance_algo='none'):
    obj.compute_state_matrices(max_model_order=WP_MO)
    obj.prepare_sensitivities(variance_algo=variance_algo)
    obj.compute_modal_params()
    return obj


def _physical_freqs(obj, order=4):
    f = obj.modal_frequencies[order]
    return np.sort(f[f > 0])


class TestWeightedProjection:
    """Build-time weighted projection (weighted UPC), two readings selected
    by ``experimental_weighted_projection``:

    * ``False`` (default): every block's own per-block/joint-LQ projection is
      built exactly as in the unweighted case; weights only pick a weighted
      mean over the (independent, unweighted) per-block estimates, the same
      structure as the covariance method -- so ``variance_algo='fast'``
      works normally.
    * ``True`` (experimental, Phase 5): each block Hankel matrix is scaled by
      ``w_j`` in place of the uniform ``1/num_blocks`` *before* the per-block
      LQ decompositions, so both the per-block and the joint ``R11``
      normalization see the weights.  Point estimates only: variance
      preparation is rejected (``variance_algo='none'`` provides the
      point-estimates-only modal run).
    """

    def test_flag_rejected_for_covariance(self, prep_signals_with_corr):
        obj = VarSSIRef(prep_signals_with_corr)
        with pytest.raises(ValueError):
            obj.build_subspace_mat(
                num_block_columns=WP_NBC, num_blocks=WP_NB,
                subspace_method='covariance',
                experimental_weighted_projection=True)

    def test_uniform_regression(self, prep_signals_with_corr):
        """Uniform weights reproduce the unweighted projection build, under
        either weighting reading."""
        ref = _modal_proj(_build_proj(prep_signals_with_corr))
        for flag in (False, True):
            uni = _modal_proj(_build_proj(
                prep_signals_with_corr,
                weights=np.full(WP_NB, 1.0 / WP_NB), flag=flag))
            assert np.allclose(uni.subspace_matrix, ref.subspace_matrix,
                               rtol=1e-10, atol=1e-14)
            assert np.allclose(uni.modal_frequencies, ref.modal_frequencies,
                               rtol=1e-8, atol=1e-10)
            assert np.allclose(uni.modal_damping, ref.modal_damping,
                               rtol=1e-8, atol=1e-8)
            assert np.allclose(uni.mode_shapes, ref.mode_shapes,
                               rtol=1e-7, atol=1e-10)
            assert np.allclose(uni.weights, 1.0 / WP_NB)
            assert np.isclose(uni.n_eff, WP_NB)


class TestWeightedProjectionExperimental:
    """experimental_weighted_projection=True: point estimates only."""

    def test_variance_prep_rejected(self, prep_signals_with_corr):
        obj = _build_proj(prep_signals_with_corr, weights=WP_WEIGHTS, flag=True)
        obj.compute_state_matrices(max_model_order=WP_MO)
        for algo in ('fast', 'slow'):
            with pytest.raises(NotImplementedError):
                obj.prepare_sensitivities(variance_algo=algo)

    def test_point_estimates_only_run_has_zero_std(self, prep_signals_with_corr):
        obj = _modal_proj(_build_proj(
            prep_signals_with_corr, weights=WP_WEIGHTS, flag=True))
        assert np.any(obj.modal_frequencies > 0)
        assert np.all(obj.std_frequencies == 0)
        assert np.all(obj.std_damping == 0)
        assert np.all(obj.std_mode_shapes == 0)
        # Tier A / Tier B post-hoc APIs are unavailable on this build
        with pytest.raises(NotImplementedError):
            obj.compute_modal_params_weighted(None)
        with pytest.raises(RuntimeError):
            obj.apply_block_weights(None)

    def test_nonuniform_weights_change_subspace(self, prep_signals_with_corr):
        ref = _build_proj(prep_signals_with_corr)
        wgt = _build_proj(prep_signals_with_corr, weights=WP_WEIGHTS, flag=True)
        assert not np.allclose(wgt.subspace_matrix, ref.subspace_matrix)

    def test_weighted_projection_matches_weighted_covariance(
            self, prep_signals_with_corr):
        """Weighted projection and weighted covariance identify the same
        physical modes within statistical scatter (brief Phase 5 gate)."""
        proj = _modal_proj(_build_proj(
            prep_signals_with_corr, weights=WP_WEIGHTS, flag=True))

        cov = VarSSIRef(prep_signals_with_corr)
        cov.build_subspace_mat(
            num_block_columns=WP_NBC, num_blocks=WP_NB,
            subspace_method='covariance', weights=WP_WEIGHTS)
        cov.compute_state_matrices(max_model_order=WP_MO)
        cov.prepare_sensitivities(variance_algo='fast')
        cov.compute_modal_params()

        f_proj = _physical_freqs(proj)
        f_cov = _physical_freqs(cov)
        assert len(f_proj) == len(f_cov) == 2
        assert np.allclose(f_proj, f_cov, rtol=0.02)


class TestWeightedProjectionDefault:
    """experimental_weighted_projection=False (default): weighted mean of
    independent unweighted per-block projections -- variance-capable, the
    structure a build-time-weighted point estimate with real uncertainty
    relies on (feeds method='sd' in uq_oma_a's UQ_OMA_weighted.py)."""

    def test_experimental_flag_defaults_false(self, prep_signals_with_corr):
        obj = _build_proj(prep_signals_with_corr, weights=WP_WEIGHTS)
        assert obj.experimental_weighted_projection is False

    def test_per_block_projections_stay_unweighted(self, prep_signals_with_corr):
        """self.subspace_matrices (the per-block list) must equal the
        unweighted build's -- only the final combination differs."""
        ref = _build_proj(prep_signals_with_corr)
        wgt = _build_proj(prep_signals_with_corr, weights=WP_WEIGHTS)
        for h_ref, h_wgt in zip(ref.subspace_matrices, wgt.subspace_matrices):
            assert np.allclose(h_ref, h_wgt, rtol=1e-10, atol=1e-14)

    def test_subspace_matrix_is_weighted_mean(self, prep_signals_with_corr):
        ref = _build_proj(prep_signals_with_corr)
        wgt = _build_proj(prep_signals_with_corr, weights=WP_WEIGHTS)
        expected = np.tensordot(WP_WEIGHTS, ref.subspace_matrices, axes=(0, 0))
        assert np.allclose(wgt.subspace_matrix, expected, rtol=1e-10, atol=1e-14)
        assert not np.allclose(wgt.subspace_matrix, ref.subspace_matrix)

    def test_fast_variance_available(self, prep_signals_with_corr):
        obj = _modal_proj(
            _build_proj(prep_signals_with_corr, weights=WP_WEIGHTS),
            variance_algo='fast')
        assert np.any(obj.modal_frequencies > 0)
        std_f = obj.std_frequencies[~np.isnan(obj.std_frequencies)]
        std_d = obj.std_damping[~np.isnan(obj.std_damping)]
        assert np.any(std_f > 0)
        assert np.all(std_f >= 0)
        assert np.all(std_d >= 0)

    def test_slow_variance_still_rejected(self, prep_signals_with_corr):
        """The blanket 'slow' rejection for any weighted/external build is
        unrelated to the projection-specific guard and still applies."""
        obj = _build_proj(prep_signals_with_corr, weights=WP_WEIGHTS)
        obj.compute_state_matrices(max_model_order=WP_MO)
        with pytest.raises(NotImplementedError):
            obj.prepare_sensitivities(variance_algo='slow')

    def test_posthoc_reweighting_still_requires_unweighted_build(
            self, prep_signals_with_corr):
        obj = _modal_proj(
            _build_proj(prep_signals_with_corr, weights=WP_WEIGHTS),
            variance_algo='fast')
        with pytest.raises(NotImplementedError):
            obj.compute_modal_params_weighted(None)

    def test_weighted_projection_matches_weighted_covariance(
            self, prep_signals_with_corr):
        """Same physical-mode cross-check as the experimental reading, for
        the default (variance-capable) weighted-mean build."""
        proj = _modal_proj(
            _build_proj(prep_signals_with_corr, weights=WP_WEIGHTS),
            variance_algo='fast')

        cov = VarSSIRef(prep_signals_with_corr)
        cov.build_subspace_mat(
            num_block_columns=WP_NBC, num_blocks=WP_NB,
            subspace_method='covariance', weights=WP_WEIGHTS)
        cov.compute_state_matrices(max_model_order=WP_MO)
        cov.prepare_sensitivities(variance_algo='fast')
        cov.compute_modal_params()

        f_proj = _physical_freqs(proj)
        f_cov = _physical_freqs(cov)
        assert len(f_proj) == len(f_cov) == 2
        assert np.allclose(f_proj, f_cov, rtol=0.02)

    def test_save_load_round_trip(self, prep_signals_with_corr, tmp_path):
        obj = _modal_proj(
            _build_proj(prep_signals_with_corr, weights=WP_WEIGHTS),
            variance_algo='fast')
        fname = tmp_path / 'weighted_projection.npz'
        obj.save_state(str(fname))
        loaded = VarSSIRef.load_state(str(fname), prep_signals_with_corr)
        assert loaded.experimental_weighted_projection is False
        assert np.allclose(loaded.weights, obj.weights)
        assert np.allclose(loaded.subspace_matrix, obj.subspace_matrix)


def _raw_hankel_blocks(prep_signals, num_block_rows, num_blocks):
    """Replicates VarSSIRef._build_subspace_projection's internal signal
    splitting (before the 1/(sqrt(block_length)*num_blocks) scaling), to
    build an equivalent external ``hankel_matrices=`` input for round-trip
    testing against the internal (prep_signals-driven) build."""
    total_time_steps = prep_signals.total_time_steps
    measurement = prep_signals.signals
    ref_channels = sorted(prep_signals.ref_channels)
    n_l = prep_signals.num_analised_channels
    n_r = prep_signals.num_ref_channels

    block_length = int(np.floor((total_time_steps - 2 * num_block_rows) / num_blocks))
    N = block_length * num_blocks
    Y_minus = np.zeros((num_block_rows * n_r, N))
    Y_plus = np.zeros(((num_block_rows + 1) * n_l, N))
    for ii in range(num_block_rows):
        Y_minus[(num_block_rows - ii - 1) * n_r:(num_block_rows - ii) * n_r, :] = \
            measurement[(ii):(ii + N), ref_channels].T
    for ii in range(num_block_rows + 1):
        Y_plus[ii * n_l:(ii + 1) * n_l, :] = \
            measurement[(num_block_rows + ii):(num_block_rows + ii + N)].T

    Hankel_matrix = np.vstack((Y_minus, Y_plus))
    return np.hsplit(
        Hankel_matrix, np.arange(block_length, block_length * num_blocks, block_length))


class TestExternalHankelProjection:
    """D3: external raw-Hankel input for the projection method
    (``hankel_matrices=``), analogous to ``corr_matrices=`` for the
    covariance method -- bypasses ``prep_signals.signals`` so a pipeline can
    feed one raw block-Hankel matrix per aleatory sample."""

    NBC = 15
    NB = 6
    MO = 8

    def test_round_trip_matches_internal_build(self, prep_signals_with_corr):
        blocks = _raw_hankel_blocks(prep_signals_with_corr, self.NBC, self.NB)

        ref = VarSSIRef(prep_signals_with_corr)
        ref.build_subspace_mat(num_block_columns=self.NBC, num_blocks=self.NB,
                               subspace_method='projection')

        ext = VarSSIRef(prep_signals_with_corr)
        ext.build_subspace_mat(num_block_columns=self.NBC,
                               subspace_method='projection',
                               hankel_matrices=blocks)

        assert ext.num_blocks == self.NB
        assert ext.external_corr
        assert np.allclose(ext.subspace_matrix, ref.subspace_matrix,
                           rtol=1e-10, atol=1e-14)
        for h_ext, h_ref in zip(ext.subspace_matrices, ref.subspace_matrices):
            assert np.allclose(h_ext, h_ref, rtol=1e-10, atol=1e-14)

    def test_round_trip_full_pipeline_with_variance(self, prep_signals_with_corr):
        """Unweighted external build supports fast variance and the Tier B
        post-hoc reweighting API unchanged (brief D3 item 2)."""
        blocks = _raw_hankel_blocks(prep_signals_with_corr, self.NBC, self.NB)
        ext = VarSSIRef(prep_signals_with_corr)
        ext.build_subspace_mat(num_block_columns=self.NBC,
                               subspace_method='projection',
                               hankel_matrices=blocks)
        ext.compute_state_matrices(max_model_order=self.MO, lsq_method='pinv')
        ext.prepare_sensitivities(variance_algo='fast')
        ext.compute_modal_params()
        assert np.any(ext.modal_frequencies > 0)
        std_f_before = np.copy(ext.std_frequencies)
        assert np.all(std_f_before[~np.isnan(std_f_before)] >= 0)

        w = np.linspace(1.0, 2.0, self.NB)
        w /= w.sum()
        ext.compute_modal_params_weighted(w)
        std_f_after = ext.std_frequencies
        assert np.all(std_f_after[~np.isnan(std_f_after)] >= 0)
        assert not np.allclose(std_f_before, std_f_after, equal_nan=True)

    def test_build_time_weighted_with_external_hankel_and_variance(
            self, prep_signals_with_corr):
        """The combination uq_oma_a's method='sd'+weighting='build' needs:
        externally supplied per-aleatory-sample Hankel blocks, build-time
        weights (default, non-experimental reading), and working fast
        variance -- exercised together, not just each in isolation."""
        blocks = _raw_hankel_blocks(prep_signals_with_corr, self.NBC, self.NB)
        weights = np.array([0.35, 0.25, 0.2, 0.1, 0.06, 0.04])

        ext = VarSSIRef(prep_signals_with_corr)
        ext.build_subspace_mat(num_block_columns=self.NBC,
                               subspace_method='projection',
                               hankel_matrices=blocks, weights=weights)
        assert not ext.experimental_weighted_projection
        ext.compute_state_matrices(max_model_order=self.MO, lsq_method='pinv')
        ext.prepare_sensitivities(variance_algo='fast')
        ext.compute_modal_params()

        assert np.any(ext.modal_frequencies > 0)
        std_f = ext.std_frequencies[~np.isnan(ext.std_frequencies)]
        assert np.any(std_f > 0)
        assert np.all(std_f >= 0)
        assert np.isclose(ext.n_eff, 1.0 / np.sum(weights ** 2))

    def test_shape_validation(self, prep_signals_with_corr):
        n_l = prep_signals_with_corr.num_analised_channels
        n_r = prep_signals_with_corr.num_ref_channels
        wrong_rows = self.NBC * n_r + (self.NBC + 1) * n_l - 1
        obj = VarSSIRef(prep_signals_with_corr)
        with pytest.raises(ValueError):
            obj.build_subspace_mat(
                num_block_columns=self.NBC, subspace_method='projection',
                hankel_matrices=[np.zeros((wrong_rows, 50))] * self.NB)

    def test_num_blocks_contradiction_raises(self, prep_signals_with_corr):
        blocks = _raw_hankel_blocks(prep_signals_with_corr, self.NBC, self.NB)
        obj = VarSSIRef(prep_signals_with_corr)
        with pytest.raises(ValueError):
            obj.build_subspace_mat(
                num_block_columns=self.NBC, num_blocks=self.NB + 1,
                subspace_method='projection', hankel_matrices=blocks)

    def test_covariance_rejects_hankel_matrices(self, prep_signals_with_corr):
        obj = VarSSIRef(prep_signals_with_corr)
        with pytest.raises(NotImplementedError):
            obj.build_subspace_mat(
                num_block_columns=10, subspace_method='covariance',
                hankel_matrices=[np.zeros((5, 5))])

    def test_unequal_block_lengths_allowed(self, prep_signals_with_corr):
        """Each block is scaled by its own length, unlike the internal
        path's single shared block_length; exercised with a synthetic,
        deliberately shortened last block (not a round-trip check)."""
        blocks = _raw_hankel_blocks(prep_signals_with_corr, self.NBC, self.NB)
        trimmed = list(blocks[:-1]) + [blocks[-1][:, :blocks[-1].shape[1] // 2]]
        obj = VarSSIRef(prep_signals_with_corr)
        obj.build_subspace_mat(num_block_columns=self.NBC,
                               subspace_method='projection',
                               hankel_matrices=trimmed)
        assert obj.num_blocks == self.NB
        obj.compute_state_matrices(max_model_order=self.MO, lsq_method='pinv')
        obj.prepare_sensitivities(variance_algo='fast')
        obj.compute_modal_params()
        assert np.any(obj.modal_frequencies > 0)
