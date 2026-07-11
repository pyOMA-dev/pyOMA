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

    def test_projection_method_rejects_weights(self, prep_signals_with_corr):
        obj = VarSSIRef(prep_signals_with_corr)
        with pytest.raises(NotImplementedError):
            obj.build_subspace_mat(
                num_block_columns=10, subspace_method='projection',
                weights=np.ones(NUM_BLOCKS))


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
