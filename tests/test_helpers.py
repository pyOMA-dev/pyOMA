"""Unit tests for pyOMA.core.Helpers."""
import numpy as np
import pytest

from pyOMA.core.Helpers import (
    calculateMAC,
    calculateMPC,
    calculateMPD,
    calc_xyz,
    validate_array,
    rq_decomp,
    lq_decomp,
)


class TestCalculateMAC:
    def test_identical_real_vectors_give_mac_one(self):
        v = np.random.default_rng(0).standard_normal((10, 3))
        mac = calculateMAC(v, v)
        assert mac.shape == (3, 3)
        np.testing.assert_allclose(np.diag(mac), 1.0, atol=1e-12)

    def test_orthogonal_vectors_give_mac_zero(self):
        v1 = np.eye(4)          # 4×4 identity
        mac = calculateMAC(v1, v1)
        # diagonal = 1, off-diagonal = 0
        np.testing.assert_allclose(np.diag(mac), 1.0, atol=1e-12)
        off_diag = mac - np.diag(np.diag(mac))
        np.testing.assert_allclose(off_diag, 0.0, atol=1e-12)

    def test_mac_values_bounded_zero_to_one(self):
        rng = np.random.default_rng(1)
        v1 = rng.standard_normal((8, 5))
        v2 = rng.standard_normal((8, 5))
        mac = calculateMAC(v1, v2)
        assert np.all(mac >= 0.0)
        assert np.all(mac <= 1.0 + 1e-12)

    def test_mac_is_symmetric_for_same_input(self):
        rng = np.random.default_rng(2)
        v = rng.standard_normal((6, 4))
        mac = calculateMAC(v, v)
        np.testing.assert_allclose(mac, mac.T, atol=1e-12)

    def test_mac_with_complex_vectors(self):
        rng = np.random.default_rng(3)
        v = rng.standard_normal((6, 3)) + 1j * rng.standard_normal((6, 3))
        mac = calculateMAC(v, v)
        np.testing.assert_allclose(np.diag(mac), 1.0, atol=1e-12)
        assert np.all(np.isreal(mac))


class TestCalculateMPC:
    def test_real_mode_shape_gives_mpc_one(self):
        v = np.random.default_rng(4).standard_normal((8, 3))
        mpc = calculateMPC(v)
        np.testing.assert_allclose(mpc, 1.0, atol=1e-12)

    def test_complex_mode_shape_gives_mpc_less_than_one(self):
        rng = np.random.default_rng(5)
        v = rng.standard_normal((8, 2)) + 1j * rng.standard_normal((8, 2))
        mpc = calculateMPC(v)
        assert np.all(mpc <= 1.0 + 1e-12)
        assert np.all(mpc >= 0.0)

    def test_mpc_bounded_zero_to_one(self):
        rng = np.random.default_rng(6)
        for _ in range(20):
            v = rng.standard_normal((10, 4)) + 1j * rng.standard_normal((10, 4))
            mpc = calculateMPC(v)
            assert np.all(mpc >= -1e-12)
            assert np.all(mpc <= 1.0 + 1e-12)


class TestCalculateMPD:
    @pytest.mark.parametrize('method', ['ortho', 'usv'])
    def test_real_mode_gives_zero_phase_deviation(self, method):
        v = np.random.default_rng(7).standard_normal((8, 3)).astype(complex)
        mpd, mp = calculateMPD(v, regression_type=method)
        np.testing.assert_allclose(mpd, 0.0, atol=1e-8)

    def test_arithm_all_positive_real_gives_zero_phase_deviation(self):
        v = np.abs(np.random.default_rng(11).standard_normal((8, 3))).astype(complex)
        mpd, _ = calculateMPD(v, regression_type='arithm')
        np.testing.assert_allclose(mpd, 0.0, atol=1e-8)

    @pytest.mark.parametrize('method', ['ortho', 'arithm', 'usv'])
    def test_mpd_non_negative(self, method):
        rng = np.random.default_rng(8)
        v = rng.standard_normal((10, 4)) + 1j * rng.standard_normal((10, 4))
        mpd, _ = calculateMPD(v, regression_type=method)
        assert np.all(mpd >= 0.0)


class TestCalcXYZ:
    def test_zero_elevation_gives_xy_plane_point(self):
        x, y, z = calc_xyz(az=0.0, elev=0.0, r=1.0)
        assert abs(z) < 1e-12
        assert abs(x - 1.0) < 1e-12
        assert abs(y) < 1e-12

    def test_ninety_degree_elevation_gives_z_point(self):
        x, y, z = calc_xyz(az=0.0, elev=np.pi / 2, r=1.0)
        assert abs(z - 1.0) < 1e-10

    def test_unit_sphere_radius(self):
        az_vals = np.linspace(0, 2 * np.pi, 12, endpoint=False)
        el_vals = np.linspace(-np.pi / 2, np.pi / 2, 6)
        for az in az_vals:
            for el in el_vals:
                x, y, z = calc_xyz(az, el, r=1.0)
                r = np.sqrt(x**2 + y**2 + z**2)
                assert abs(r - 1.0) < 1e-12, f"r={r} for az={az:.2f}, el={el:.2f}"


class TestValidateArray:
    def test_none_returns_none(self):
        assert validate_array(None) is None

    def test_numeric_array_returned_unchanged(self):
        arr = np.array([1.0, 2.0, 3.0])
        result = validate_array(arr)
        np.testing.assert_array_equal(result, arr)

    def test_scalar_array_returns_python_scalar(self):
        arr = np.array(42.0)
        result = validate_array(arr)
        assert isinstance(result, float)
        assert result == 42.0

    def test_object_array_returns_list(self):
        arr = np.array(['a', 'b', 'c'])
        result = validate_array(arr)
        assert isinstance(result, list)


class TestMatrixDecompositions:
    def test_lq_decomp_reconstruction(self):
        rng = np.random.default_rng(9)
        A = rng.standard_normal((6, 4))
        L, Q = lq_decomp(A, mode='reduced')
        np.testing.assert_allclose(L @ Q, A, atol=1e-12)

    def test_rq_decomp_reconstruction(self):
        rng = np.random.default_rng(10)
        A = rng.standard_normal((4, 6))
        R, Q = rq_decomp(A, mode='reduced')
        np.testing.assert_allclose(R @ Q, A, atol=1e-12)
