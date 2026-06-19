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
        mpd, _mp = calculateMPD(v, regression_type=method)
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
        _x, _y, z = calc_xyz(az=0.0, elev=np.pi / 2, r=1.0)
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


# ── ConfigFile ────────────────────────────────────────────────────────────────

class TestConfigFile:
    """Unit tests for ConfigFile: the key-value config file parser."""

    @pytest.fixture
    def cfg_path(self, tmp_path):
        p = tmp_path / 'test.txt'
        p.write_text(
            'String Key:\nsome value\n'
            'Int Key:\n42\n'
            'Float Key:\n3.14\n'
            'List Key:\n1 2 3\n'
            'Empty Key:\n\n'
        )
        return p

    def test_str_value(self, cfg_path):
        from pyOMA.core.Helpers import ConfigFile
        assert ConfigFile(cfg_path).str('String Key') == 'some value'

    def test_int_value(self, cfg_path):
        from pyOMA.core.Helpers import ConfigFile
        assert ConfigFile(cfg_path).int('Int Key') == 42

    def test_float_value(self, cfg_path):
        from pyOMA.core.Helpers import ConfigFile
        assert ConfigFile(cfg_path).float('Float Key') == pytest.approx(3.14)

    def test_int_list_value(self, cfg_path):
        from pyOMA.core.Helpers import ConfigFile
        assert ConfigFile(cfg_path).int_list('List Key') == [1, 2, 3]

    def test_empty_int_list_returns_empty(self, cfg_path):
        from pyOMA.core.Helpers import ConfigFile
        assert ConfigFile(cfg_path).int_list('Empty Key') == []

    def test_blank_lines_between_pairs_are_skipped(self, tmp_path):
        p = tmp_path / 'c.txt'
        p.write_text('\nKey One:\nfoo\n\n\nKey Two:\nbar\n')
        from pyOMA.core.Helpers import ConfigFile
        cfg = ConfigFile(p)
        assert cfg.str('Key One') == 'foo'
        assert cfg.str('Key Two') == 'bar'

    def test_comment_lines_are_skipped(self, tmp_path):
        p = tmp_path / 'c.txt'
        p.write_text('# header comment\nKey:\nval\n# trailing\n')
        from pyOMA.core.Helpers import ConfigFile
        assert ConfigFile(p).str('Key') == 'val'

    def test_key_order_does_not_matter(self, tmp_path):
        p = tmp_path / 'c.txt'
        p.write_text('B:\n2\nA:\n1\n')
        from pyOMA.core.Helpers import ConfigFile
        cfg = ConfigFile(p)
        assert cfg.int('A') == 1
        assert cfg.int('B') == 2

    def test_missing_key_raises_keyerror_naming_key(self, cfg_path):
        from pyOMA.core.Helpers import ConfigFile
        with pytest.raises(KeyError, match='Nonexistent Key'):
            ConfigFile(cfg_path).str('Nonexistent Key')

    def test_missing_key_raises_keyerror_naming_file(self, cfg_path):
        from pyOMA.core.Helpers import ConfigFile
        with pytest.raises(KeyError, match=cfg_path.name):
            ConfigFile(cfg_path).str('Nonexistent Key')

    def test_bad_int_raises_valueerror_naming_key(self, tmp_path):
        p = tmp_path / 'c.txt'
        p.write_text('My Key:\nnot_an_int\n')
        from pyOMA.core.Helpers import ConfigFile
        with pytest.raises(ValueError, match='My Key'):
            ConfigFile(p).int('My Key')

    def test_bad_float_raises_valueerror_naming_key(self, tmp_path):
        p = tmp_path / 'c.txt'
        p.write_text('My Key:\nnot_a_float\n')
        from pyOMA.core.Helpers import ConfigFile
        with pytest.raises(ValueError, match='My Key'):
            ConfigFile(p).float('My Key')

    def test_bad_int_list_raises_valueerror_naming_key(self, tmp_path):
        p = tmp_path / 'c.txt'
        p.write_text('My Key:\n1 2 bad\n')
        from pyOMA.core.Helpers import ConfigFile
        with pytest.raises(ValueError, match='My Key'):
            ConfigFile(p).int_list('My Key')

    def test_missing_file_raises_filenotfounderror(self, tmp_path):
        from pyOMA.core.Helpers import ConfigFile
        with pytest.raises(FileNotFoundError):
            ConfigFile(tmp_path / 'does_not_exist.txt')

    # ── Real config files: verify parsed values match file contents ───────────

    def test_ssi_config_parses_correctly(self, test_files_dir):
        from pyOMA.core.Helpers import ConfigFile
        cfg = ConfigFile(test_files_dir / 'ssi_config.txt')
        assert cfg.int('Number of Block-Columns') == 200
        assert cfg.int('Maximum Model Order') == 40

    def test_plscf_config_parses_correctly(self, test_files_dir):
        from pyOMA.core.Helpers import ConfigFile
        cfg = ConfigFile(test_files_dir / 'plscf_config.txt')
        assert cfg.float('Begin Frequency') == pytest.approx(0.0)
        assert cfg.float('End Frequency') == pytest.approx(20.0)
        assert cfg.int('Samples per time segment') == 4096
        assert cfg.int('Maximum Model Order') == 50

    def test_prce_config_parses_correctly(self, test_files_dir):
        from pyOMA.core.Helpers import ConfigFile
        cfg = ConfigFile(test_files_dir / 'prce_config.txt')
        assert cfg.int('Number of Correlation Samples') == 200
        assert cfg.int('Maximum Model Order') == 100

    def test_varssi_config_parses_correctly(self, test_files_dir):
        from pyOMA.core.Helpers import ConfigFile
        cfg = ConfigFile(test_files_dir / 'varssi_config.txt')
        assert cfg.int('Number of Block-Columns') == 200
        assert cfg.int('Maximum Model Order') == 100
        assert cfg.int('Number of Blocks') == 15
        assert cfg.str('Subspace Method (projection/covariance)') == 'covariance'
        assert cfg.str('LSQ Method for A (pinv/qr)') == 'pinv'
        assert cfg.str('Variance Algorithm (fast/slow)') == 'fast'

    def test_setup_info_parses_correctly(self, test_files_dir):
        from pyOMA.core.Helpers import ConfigFile
        cfg = ConfigFile(test_files_dir / 'measurement_1' / 'setup_info.txt')
        assert cfg.str('Setup Name') == 'measurement_1'
        assert cfg.float('Sampling Rate [Hz]') == pytest.approx(256.0)
        assert cfg.int_list('Reference Channels') == [3, 4]
        assert cfg.int_list('Delete Channels') == [5]
        assert cfg.int_list('Accel. Channels') == [3, 4, 5]
        assert cfg.int_list('Velo. Channels') == [0, 1, 2]
        assert cfg.int_list('Disp. Channels') == []
