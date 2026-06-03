import numpy as np
import pytest

from pysrf import cross_val_score
from pysrf.coherence import CVCalibration, calibrate_cross_validation
from helpers import make_missing_matrix, make_symmetric_matrix


@pytest.fixture(scope="module")
def s():
    return make_symmetric_matrix(n=60, rank=5, noise_level=0.05, seed=0)


def test_returns_cv_calibration(s):
    calibration = calibrate_cross_validation(s, n_bootstrap=5, n_jobs=1)
    assert isinstance(calibration, CVCalibration)


def test_spectral_cutoff_near_true_rank(s):
    calibration = calibrate_cross_validation(s, n_bootstrap=10, n_jobs=1)
    assert abs(calibration.spectral_cutoff - 5) <= 1


def test_sampling_fraction_in_unit_interval(s):
    calibration = calibrate_cross_validation(s, n_bootstrap=5, n_jobs=1)
    assert 0.0 < calibration.sampling_fraction <= 0.95
    assert calibration.detectability_floor <= calibration.sampling_fraction


def test_diagnostic_shapes(s):
    calibration = calibrate_cross_validation(
        s, max_eigenpairs=15, n_bootstrap=5, n_jobs=1
    )
    assert calibration.eigvals.shape == (15,)
    assert calibration.leakage.shape == (15,)
    assert calibration.sampling_grid.shape == calibration.signal_loss_raw.shape
    assert calibration.sampling_grid.shape == calibration.signal_loss_monotone.shape
    assert calibration.n_features_in == s.shape[0]


def test_signal_loss_monotone_non_increasing(s):
    calibration = calibrate_cross_validation(s, n_bootstrap=5, n_jobs=1)
    assert np.all(np.diff(calibration.signal_loss_monotone) <= 1e-12)


def test_deterministic(s):
    a = calibrate_cross_validation(s, n_bootstrap=5, random_state=7, n_jobs=1)
    b = calibrate_cross_validation(s, n_bootstrap=5, random_state=7, n_jobs=1)
    assert a.spectral_cutoff == b.spectral_cutoff
    assert a.sampling_fraction == pytest.approx(b.sampling_fraction)
    np.testing.assert_allclose(a.eigvals, b.eigvals)
    np.testing.assert_allclose(a.signal_loss_monotone, b.signal_loss_monotone)


def test_handles_missing_entries(s):
    s_missing = make_missing_matrix(s, fraction=0.2, seed=1)
    calibration = calibrate_cross_validation(s_missing, n_bootstrap=5, n_jobs=1)
    assert isinstance(calibration, CVCalibration)
    assert calibration.spectral_cutoff >= 1


def test_frozen_dataclass(s):
    calibration = calibrate_cross_validation(s, n_bootstrap=5, n_jobs=1)
    with pytest.raises(AttributeError):
        calibration.spectral_cutoff = 99  # type: ignore[misc]


def test_cross_validation_calibrates_sampling_fraction(s):
    result = cross_val_score(
        s,
        ranks=[3, 5],
        n_folds=3,
        n_jobs=1,
        srf_kwargs={"max_outer": 2, "max_inner": 2},
    )
    assert result.calibration is not None
    assert result.sampling_fraction < 1.0
