import numpy as np
import pandas as pd
import pytest

from pysrf import CVResult, cross_val_score
from pysrf._common import observation_mask
from pysrf.coherence import calibrate_cross_validation
from pysrf.cross_validation import (
    _entrywise_splits,
    get_fold_fraction,
)
from helpers import make_cv_diagnostic_matrix, make_symmetric_matrix


@pytest.fixture(scope="module")
def s():
    return make_symmetric_matrix(n=60, rank=5, noise_level=0.05, seed=0)


def test_returns_cv_result_with_fold_scores(s):
    result = cross_val_score(
        s,
        ranks=[3, 5, 7],
        sampling_fraction=0.5,
        n_folds=3,
        n_jobs=1,
        srf_kwargs={"max_outer": 2, "max_inner": 2},
    )
    assert isinstance(result, CVResult)
    assert isinstance(result.fold_scores, pd.DataFrame)
    assert set(result.fold_scores.columns) == {
        "repeat",
        "fold",
        "candidate_rank",
        "val_mse",
    }
    assert len(result.fold_scores) == 3 * 3  # n_ranks * n_folds * n_repeats
    assert result.spectral_cutoff is None
    assert result.candidate_ranks == (3, 5, 7)
    assert result.model_rank in result.candidate_ranks
    assert set(result.fold_scores["repeat"]) == {0}
    assert set(result.fold_scores["fold"]) == {0, 1, 2}

    fold_grid = result.fold_scores.groupby(["repeat", "fold"])["candidate_rank"].apply(
        tuple
    )
    assert all(ranks == result.candidate_ranks for ranks in fold_grid)
    assert set(result.rank_scores.columns) == {
        "candidate_rank",
        "val_mse_mean",
        "val_mse_std",
        "n_fold_scores",
        "val_mse_sem",
    }
    assert result.rank_scores["n_fold_scores"].tolist() == [3, 3, 3]


def test_argmin_near_true_rank(s):
    result = cross_val_score(
        s,
        ranks=[2, 4, 5, 6, 8],
        n_folds=5,
        n_jobs=1,
    )
    assert result.spectral_cutoff is not None
    assert result.calibration is not None
    assert result.model_rank == int(
        result.rank_scores.loc[
            result.rank_scores["val_mse_mean"].idxmin(),
            "candidate_rank",
        ]
    )
    assert abs(result.model_rank - 5) <= 1


def test_calibrates_sampling_fraction_when_omitted(s):
    result = cross_val_score(
        s,
        ranks=[3, 5, 7],
        n_folds=3,
        n_jobs=1,
        srf_kwargs={"max_outer": 2, "max_inner": 2},
    )
    assert result.calibration is not None
    assert result.spectral_cutoff == result.calibration.spectral_cutoff
    assert result.candidate_ranks == (3, 5, 7)
    assert set(result.rank_scores.columns) == {
        "candidate_rank",
        "val_mse_mean",
        "val_mse_std",
        "n_fold_scores",
        "val_mse_sem",
    }


def test_validates_sampling_fraction(s):
    with pytest.raises(ValueError, match="sampling_fraction"):
        cross_val_score(s, ranks=[3], sampling_fraction=1.5)
    with pytest.raises(ValueError, match="sampling_fraction"):
        cross_val_score(s, ranks=[3], sampling_fraction=0.0)


def test_validates_n_folds(s):
    with pytest.raises(ValueError, match="n_folds"):
        cross_val_score(s, ranks=[3], sampling_fraction=0.5, n_folds=1)


def test_deterministic(s):
    a = cross_val_score(
        s,
        ranks=[3, 5],
        sampling_fraction=0.5,
        n_folds=3,
        random_state=11,
        n_jobs=1,
        srf_kwargs={"max_outer": 2, "max_inner": 2},
    )
    b = cross_val_score(
        s,
        ranks=[3, 5],
        sampling_fraction=0.5,
        n_folds=3,
        random_state=11,
        n_jobs=1,
        srf_kwargs={"max_outer": 2, "max_inner": 2},
    )
    assert a.model_rank == b.model_rank
    pd.testing.assert_frame_equal(a.fold_scores, b.fold_scores)
    pd.testing.assert_frame_equal(a.rank_scores, b.rank_scores)


def test_splits_hide_only_observed_entries(s):
    s_missing = s.copy()
    s_missing[0, 1] = s_missing[1, 0] = np.nan
    observed = observation_mask(s_missing)
    splits = _entrywise_splits(
        observed,
        fold_fraction=get_fold_fraction(0.6, n_folds=3, n=s.shape[0]),
        n_folds=3,
        split_seeds=np.array([11], dtype=np.uint32),
    )
    for split in splits:
        assert np.all(split.train_mask == split.train_mask.T)
        assert np.all(split.validation_mask == split.validation_mask.T)
        assert not np.any(split.train_mask & split.validation_mask)
        assert np.all(split.train_mask <= observed)
        assert np.all(split.validation_mask <= observed)
        diag = np.diag_indices_from(observed)
        np.testing.assert_array_equal(split.train_mask[diag], observed[diag])
        assert not split.validation_mask[diag].any()


def test_custom_missing_value(s):
    s_missing = s.copy()
    s_missing[0, 1] = s_missing[1, 0] = -1.0
    result = cross_val_score(
        s_missing,
        ranks=[3],
        sampling_fraction=0.5,
        n_folds=3,
        n_jobs=1,
        missing_values=-1.0,
        srf_kwargs={"max_outer": 2, "max_inner": 2},
    )
    assert len(result.fold_scores) == 3
    assert result.spectral_cutoff is None
    assert set(result.fold_scores["candidate_rank"]) == {3}
    assert np.isfinite(result.fold_scores["val_mse"]).all()
    assert "rank" not in result.fold_scores.columns
    assert "rep" not in result.fold_scores.columns


def test_cap_warning_when_inflation_exceeds_cap(s):
    with pytest.warns(RuntimeWarning, match="cap"):
        cross_val_score(
            s,
            ranks=[3],
            sampling_fraction=0.95,
            n_folds=3,
            n_jobs=1,
            srf_kwargs={"max_outer": 2, "max_inner": 2},
        )


def test_end_to_end_calibrated_cross_validation_selects_model_rank():
    true_rank = 4
    seed = 8
    s = make_cv_diagnostic_matrix(
        n=70,
        rank=true_rank,
        nuisance_rank=8,
        nuisance_scale=1.05,
        noise=0.05,
        seed=seed,
    )
    calibration = calibrate_cross_validation(
        s,
        max_eigenpairs=12,
        sampling_grid=np.linspace(0.15, 0.9, 8),
        n_bootstrap=5,
        random_state=seed,
        n_jobs=1,
    )
    result = cross_val_score(
        s,
        ranks=range(1, 13),
        sampling_fraction=calibration.sampling_fraction,
        n_folds=5,
        n_repeats=2,
        random_state=seed,
        n_jobs=1,
        srf_kwargs={
            "max_outer": 20,
            "max_inner": 10,
            "check_input": False,
            "tol": 1e-5,
        },
    )

    best_row = result.rank_scores["val_mse_mean"].idxmin()
    assert result.sampling_fraction == pytest.approx(calibration.sampling_fraction)
    assert result.model_rank == int(result.rank_scores.loc[best_row, "candidate_rank"])
    assert result.model_rank == true_rank
