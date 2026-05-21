from rankcloak.bootstrap_statistics import bootstrap_difference_ci, bootstrap_mean_ci


def test_bootstrap_mean_ci_is_deterministic_and_contains_mean():
    first = bootstrap_mean_ci([1, 2, 3, 4], n_resamples=100, seed=123)
    second = bootstrap_mean_ci([1, 2, 3, 4], n_resamples=100, seed=123)
    assert first == second
    assert first["bootstrap_ci_low_95"] <= first["mean"] <= first["bootstrap_ci_high_95"]


def test_bootstrap_difference_ci_is_deterministic():
    first = bootstrap_difference_ci([1, 2, 3], [3, 4, 5], n_resamples=100, seed=321)
    second = bootstrap_difference_ci([1, 2, 3], [3, 4, 5], n_resamples=100, seed=321)
    assert first == second
    assert first["bootstrap_ci_low_95"] <= first["difference_b_minus_a"] <= first["bootstrap_ci_high_95"]
