"""Checks that computed results match the executed notebook. Run: pytest -q"""
import pytest
from src import metrics as m


@pytest.fixture(scope="module")
def data():
    return m.load()


def test_split_sizes(data):
    inc, _ = data
    _, split = m.evaluate(inc)
    assert split == {"train_hours": 120, "train_critical": 17, "test_hours": 48, "test_critical": 9}


def test_same_hour_results(data):
    inc, _ = data
    row = m.evaluate(inc)[0].set_index("test")
    for name in ["Same hour: NEFT < 95% rule", "Same hour: logistic, all KPIs", "Same hour: logistic, without NEFT"]:
        assert row.loc[name, "f1"] == 1.0


def test_early_warning_results(data):
    inc, _ = data
    row = m.evaluate(inc)[0].set_index("test")
    assert row.loc["Early warning: previous hour NEFT < 95%", "f1"] == 0.0
    lag = row.loc["Early warning: logistic, lagged KPIs"]
    assert (lag["precision"], lag["recall"], lag["f1"]) == (0.214, 0.333, 0.261)


def test_every_kpi_separates_labels(data):
    inc, _ = data
    assert m.separation_table(inc)["separates_perfectly"].all()


def test_file_mismatch_counts(data):
    inc, infra = data
    mm = m.file_mismatch(inc, infra)
    assert mm["hours_compared"] == 168
    assert mm["matching_hours"] == 142
    assert mm["critical_hours"] == 26
    assert mm["critical_hours_matching"] == 0


def test_no_hardcoded_results_in_metrics_module():
    src = open(m.__file__).read()
    for fake in ["0.92", "0.94", "99.85", "44.8"]:
        assert fake not in src
