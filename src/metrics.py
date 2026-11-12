"""
Computed results for the v1 incident classification project.

Every number here is calculated from the CSV files. Nothing is hardcoded.
"""
import os
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

TARGET = "is_critical"
KPIS = ["cpu_utilization", "memory_usage", "db_connection_pool_usage",
        "application_error_rate", "neft_success_rate"]
NO_NEFT = [k for k in KPIS if k != "neft_success_rate"]
NEFT_RULE = 95.0
TRAIN_DAYS = 5


def default_data_dir():
    return os.environ.get("CAPSTONE_DATA_DIR", os.path.join(os.path.dirname(__file__), "..", "data"))


def load(data_dir=None):
    data_dir = data_dir or default_data_dir()
    inc = pd.read_csv(os.path.join(data_dir, "incident_reports.csv"), parse_dates=["timestamp"])
    infra = pd.read_csv(os.path.join(data_dir, "infrastructure_metrics.csv"), parse_dates=["timestamp"])
    return inc.sort_values("timestamp").reset_index(drop=True), infra.sort_values("timestamp").reset_index(drop=True)


def time_cutoff(inc):
    return inc["timestamp"].min() + pd.Timedelta(days=TRAIN_DAYS)


def kpi_by_label(inc):
    return inc.groupby(TARGET)[KPIS].mean().round(2)


def separation_table(inc):
    rows = []
    for f in KPIS:
        normal, critical = inc.loc[inc[TARGET] == 0, f], inc.loc[inc[TARGET] == 1, f]
        if critical.min() > normal.max():
            rows.append((f, f"critical if > {normal.max():.2f}", True))
        elif critical.max() < normal.min():
            rows.append((f, f"critical if < {normal.min():.2f}", True))
        else:
            rows.append((f, "overlaps", False))
    return pd.DataFrame(rows, columns=["kpi", "rule", "separates_perfectly"])


def file_mismatch(inc, infra):
    joined = inc.set_index("timestamp").join(
        infra.set_index("timestamp")[["neft_success_rate"]], rsuffix="_infra", how="inner")
    match = (joined["neft_success_rate"] - joined["neft_success_rate_infra"]).abs() < 0.01
    critical = joined[TARGET] == 1
    return {
        "hours_compared": int(len(joined)),
        "matching_hours": int(match.sum()),
        "critical_hours": int(critical.sum()),
        "critical_hours_matching": int((match & critical).sum()),
    }


def _metrics(y_true, y_pred):
    return {
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 3),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 3),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 3),
    }


def _logreg():
    return make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000))


def evaluate(inc):
    """Return the v1 results table: same-hour tests and early-warning tests on the same test hours."""
    cutoff = time_cutoff(inc)
    train, test = inc[inc["timestamp"] < cutoff], inc[inc["timestamp"] >= cutoff]
    results = {}

    # Same-hour tests
    y = test[TARGET]
    results["Same hour: NEFT < 95% rule"] = _metrics(y, (test["neft_success_rate"] < NEFT_RULE).astype(int))
    for name, feats in [("Same hour: logistic, all KPIs", KPIS), ("Same hour: logistic, without NEFT", NO_NEFT)]:
        model = _logreg().fit(train[feats], train[TARGET])
        results[name] = _metrics(y, model.predict(test[feats]))

    # Early-warning tests: previous hour's KPIs predict this hour's label
    lagged = inc[KPIS].shift(1).add_suffix("_prev")
    d = pd.concat([inc[["timestamp", TARGET]], lagged], axis=1).dropna().reset_index(drop=True)
    tr, te = d[d["timestamp"] < cutoff], d[d["timestamp"] >= cutoff]
    results["Early warning: previous hour NEFT < 95%"] = _metrics(
        te[TARGET], (te["neft_success_rate_prev"] < NEFT_RULE).astype(int))
    lag_feats = [c for c in d.columns if c.endswith("_prev")]
    model = _logreg().fit(tr[lag_feats], tr[TARGET])
    results["Early warning: logistic, lagged KPIs"] = _metrics(te[TARGET], model.predict(te[lag_feats]))

    table = pd.DataFrame(results).T.reset_index().rename(columns={"index": "test"})
    split = {
        "train_hours": int(len(train)), "train_critical": int(train[TARGET].sum()),
        "test_hours": int(len(test)), "test_critical": int(test[TARGET].sum()),
    }
    return table, split
