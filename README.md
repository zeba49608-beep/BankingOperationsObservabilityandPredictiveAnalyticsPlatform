# Banking Incident Classification: v1

Can critical incident hours be detected from operational KPIs, and does the previous hour give early warning?

## Results
All numbers are computed in `src/metrics.py` and checked by `tests/test_metrics.py`.

- Same-hour detection is perfect (F1 = 1.000). A single KPI threshold separates the labels, so the labels appear rule-based.
- Early warning is weak. The best lagged model catches 3 of 9 critical test hours (precision 0.214, recall 0.333, F1 0.261), with 11 false alarms.

## Data
Place `incident_reports.csv` and `infrastructure_metrics.csv` in `data/`. The data is not committed to the repo.

## Run
```bash
pip install -r requirements.txt
pytest -q
streamlit run app.py
```

## Limitations
- One week of hourly data.
- Labels appear derived from thresholds on the same KPIs.
- 26 critical hours in total, 9 in the test set.
- The two source files disagree at every critical hour. v1 uses the incident file only.

## Not included in v1
Minute-level anomaly detection, log clustering, RAG, FastAPI, Docker.
