"""Streamlit page. Shows only values computed in src/metrics.py."""
import matplotlib.pyplot as plt
import streamlit as st
from src import metrics as m

st.set_page_config(page_title="Banking Incident Classification v1", layout="wide")
st.title("Banking incident classification: v1")
st.caption("Every number on this page is computed from the CSV files. The labels appear rule-based, so read the results with that in mind.")

inc, infra = m.load()
table, split = m.evaluate(inc)

st.subheader("Results")
st.caption(f"Test period: {split['test_hours']} hours with {split['test_critical']} critical. "
           f"Training: {split['train_hours']} hours with {split['train_critical']} critical.")
st.dataframe(table.set_index("test"), use_container_width=True)
st.markdown("Same-hour scores are perfect because one KPI threshold separates the labels. "
            "The early-warning rows are the real test: they miss most critical hours and raise false alarms.")

st.subheader("Critical vs. normal hours")
st.dataframe(m.kpi_by_label(inc), use_container_width=True)
st.dataframe(m.separation_table(inc), use_container_width=True, hide_index=True)

st.subheader("NEFT success over time")
fig, ax = plt.subplots(figsize=(11, 3.5))
ax.plot(inc["timestamp"], inc["neft_success_rate"], color="steelblue", label="NEFT success (%)")
crit = inc[inc["is_critical"] == 1]
ax.scatter(crit["timestamp"], crit["neft_success_rate"], color="red", zorder=3, label="Critical hour")
ax.axhline(m.NEFT_RULE, color="grey", linestyle="--", label="95% rule")
ax.legend()
st.pyplot(fig)

st.subheader("Data files disagree")
mm = m.file_mismatch(inc, infra)
st.write(f"{mm['matching_hours']} of {mm['hours_compared']} hours match between the two files. "
         f"{mm['critical_hours_matching']} of {mm['critical_hours']} critical hours match. "
         "v1 uses the incident file only. The source of truth is an open question for the instructor.")
