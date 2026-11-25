"""Interactive web app to test the ride classifier.

API_URL=http://localhost:8000 uv run streamlit run webapp/app.py
"""

from __future__ import annotations

import os

import client
import pandas as pd
import streamlit as st

CLASSES = ["Safe", "Safe overall", "Bad weather", "Tandem", "Reckless"]

st.set_page_config(page_title="Ride classifier", layout="wide")
st.title("Scooter ride classifier")
st.caption("Send the IMU signal of a ride (accelerometer + gyroscope, 10 Hz) and get its category.")

api_url = st.sidebar.text_input("API URL", os.getenv("API_URL", "http://localhost:8000"))
try:
    info = client.health(api_url)
    if info["status"] == "ok":
        st.sidebar.success(f"API ok, model v{info['model_version']}")
    else:
        st.sidebar.warning("API reachable but no model loaded (degraded)")
except client.ApiError as exc:
    st.sidebar.error(str(exc))

source = st.radio("Ride", ["Demo ride", "Upload a file (CSV / Parquet)"], horizontal=True)

ride: pd.DataFrame | None = None
ride_id: str | None = None
if source == "Demo ride":
    n = st.slider("Number of samples", 100, 5000, 1500, step=100)
    ride, ride_id = client.demo_ride(n), "demo"
else:
    uploaded = st.file_uploader("Columns ax, ay, az, rx, ry, rz (+ optional ride_id, time_index)", ["csv", "parquet"])
    if uploaded is not None:
        try:
            df = client.read_ride(uploaded.name, uploaded.getvalue())
        except Exception as exc:
            st.error(f"Cannot read this file: {exc}")
            df = None
        if df is not None:
            if "ride_id" in df.columns and df["ride_id"].nunique() > 1:
                ride_id = st.selectbox(
                    f"{df['ride_id'].nunique()} rides in this file, pick one", sorted(df["ride_id"].unique())
                )
                df = df[df["ride_id"] == ride_id]
            ride = df

if ride is not None:
    st.subheader(f"{len(ride)} samples")
    plot = ride.sort_values("time_index") if "time_index" in ride.columns else ride
    left, right = st.columns(2)
    left.line_chart(plot[["ax", "ay", "az"]].reset_index(drop=True), height=220)
    right.line_chart(plot[["rx", "ry", "rz"]].reset_index(drop=True), height=220)
    nulls = int(ride[client.CHANNELS].isna().sum().sum())
    if nulls:
        st.info(f"{nulls} missing values: sent as null, the model handles them.")

    if st.button("Classify this ride", type="primary"):
        try:
            with st.spinner("Calling the API..."):
                st.session_state["result"] = client.predict(api_url, client.to_payload(ride, str(ride_id)))
        except client.ApiError as exc:
            st.session_state.pop("result", None)
            st.error(str(exc))

result = st.session_state.get("result")
if result:
    st.subheader(f"Prediction: {result['predicted_class']} ({result['confidence']:.0%})")
    proba = pd.Series(result["probabilities"]).reindex(CLASSES)
    st.bar_chart(proba)
    st.caption(f"Model version {result['model_version']}, {result['n_samples']} samples")

    st.markdown("**Was the prediction right?** Your answer is sent to the API and shown in the monitoring.")
    actual = st.selectbox("Actual class", CLASSES, index=CLASSES.index(result["predicted_class"]))
    if st.button("Send feedback"):
        try:
            client.send_feedback(api_url, result["predicted_class"], actual, ride_hash=result.get("ride_hash"))
            st.success("Feedback recorded")
        except client.ApiError as exc:
            st.error(str(exc))
