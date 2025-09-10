"""
energy_edge_app.py
Advanced Streamlit app (fixed):
- Simulates real-time energy sensor data
- Trains LSTM (sequence) and RandomForest forecasting models
- Uses IsolationForest for anomaly detection
- Visualizes forecasts, anomalies, and allows export (CSV/Excel, TFLite)
- Designed to run in VS Code terminal via: `streamlit run energy_edge_app.py`
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import plotly.express as px
import io
import os

# Try importing tensorflow; gracefully degrade
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

st.set_page_config(layout="wide", page_title="Edge AI Energy Forecasting + Anomaly Detection (Fixed)")

# -------------------------
# Utility / Simulation
# -------------------------
@st.cache_data
def simulate_energy_data(hours=24*30, seed=42):
    """
    Simulate energy-consumption time-series data for 'hours' number of hours.
    Returns a DataFrame with columns: timestamp, energy (kW), temp, humidity
    """
    np.random.seed(seed)
    base = pd.date_range(end=datetime.now(), periods=hours, freq='H')
    df = pd.DataFrame(index=base)
    df['hour'] = df.index.hour
    # daily pattern: sin wave
    df['daily'] = 1.5 + np.sin(2 * np.pi * df['hour'] / 24) * 1.2
    # weekday/weekend effect
    df['weekday'] = (df.index.dayofweek < 5).astype(int)
    # temperature seasonal + daily
    df['temp'] = (20
                  + 6 * np.sin(2 * np.pi * (df.index.dayofyear) / 365)
                  + 3 * np.sin(2 * np.pi * df['hour'] / 24)
                  + np.random.normal(0, 0.8, size=len(df)))
    # occupancy proxy
    df['occupancy'] = (
        ((df['hour'] >= 18) & (df['hour'] <= 22)).astype(int) * (0.5 + np.random.rand(len(df)) * 0.5)
        + ((df['hour'] >= 6) & (df['hour'] <= 8)).astype(int) * 0.4
    )
    df['occupancy'] = df['occupancy'] + np.random.normal(0, 0.05, size=len(df))
    # appliance base loads
    df['appliance_load'] = 0.2 + 0.5 * df['occupancy'] + 0.3 * (df['hour'] >= 19).astype(int)
    # weekend lower load
    df['appliance_load'] *= (1 - 0.12 * (df['weekday'] == 0))
    # main energy signal (kW)
    noise = np.random.normal(0, 0.12, size=len(df))
    df['energy'] = np.clip(df['daily'] * df['appliance_load'] + 0.02 * df['temp'] + noise, 0.01, None)
    # reset index to have timestamp column
    df = df.reset_index().rename(columns={'index': 'timestamp'})
    # add humidity BEFORE selecting columns to avoid KeyError
    df['humidity'] = 40 + 10 * np.sin(2 * np.pi * (df['timestamp'].dt.dayofyear) / 365) + np.random.normal(0, 2, len(df))
    # Introduce sporadic anomalies (spikes)
    for _ in range(max(1, hours // 250)):
        idx = np.random.randint(0, len(df))
        df.loc[idx, 'energy'] *= (3 + np.random.rand() * 4)  # spike
    return df[['timestamp', 'energy', 'temp', 'humidity']]

# -------------------------
# Feature engineering
# -------------------------
def add_time_features(df):
    df = df.copy()
    df['hour'] = df['timestamp'].dt.hour
    df['dayofweek'] = df['timestamp'].dt.dayofweek
    # cyclical encoding
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['day_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
    return df

def create_sequences(data, seq_len=24, target_col=0):
    """
    data: numpy array (rows x features) or pandas DataFrame converted to numpy
    target_col: int index of target column inside 'data' rows
    """
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i + seq_len])
        y.append(data[i + seq_len][target_col])
    if len(X) == 0:
        return np.empty((0, seq_len, data.shape[1])), np.empty((0,))
    return np.array(X), np.array(y)

# -------------------------
# Models
# -------------------------
def train_random_forest(X_train, y_train):
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    return model

def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(64, activation='tanh', input_shape=input_shape))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

# -------------------------
# Anomaly Detection
# -------------------------
def fit_isolation_forest(X, contamination=0.01):
    iso = IsolationForest(n_estimators=200, contamination=contamination, random_state=42)
    iso.fit(X)
    return iso

# -------------------------
# Utilities: Export & Reports
# -------------------------
def df_to_excel_bytes(df):
    towrite = io.BytesIO()
    df.to_excel(towrite, index=False, engine='openpyxl')
    towrite.seek(0)
    return towrite.getvalue()

def df_to_csv_bytes(df):
    return df.to_csv(index=False).encode('utf-8')

def save_tf_to_tflite(model, filename='model.tflite'):
    # model is a tf.keras model
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        with open(filename, 'wb') as f:
            f.write(tflite_model)
        return filename
    except Exception as e:
        return None

# -------------------------
# Streamlit App UI
# -------------------------
st.title("Edge AI — Energy Forecasting & Anomaly Detection (Fixed)")
st.markdown("Sanskar — improved single-file template: LSTM + RandomForest forecasting, IsolationForest anomaly detection, TFLite export.")

# Sidebar controls
st.sidebar.header("Settings")
hours_sim = st.sidebar.slider("Hours to simulate (historic)", min_value=24*7, max_value=24*365, value=24*90, step=24)
train_frac = st.sidebar.slider("Training fraction", 0.5, 0.9, 0.8)
seq_len = st.sidebar.number_input("LSTM sequence length (hours)", min_value=6, max_value=168, value=24, step=1)
use_lstm = st.sidebar.checkbox("Train LSTM model", value=TF_AVAILABLE)
use_rf = st.sidebar.checkbox("Train RandomForest model", value=True)
contamination = st.sidebar.slider("Anomaly contamination (IForest)", 0.001, 0.1, 0.02)

st.sidebar.markdown("---")
st.sidebar.write("TensorFlow available: " + ("✅" if TF_AVAILABLE else "❌ (LSTM disabled)"))

# Generate / load data
if 'df' not in st.session_state:
    st.session_state.df = simulate_energy_data(hours=hours_sim)
else:
    # If user changed simulation size, regenerate so slider takes effect
    if st.session_state.df.shape[0] != hours_sim:
        st.session_state.df = simulate_energy_data(hours=hours_sim)

col1, col2 = st.columns([2, 1])
with col2:
    if st.button("Regenerate simulation"):
        st.session_state.df = simulate_energy_data(hours=hours_sim, seed=int(time.time()) % 100000)
        st.success("Simulated data regenerated.")

with col1:
    st.subheader("Data preview (latest 200 rows)")
    df = st.session_state.df.copy()
    st.dataframe(df.tail(200))

# Feature engineering
df = add_time_features(df)
feature_cols = ['energy', 'temp', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos']
st.write("Features used:", feature_cols)

# Train/Test split for classic ML
X_ml = df[feature_cols].values
y_ml = df['energy'].values
X_train_ml, X_test_ml, y_train_ml, y_test_ml = train_test_split(X_ml, y_ml, train_size=train_frac, shuffle=False)

# Scale for ML and IsolationForest
scaler_ml = StandardScaler()
X_train_ml_scaled = scaler_ml.fit_transform(X_train_ml)
X_test_ml_scaled = scaler_ml.transform(X_test_ml)

# Store scaler_ml so it can be reused across reruns (preview/export)
st.session_state['scaler_ml'] = scaler_ml

# Train RandomForest if chosen
rf_model = None
if use_rf:
    with st.spinner("Training RandomForest..."):
        rf_model = train_random_forest(X_train_ml_scaled, y_train_ml)
    y_pred_rf = rf_model.predict(X_test_ml_scaled)
    rmse_rf = np.sqrt(mean_squared_error(y_test_ml, y_pred_rf))
    st.success(f"RandomForest trained — Test RMSE: {rmse_rf:.4f} kW")
    # Append to session state for later
    st.session_state['rf_model'] = rf_model

# Prepare LSTM sequences (if TF available and user wants)
lstm_available = TF_AVAILABLE and use_lstm
lstm_model = None
if lstm_available:
    st.info("Preparing data for LSTM (this can take a bit)...")
    # Build dataset for sequences using features (not raw scaled)
    ml_df = df[['timestamp', 'energy', 'temp', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos']].copy()
    feature_seq = ['energy', 'temp', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos']
    scaler_seq = StandardScaler()
    scaled_features = scaler_seq.fit_transform(ml_df[feature_seq])
    # save scaler and feature list to session_state so preview can reuse them
    st.session_state['scaler_seq'] = scaler_seq
    st.session_state['feature_seq'] = feature_seq

    seq_data = np.array(scaled_features)
    # if dataset too small for sequences, warn
    if len(seq_data) <= seq_len:
        st.warning(f"Not enough data for LSTM sequences. Need > seq_len ({seq_len}) rows; currently {len(seq_data)} rows.")
    else:
        X_seq, y_seq = create_sequences(seq_data, seq_len=seq_len, target_col=0)  # energy at position 0
        # train/test split keeping order
        split_idx = int(len(X_seq) * train_frac)
        if split_idx < 1:
            st.warning("Training split too small for LSTM; increase training fraction or supply more data.")
        else:
            X_train_seq, X_test_seq = X_seq[:split_idx], X_seq[split_idx:]
            y_train_seq, y_test_seq = y_seq[:split_idx], y_seq[split_idx:]

            # Build and train model when button clicked
            if st.button("Train LSTM (may take time)"):
                with st.spinner("Training LSTM..."):
                    input_shape = (X_train_seq.shape[1], X_train_seq.shape[2])
                    lstm_model = build_lstm_model(input_shape)
                    history = lstm_model.fit(X_train_seq, y_train_seq, epochs=10, batch_size=32, validation_split=0.1, verbose=0)
                    st.session_state['lstm_model'] = lstm_model
                    st.success("LSTM trained.")
                    # Evaluate (invert scaling for RMSE calc)
                    y_pred_seq_scaled = lstm_model.predict(X_test_seq).flatten()

                    def inverse_energy(vals_scaled, scaler=scaler_seq):
                        # create placeholder array with appropriate columns
                        dummy = np.zeros((len(vals_scaled), len(feature_seq)))
                        dummy[:, 0] = vals_scaled
                        inv = scaler.inverse_transform(dummy)[:, 0]
                        return inv

                    y_pred_inv = inverse_energy(y_pred_seq_scaled)
                    y_test_inv = inverse_energy(y_test_seq)
                    rmse_lstm = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
                    st.info(f"LSTM Test RMSE (kW): {rmse_lstm:.4f}")
            else:
                st.info("Press the 'Train LSTM' button to train the LSTM model.")
else:
    if use_lstm and not TF_AVAILABLE:
        st.info("TensorFlow not available — LSTM disabled.")

# -------------------------
# Anomaly Detection
# -------------------------
st.subheader("Anomaly Detection (IsolationForest)")
if st.button("Run IsolationForest"):
    with st.spinner("Fitting IsolationForest..."):
        iso = fit_isolation_forest(X_train_ml_scaled, contamination=contamination)
        preds = iso.predict(X_ml)  # -1 anomaly, 1 normal
        df['anomaly'] = (preds == -1).astype(int)
        n_anom = int(df['anomaly'].sum())
        st.session_state['anomaly_df'] = df.copy()
        st.success(f"IsolationForest done — detected {n_anom} anomalies.")
else:
    if 'anomaly_df' not in st.session_state:
        st.write("No anomalies detected yet. Click 'Run IsolationForest' to detect.")
    else:
        df = st.session_state['anomaly_df']

# -------------------------
# Forecasting outputs & visualization
# -------------------------
st.subheader("Forecast vs Actual (Test period)")
preview_h = 24 * 5
preview_df = df.tail(preview_h).copy()
preview_df['timestamp_str'] = preview_df['timestamp'].dt.strftime("%Y-%m-%d %H:%M")

# If RF exists, predict preview
if 'rf_model' in st.session_state:
    rf_model = st.session_state['rf_model']
    X_preview = preview_df[feature_cols].values
    scaler_ml = st.session_state.get('scaler_ml', scaler_ml)
    X_preview_scaled = scaler_ml.transform(X_preview)
    preview_df['pred_rf'] = rf_model.predict(X_preview_scaled)

# If LSTM trained & saved, produce LSTM preview (requires scaler_seq & session model)
if TF_AVAILABLE and 'lstm_model' in st.session_state and 'scaler_seq' in st.session_state:
    try:
        lstm_model = st.session_state['lstm_model']
        scaler_seq = st.session_state['scaler_seq']
        feature_seq = st.session_state['feature_seq']
        # prepare last seq_len data
        df_features = df[feature_seq].copy()
        scaled_all = scaler_seq.transform(df_features)
        if len(scaled_all) >= seq_len:
            last_scaled = scaled_all[-seq_len:]
            X_in = np.expand_dims(last_scaled, axis=0)
            pred_scaled = lstm_model.predict(X_in).flatten()[0]
            # invert only energy
            dummy = np.zeros((1, scaled_all.shape[1]))
            dummy[0, 0] = pred_scaled
            pred_energy = scaler_seq.inverse_transform(dummy)[0, 0]
            preview_df['pred_lstm_next_hour'] = np.nan
            preview_df.at[preview_df.index[-1], 'pred_lstm_next_hour'] = pred_energy
    except Exception as e:
        st.warning("Couldn't produce LSTM preview: " + str(e))

# Plot interactive forecast chart
fig = px.line(preview_df, x='timestamp', y=['energy'], title="Recent Actual Energy (kW)")
if 'pred_rf' in preview_df.columns:
    fig.add_scatter(x=preview_df['timestamp'], y=preview_df['pred_rf'], mode='lines', name='RF Forecast')
if 'pred_lstm_next_hour' in preview_df.columns and not preview_df['pred_lstm_next_hour'].isna().all():
    fig.add_scatter(x=[preview_df['timestamp'].iloc[-1] + pd.Timedelta(hours=1)],
                    y=[preview_df['pred_lstm_next_hour'].iloc[-1]],
                    mode='markers+lines', name='LSTM next-hour prediction')
# mark anomalies with red X
if 'anomaly' in preview_df.columns:
    anom_points = preview_df[preview_df['anomaly'] == 1]
    if not anom_points.empty:
        fig.add_scatter(x=anom_points['timestamp'], y=anom_points['energy'],
                        mode='markers', marker=dict(size=10, symbol='x', color='red'),
                        name='Anomaly')
st.plotly_chart(fig, use_container_width=True)

# -------------------------
# Live Simulation Mode (client-side, simulated streaming)
# -------------------------
st.subheader("Simulated Real-time Stream (client-side)")
col_a, col_b = st.columns([2, 1])
with col_b:
    stream_seconds = st.number_input("Stream interval (sec)", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
    n_stream = st.number_input("Number of stream steps", min_value=1, max_value=500, value=30)
    start_stream = st.button("Start Simulation Stream")
with col_a:
    st.write("You will see simulated incoming readings appended in real-time in the table below.")

if start_stream:
    buffer = []
    last_time = df['timestamp'].max()
    for i in range(int(n_stream)):
        next_ts = last_time + pd.Timedelta(hours=1)
        hour = next_ts.hour
        daily = 1.5 + np.sin(2 * np.pi * hour / 24) * 1.2
        temp = 20 + 3 * np.sin(2 * np.pi * hour / 24) + np.random.normal(0, 0.8)
        occupancy = 0.6 if (18 <= hour <= 22) else (0.4 if (6 <= hour <= 8) else 0.15)
        energy = max(0.02 * temp + daily * occupancy + np.random.normal(0, 0.12), 0.01)
        row = {'timestamp': next_ts, 'energy': energy, 'temp': temp, 'humidity': 40 + np.random.normal(0, 2)}
        buffer.append(row)
        last_time = next_ts
        st.write(pd.DataFrame([row]))
        time.sleep(stream_seconds)

    append_df = pd.DataFrame(buffer)
    st.session_state.df = pd.concat([st.session_state.df, append_df], ignore_index=True)
    st.success(f"Streamed {len(buffer)} new samples and appended to dataset.")

# -------------------------
# Exporting: Reports & Edge Model
# -------------------------
st.subheader("Export & Edge Options")
exp_col1, exp_col2 = st.columns(2)

with exp_col1:
    st.write("Download dataset snapshot")
    csv_bytes = df_to_csv_bytes(df.tail(500))
    st.download_button("Download CSV (last 500 rows)", data=csv_bytes, file_name="energy_snapshot.csv", mime="text/csv")
    excel_bytes = df_to_excel_bytes(df.tail(500))
    st.download_button("Download Excel (last 500 rows)", data=excel_bytes, file_name="energy_snapshot.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

with exp_col2:
    if TF_AVAILABLE and 'lstm_model' in st.session_state:
        if st.button("Export LSTM -> TFLite (edge)"):
            with st.spinner("Converting to TFLite..."):
                filename = f"lstm_model_{int(time.time())}.tflite"
                res = save_tf_to_tflite(st.session_state['lstm_model'], filename=filename)
                if res:
                    with open(res, 'rb') as f:
                        data = f.read()
                    st.download_button("Download TFLite model", data=data, file_name=os.path.basename(res),
                                       mime="application/octet-stream")
                    st.success("TFLite model exported.")
                else:
                    st.error("TFLite conversion failed (see console).")
    else:
        st.info("TFLite export unavailable (TensorFlow not available or LSTM not trained).")

# -------------------------
# Short energy-saving suggestions (rule-based)
# -------------------------
st.subheader("AI-driven Suggestions (rule-based)")
try:
    avg_by_hour = df.groupby(df['timestamp'].dt.hour)['energy'].mean()
    peak_hour = int(avg_by_hour.idxmax())
    st.write(f"Peak average usage hour: **{peak_hour}:00** — average {avg_by_hour.max():.3f} kW")
except Exception:
    st.write("Not enough data to compute hourly averages.")

suggestions = []
if 'avg_by_hour' in locals() and avg_by_hour.max() > 1.0:
    suggestions.append("Usage is high overall. Consider shifting heavy tasks to off-peak hours or scheduling them for night.")
if 'avg_by_hour' in locals() and 18 <= peak_hour <= 22:
    suggestions.append("Evening peak — avoid running multiple high-power appliances simultaneously between 18:00-22:00.")
if 'anomaly' in df.columns and df['anomaly'].sum():
    suggestions.append("Anomalies detected — check for faulty devices or unexpected usage spikes at the flagged timestamps.")
if not suggestions:
    suggestions.append("Consumption looks stable. Keep monitoring and consider adding smart plugs for appliance-level tracking.")

for s in suggestions:
    st.info(s)

# -------------------------
# End
# -------------------------
st.markdown("---")
st.caption("This app is an improved template — integrate real MQTT/Edge devices, expand features, or persist models to disk. Ask if you want it modularized or to add MQTT ingestion. — Sanskar")
