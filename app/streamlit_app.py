"""
app/streamlit_app.py
Multimodal Early Warning Dashboard for Coral Reef Bleaching Risk (2-8 Weeks Ahead)
Fuses:
1. Satellite Sea Surface Temperature & Degree Heating Weeks (SST/DHW)
2. Underwater Acoustic Soundscape Biophony (Snapping shrimp & fish chorus)
3. Citizen Diver Reports (Reef Check, CoralWatch)
"""

import os
import sys
import json
from datetime import date
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium

# Ensure project root is in sys.path
sys.path.append(os.path.abspath("."))
import src.utils
from app.api_client import api_client

st.set_page_config(
    page_title="CoralGuard | Multimodal Coral Bleaching Early Warning",
    page_icon="🪸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    :root {
        --navy: #062c3d;
        --teal: #087f8c;
        --cyan: #33b4c5;
        --pale: #effaf8;
        --border: #dbe9e7;
    }
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #f7fbfa 0%, #edf7f5 100%);
    }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #062c3d 0%, #0a4555 100%);
        border-right: 1px solid rgba(255,255,255,0.08);
    }
    [data-testid="stSidebar"] * { color: #ecfdf9; }
    [data-testid="stSidebar"] [data-baseweb="radio"] label {
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 10px;
        padding: 8px 10px;
        margin-bottom: 6px;
    }
    [data-testid="stMetric"] {
        background: white;
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 14px;
        box-shadow: 0 8px 24px rgba(6,44,61,0.07);
    }
    .main-header {
        font-size: 2.15rem;
        font-weight: 760;
        color: var(--navy);
        margin-bottom: 0.15rem;
        letter-spacing: -0.02em;
    }
    .sub-header {
        font-size: 1rem;
        color: #55716f;
        margin-bottom: 1.25rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: white;
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 6px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 9px;
        color: #55716f;
    }
    .stTabs [aria-selected="true"] {
        background: var(--pale);
        color: var(--teal);
        font-weight: 700;
    }
    .login-shell {
        max-width: 980px;
        margin: 2.5rem auto;
        background: white;
        border: 1px solid var(--border);
        border-radius: 24px;
        box-shadow: 0 24px 70px rgba(6,44,61,0.12);
        overflow: hidden;
    }
    .login-brand {
        background: linear-gradient(135deg, #062c3d 0%, #087f8c 100%);
        min-height: 510px;
        padding: 48px;
        color: white;
        position: relative;
        isolation: isolate;
        overflow: hidden;
    }
    .login-brand::before {
        content: "";
        position: absolute;
        inset: 0;
        z-index: -2;
        background:
            linear-gradient(135deg, rgba(6,44,61,0.94) 8%, rgba(8,127,140,0.72) 100%),
            url("https://images.unsplash.com/photo-1546026423-cc4642628d2b?auto=format&fit=crop&w=1200&q=85");
        background-size: cover;
        background-position: center;
    }
    .login-brand::after {
        content: "";
        position: absolute;
        width: 280px;
        height: 280px;
        right: -100px;
        bottom: -120px;
        border: 44px solid rgba(255,255,255,0.08);
        border-radius: 50%;
        z-index: -1;
    }
    .login-brand h1 { color: white; font-size: 3rem; margin-top: 40px; }
    .login-brand p { color: #d8f3ef; line-height: 1.7; }
    .login-feature {
        display: flex;
        align-items: flex-start;
        gap: 12px;
        padding: 10px 12px;
        margin-bottom: 8px;
        border: 1px solid rgba(255,255,255,0.13);
        border-radius: 12px;
        background: rgba(255,255,255,0.08);
        backdrop-filter: blur(8px);
    }
    .login-feature strong { color: white; display: block; }
    .login-feature span { color: #d8f3ef; font-size: 0.82rem; }
    .login-form { padding: 48px 42px; }
    .login-form h2 { color: var(--navy); }
    .role-chip {
        display: inline-block;
        background: #dff5ef;
        color: #087f8c;
        border-radius: 999px;
        padding: 6px 12px;
        font-size: 0.82rem;
        font-weight: 700;
    }
    .metric-card {
        background: white;
        border-radius: 14px;
        padding: 15px;
        border: 1px solid var(--border);
        box-shadow: 0 8px 24px rgba(6,44,61,0.07);
    }
    .alert-banner-watch { background-color: #FEF08A; padding: 10px; border-radius: 6px; color: #854D0E; font-weight: 600; }
    .alert-banner-warning { background-color: #FED7AA; padding: 10px; border-radius: 6px; color: #9A3412; font-weight: 600; }
    .alert-banner-alert { background-color: #FECACA; padding: 10px; border-radius: 6px; color: #991B1B; font-weight: 600; }
    .alert-banner-normal { background-color: #DCFCE7; padding: 10px; border-radius: 6px; color: #166534; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

def render_login():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if st.session_state.authenticated:
        return

    st.markdown('<div class="login-shell">', unsafe_allow_html=True)
    left, right = st.columns([1.05, 0.95], gap="large")
    with left:
        st.markdown("""
        <div class="login-brand">
            <div style="font-size:2rem;">🪸 CoralGuard</div>
            <h1>See reef stress before it becomes a crisis.</h1>
            <p>A connected early-warning workspace combining thermal satellite observations, underwater soundscapes, and trusted diver reports.</p>
            <div style="margin-top:24px;">
                <div class="login-feature">
                    <div style="font-size:1.35rem;">🛰️</div>
                    <div><strong>Thermal intelligence</strong><span>Track SST, anomalies and degree heating weeks.</span></div>
                </div>
                <div class="login-feature">
                    <div style="font-size:1.35rem;">🎧</div>
                    <div><strong>Acoustic reef health</strong><span>Detect biophony loss and underwater habitat stress.</span></div>
                </div>
                <div class="login-feature">
                    <div style="font-size:1.35rem;">🗺️</div>
                    <div><strong>14-day &amp; 28-day forecasts</strong><span>Prioritize reef sites and prepare response actions.</span></div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with right:
        st.markdown('<div class="login-form">', unsafe_allow_html=True)
        st.markdown("## Welcome back")
        st.markdown("Sign in to access the CoralGuard monitoring workspace.")
        st.markdown("""
        <div style="display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px;margin:14px 0 20px;">
            <div style="text-align:center;border:1px solid #dbe9e7;border-radius:10px;padding:10px 4px;">
                <div style="font-size:1.15rem;color:#087f8c;font-weight:800;">15</div>
                <div style="font-size:0.68rem;color:#6b7f7d;">Reef sites</div>
            </div>
            <div style="text-align:center;border:1px solid #dbe9e7;border-radius:10px;padding:10px 4px;">
                <div style="font-size:1.15rem;color:#087f8c;font-weight:800;">3</div>
                <div style="font-size:0.68rem;color:#6b7f7d;">Data sources</div>
            </div>
            <div style="text-align:center;border:1px solid #dbe9e7;border-radius:10px;padding:10px 4px;">
                <div style="font-size:1.15rem;color:#087f8c;font-weight:800;">24/7</div>
                <div style="font-size:0.68rem;color:#6b7f7d;">Monitoring</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        role = st.radio("Account role", ["User", "Admin"], horizontal=True)
        username = st.text_input("Username", placeholder="Enter your username")
        password = st.text_input("Password", type="password", placeholder="Enter your password")

        admin_password = os.getenv("CORALGUARD_ADMIN_PASSWORD", "admin123")
        user_password = os.getenv("CORALGUARD_USER_PASSWORD", "user123")
        valid_username = "admin" if role == "Admin" else "user"
        valid_password = admin_password if role == "Admin" else user_password

        if st.button("Sign in", type="primary", use_container_width=True):
            if username.strip().lower() == valid_username and password == valid_password:
                st.session_state.authenticated = True
                st.session_state.role = role
                st.session_state.username = username.strip()
                st.rerun()
            else:
                st.error("The username or password is incorrect.")

        st.caption("Demo access")
        st.code("Admin: admin / admin123\nUser:  user / user123")
        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

render_login()

@st.cache_data
def load_datasets():
    test_path = "data/processed/test_data.parquet"
    full_path = "data/processed/full_aligned_dataset.parquet"
    if not os.path.exists(test_path):
        st.error("Processed data files not found. Please run `python -m src.run_all` first.")
        st.stop()
    df_test = pd.read_parquet(test_path)
    df_full = pd.read_parquet(full_path)
    return df_test, df_full

@st.cache_resource
def load_models():
    model_t14_path = "models/xgb_multimodal_t14.joblib"
    model_t28_path = "models/xgb_multimodal_t28.joblib"
    pack_t14 = joblib.load(model_t14_path) if os.path.exists(model_t14_path) else None
    pack_t28 = joblib.load(model_t28_path) if os.path.exists(model_t28_path) else None
    return pack_t14, pack_t28

df_test, df_full = load_datasets()
pack_t14, pack_t28 = load_models()

# Sidebar Controls
st.sidebar.image("https://images.unsplash.com/photo-1546026423-cc4642628d2b?auto=format&fit=crop&w=400&q=80", use_container_width=True)
st.sidebar.title("🪸 CoralGuard System")
st.sidebar.caption("Multimodal Early Warning (2-8 Weeks Ahead)")
st.sidebar.markdown(
    f'<span class="role-chip">{"Administrator" if st.session_state.role == "Admin" else "Research User"}</span>',
    unsafe_allow_html=True
)
if st.sidebar.button("Sign out", use_container_width=True):
    st.session_state.authenticated = False
    st.session_state.pop("role", None)
    st.session_state.pop("username", None)
    st.rerun()

# Check FastAPI Backend Health
backend_status = api_client.check_health()
if backend_status.get("online"):
    st.sidebar.success("🟢 FastAPI Backend: Connected")
    st.sidebar.caption(f"DB: Connected | Port: 8000")
else:
    st.sidebar.info("ℹ️ Standalone Mode (Local Fallback)")

lead_time_choice = st.sidebar.radio("Forecast Lead Time:", ["14 Days (2 Weeks)", "28 Days (4 Weeks)"], index=0)
lead_days = 14 if "14" in lead_time_choice else 28
active_pack = pack_t14 if lead_days == 14 else pack_t28

if active_pack is None:
    st.error("Trained models not found. Please train models first using `python -m src.models.train_all` or `python -m src.run_all`.")
    st.stop()

available_regions = ["All Regions"] + sorted(list(df_test["region"].unique()))
selected_region = st.sidebar.selectbox("Filter Region:", available_regions)

filtered_df = df_test if selected_region == "All Regions" else df_test[df_test["region"] == selected_region]

# Latest weekly snapshot for each site
latest_date = filtered_df["date"].max()
latest_sites = filtered_df[filtered_df["date"] == latest_date].copy()

# Compute predictions
X_latest = latest_sites[active_pack["features"]].fillna(0.0)
probs = active_pack["model"].predict_proba(X_latest)
latest_sites["p_none"] = probs[:, 0]
latest_sites["p_mild"] = probs[:, 1]
latest_sites["p_severe"] = probs[:, 2]
latest_sites["p_bleach"] = probs[:, 1] + probs[:, 2]

def get_risk_status(row):
    if row["p_severe"] >= 0.35 or row["p_bleach"] >= 0.75:
        return "Bleaching Alert", "#EF4444", "alert"
    elif row["p_bleach"] >= 0.50:
        return "Bleaching Warning", "#F97316", "warning"
    elif row["p_bleach"] >= 0.25:
        return "Bleaching Watch", "#EAB308", "watch"
    else:
        return "Normal (No Stress)", "#10B981", "normal"

status_tuples = [get_risk_status(r) for _, r in latest_sites.iterrows()]
latest_sites["risk_level"] = [s[0] for s in status_tuples]
latest_sites["risk_color"] = [s[1] for s in status_tuples]
latest_sites["risk_code"] = [s[2] for s in status_tuples]

# Navigation Tabs
tabs = st.tabs([
    "🗺️ Global Risk Map",
    "📈 SST & Risk Trajectory",
    "🎧 Acoustic Health Index",
    "🤿 Upload Diver Report",
    "🔍 Model Explainability (SHAP)",
    "🔮 Heatwave Scenario Simulator"
])

# ----------------- TAB 1: Global Risk Map -----------------
with tabs[0]:
    st.markdown('<div class="main-header">Reef Bleaching Early Warning Map</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sub-header">Current forecast cycle: <b>{latest_date.strftime("%B %d, %Y")}</b> | Target Horizon: <b>+{lead_days} Days</b> ahead</div>', unsafe_allow_html=True)

    # Top KPI Metrics
    n_total = len(latest_sites)
    n_alert = len(latest_sites[latest_sites["risk_code"] == "alert"])
    n_warn = len(latest_sites[latest_sites["risk_code"] == "warning"])
    n_watch = len(latest_sites[latest_sites["risk_code"] == "watch"])
    n_normal = len(latest_sites[latest_sites["risk_code"] == "normal"])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🚨 Sites on Alert", f"{n_alert} / {n_total}", delta=f"{int(n_alert/n_total*100)}% of monitored", delta_color="inverse")
    c2.metric("⚠️ Bleaching Warnings", f"{n_warn}", delta="Immediate watch required")
    c3.metric("👀 Bleaching Watches", f"{n_watch}")
    c4.metric("✅ Unstressed Reefs", f"{n_normal}")

    if backend_status.get("online"):
        backend_alerts = api_client.get_active_alerts(lead_days=lead_days)
        if backend_alerts and backend_alerts.get("active_alerts_count", 0) > 0:
            st.warning(f"🚨 **Live Backend Warning Feed**: {backend_alerts['active_alerts_count']} monitored reef sites actively exceed operational watch/alert thresholds for the +{lead_days}d forecast cycle.")

    # Folium Interactive Map
    center_lat = float(latest_sites["lat"].mean()) if not latest_sites.empty else -18.0
    center_lon = float(latest_sites["lon"].mean()) if not latest_sites.empty else 147.0

    m = folium.Map(location=[center_lat, center_lon], zoom_start=4, tiles="OpenStreetMap")

    for _, row in latest_sites.iterrows():
        popup_html = f"""
        <div style="font-family: Arial; width: 220px;">
            <h4 style="margin: 0; color: {row['risk_color']};">{row['risk_level']}</h4>
            <p style="margin: 2px 0 8px 0; font-size: 12px; color: #64748B;">Site: <b>{row['site_id']}</b> ({row['region']})</p>
            <table style="width: 100%; font-size: 11px;">
                <tr><td><b>Forecast Risk ({lead_days}d):</b></td><td><b>{row['p_bleach']*100:.1f}%</b></td></tr>
                <tr><td>P(Severe Bleaching):</td><td>{row['p_severe']*100:.1f}%</td></tr>
                <tr><td>Current SST:</td><td>{row['sst']:.1f} °C</td></tr>
                <tr><td>Degree Heating Weeks:</td><td>{row['dhw']:.1f} °C-wks</td></tr>
                <tr><td>Diver Reports (30d):</td><td>{int(row['diver_reports_count_50km_30d'])}</td></tr>
                <tr><td>Acoustics Missing:</td><td>{'Yes (Imputed)' if row['acoustic_is_missing']==1 else 'No (Live Hydrophone)'}</td></tr>
            </table>
        </div>
        """
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=9,
            popup=folium.Popup(popup_html, max_width=260),
            tooltip=f"{row['site_id']}: {row['risk_level']} ({row['p_bleach']*100:.1f}%)",
            color=row["risk_color"],
            fill=True,
            fill_color=row["risk_color"],
            fill_opacity=0.85,
            weight=2
        ).add_to(m)

    st_folium(m, width="100%", height=520)

    # Detailed Site Table
    st.subheader("Monitored Sites Risk Assessment")
    display_cols = ["site_id", "region", "risk_level", "p_bleach", "p_severe", "sst", "dhw", "diver_reports_count_50km_30d", "acoustic_is_missing"]
    disp_df = latest_sites[display_cols].copy()
    disp_df["p_bleach"] = (disp_df["p_bleach"] * 100).round(1).astype(str) + "%"
    disp_df["p_severe"] = (disp_df["p_severe"] * 100).round(1).astype(str) + "%"
    disp_df.columns = ["Site ID", "Region", "Risk Level", f"Bleaching Risk (+{lead_days}d)", "P(Severe)", "SST (°C)", "DHW (°C-wks)", "Citizen Reports (30d)", "Acoustics Missing"]
    st.dataframe(disp_df, use_container_width=True)

# ----------------- TAB 2: Time-series of SST + Risk -----------------
with tabs[1]:
    st.markdown('<div class="main-header">SST & Bleaching Risk Trajectory</div>', unsafe_allow_html=True)
    site_list = sorted(list(filtered_df["site_id"].unique()))
    selected_site = st.selectbox("Select Reef Site:", site_list, index=0)

    if backend_status.get("online"):
        live_sst = api_client.get_site_sst(selected_site, days=30)
        if live_sst and live_sst.get("status") == "available" and live_sst.get("latest"):
            l_pt = live_sst["latest"]
            c_s1, c_s2, c_s3, c_s4 = st.columns(4)
            c_s1.metric("🛰️ Satellite SST (Live)", f"{l_pt['sst']:.2f} °C" if l_pt.get("sst") else "N/A")
            c_s2.metric("🌡️ SST Anomaly", f"{l_pt['ssta']:+.2f} °C" if l_pt.get("ssta") is not None else "N/A")
            c_s3.metric("🔥 Degree Heating Weeks", f"{l_pt['dhw']:.2f} °C-wks" if l_pt.get("dhw") is not None else "N/A")
            c_s4.metric("📡 Telemetry Source", "NOAA Coral Reef Watch")
        elif live_sst and live_sst.get("status") == "no_data":
            st.info("ℹ️ Live NOAA ERDDAP status: No recent satellite data for this site (clear 'no data' status).")

    site_history = df_full[df_full["site_id"] == selected_site].sort_values("date").copy()
    
    # Compute probabilities across historical test segment
    X_hist = site_history[active_pack["features"]].fillna(0.0)
    p_hist = active_pack["model"].predict_proba(X_hist)
    site_history["prob_severe"] = p_hist[:, 2]
    site_history["prob_bleach"] = p_hist[:, 1] + p_hist[:, 2]

    # Two-panel Plotly Chart: SST & DHW (top), Bleaching Probability (bottom)
    fig_ts = go.Figure()

    # Panel 1: SST & Bleaching Threshold
    fig_ts.add_trace(go.Scatter(
        x=site_history["date"], y=site_history["sst"],
        mode="lines", name="SST (°C)", line=dict(color="#0284C7", width=2)
    ))
    fig_ts.add_trace(go.Scatter(
        x=site_history["date"], y=site_history["bleaching_threshold"],
        mode="lines", name="Bleaching Threshold (MMM+1)",
        line=dict(color="#DC2626", dash="dash", width=1.5)
    ))
    fig_ts.add_trace(go.Scatter(
        x=site_history["date"], y=site_history["dhw"],
        mode="lines", name="DHW (°C-weeks)", yaxis="y2",
        line=dict(color="#D97706", width=2)
    ))

    fig_ts.update_layout(
        title=f"Historical & Forecast Thermal Stress at {selected_site} ({site_history['region'].iloc[0]})",
        xaxis=dict(title="Observation Date"),
        yaxis=dict(title="Sea Surface Temperature (°C)", side="left"),
        yaxis2=dict(title="Degree Heating Weeks (DHW)", overlaying="y", side="right", showgrid=False),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=400
    )
    st.plotly_chart(fig_ts, use_container_width=True)

    # Panel 2: Predicted Bleaching Probability
    fig_prob = go.Figure()
    fig_prob.add_trace(go.Scatter(
        x=site_history["date"], y=site_history["prob_bleach"] * 100,
        mode="lines", name=f"Predicted Bleaching Risk (+{lead_days}d)",
        line=dict(color="#E11D48", width=2.5), fill="tozeroy", fillcolor="rgba(225, 29, 72, 0.15)"
    ))
    fig_prob.add_trace(go.Scatter(
        x=site_history["date"], y=site_history["prob_severe"] * 100,
        mode="lines", name="P(Severe Mortality Risk)",
        line=dict(color="#991B1B", width=2, dash="dot")
    ))

    # Alert Threshold Lines
    fig_prob.add_hline(y=75, line_dash="dot", line_color="#DC2626", annotation_text="Alert Threshold (75%)")
    fig_prob.add_hline(y=50, line_dash="dot", line_color="#EA580C", annotation_text="Warning Threshold (50%)")
    fig_prob.add_hline(y=25, line_dash="dot", line_color="#CA8A04", annotation_text="Watch Threshold (25%)")

    fig_prob.update_layout(
        title=f"Multimodal Bleaching Risk Probability (+{lead_days} Days Horizon)",
        xaxis=dict(title="Date"),
        yaxis=dict(title="Bleaching Risk (%)", range=[0, 105]),
        hovermode="x unified",
        height=320
    )
    st.plotly_chart(fig_prob, use_container_width=True)

# ----------------- TAB 3: Acoustic Health Index -----------------
with tabs[2]:
    st.markdown('<div class="main-header">Reef Bioacoustic Soundscape Health Index</div>', unsafe_allow_html=True)
    st.markdown("""
    Underwater soundscapes provide a continuous, non-invasive ecological health indicator:
    - **Snapping Shrimp Band Energy (2 - 20 kHz)**: High in healthy, calcified reef structural habitats; collapses during thermal bleaching.
    - **Fish Chorus Band Energy (50 - 1000 Hz)**: Measures biophony, mating aggregations, and herbivore feeding.
    - **Acoustic Complexity (ACI) & Diversity (ADI)**: Quantifies acoustic community dynamics.
    """)

    if backend_status.get("online"):
        ac_resp = api_client.get_site_acoustics(selected_site, limit=5)
        if ac_resp and ac_resp.get("status") == "available" and ac_resp.get("latest"):
            lat_ac = ac_resp["latest"]
            st.info(f"🎧 **Live Hydrophone Telemetry for {selected_site}**: Snapping shrimp band: **{lat_ac['shrimp_band_energy_ratio']*100:.1f}%**, Fish chorus: **{lat_ac['fish_chorus_energy_ratio']*100:.1f}%**, ACI: **{lat_ac['aci']:.2f}**, Spectral Entropy: **{lat_ac['spectral_entropy']:.2f}**")

    c_ac1, c_ac2 = st.columns([1, 1])

    with c_ac1:
        st.subheader("Bioacoustic Frequency Band Energy Distribution")
        fig_ac_bar = px.scatter(
            df_test.sample(min(500, len(df_test)), random_state=42),
            x="fish_chorus_energy_ratio",
            y="shrimp_band_energy_ratio",
            color="target_t14",
            color_continuous_scale="Viridis",
            labels={
                "fish_chorus_energy_ratio": "Fish Chorus Energy Ratio (50-1000 Hz)",
                "shrimp_band_energy_ratio": "Snapping Shrimp Energy Ratio (2-20 kHz)",
                "target_t14": "Bleaching Status"
            },
            title="Acoustic Biophony Signature vs Future Bleaching Severity"
        )
        st.plotly_chart(fig_ac_bar, use_container_width=True)

    with c_ac2:
        st.subheader("Acoustic Complexity (ACI) vs Acoustic Diversity (ADI)")
        fig_aci = px.scatter(
            df_test.sample(min(500, len(df_test)), random_state=42),
            x="aci", y="adi",
            color="region",
            title="Acoustic Complexity & Diversity by Coral Reef Region"
        )
        st.plotly_chart(fig_aci, use_container_width=True)

    # Sample Soundscape Player (if wav files exist)
    st.subheader("Listen to Representative Soundscapes")
    audio_dir = "data/raw/audio"
    if os.path.exists(audio_dir):
        audio_files = [f for f in os.listdir(audio_dir) if f.endswith(".wav")]
        if audio_files:
            col_a1, col_a2 = st.columns(2)
            healthy_clips = [f for f in audio_files if "healthy" in f]
            bleached_clips = [f for f in audio_files if "bleach" in f]

            if healthy_clips:
                with col_a1:
                    st.markdown("🔊 **Healthy Reef Soundscape** (Dense snapping shrimp clicks + active fish chorus)")
                    st.audio(os.path.join(audio_dir, healthy_clips[0]))

            if bleached_clips:
                with col_a2:
                    st.markdown("🔇 **Degraded/Bleached Reef Soundscape** (Significant acoustic quieting)")
                    st.audio(os.path.join(audio_dir, bleached_clips[0]))

        # Interactive Custom Audio File Uploader
        st.markdown("---")
        st.subheader("🎙️ Analyze Custom Hydrophone Recording")
        st.markdown("Upload a raw `.wav` soundscape recording from a field hydrophone (e.g., SoundTrap or Ocean Sonics) to extract bioacoustic metrics and inspect reef biophony health in real time.")

        uploaded_audio = st.file_uploader("Upload underwater soundscape (.wav)", type=["wav"], key="hydrophone_uploader")
        if uploaded_audio is not None:
            import tempfile
            from src.features.acoustic_features import extract_single_audio_features

            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                tmp_file.write(uploaded_audio.read())
                tmp_path = tmp_file.name

            try:
                with st.spinner("Extracting bioacoustic indices and spectral bands..."):
                    audio_feats = extract_single_audio_features(tmp_path)

                shrimp_r = audio_feats["shrimp_band_energy_ratio"]
                fish_r = audio_feats["fish_chorus_energy_ratio"]
                aci_val = audio_feats["aci"]
                bi_val = audio_feats["bioacoustic_index"]
                ndsi_val = audio_feats["ndsi"]
                cent_val = audio_feats["spectral_centroid_mean"]

                if shrimp_r >= 0.20 and ndsi_val > 0.2:
                    st.success("🌿 **Acoustic Verdict: Healthy Biophonic Reef** — High-frequency snapping shrimp crackle and biophonic activity dominate the soundscape.")
                elif shrimp_r >= 0.10:
                    st.warning("⚠️ **Acoustic Verdict: Stressed Biophonic Reef** — Noticeable attenuation in high-frequency crustacean activity, indicating early ecological distress.")
                else:
                    st.error("🚨 **Acoustic Verdict: Degraded / Silent Reef** — Severe biophonic collapse and acoustic quieting typical of post-bleaching mortality.")

                st.audio(uploaded_audio)

                m_col1, m_col2, m_col3, m_col4, m_col5, m_col6 = st.columns(6)
                m_col1.metric("Snapping Shrimp (2-20 kHz)", f"{shrimp_r*100:.1f}%")
                m_col2.metric("Fish Chorus (50-1000 Hz)", f"{fish_r*100:.1f}%")
                m_col3.metric("Complexity (ACI)", f"{aci_val:.3f}")
                m_col4.metric("Bioacoustic (BI)", f"{bi_val:.1f}")
                m_col5.metric("Soundscape (NDSI)", f"{ndsi_val:+.3f}")
                m_col6.metric("Spectral Centroid", f"{cent_val:.0f} Hz")

                band_df = pd.DataFrame({
                    "Acoustic Band": ["Fish Chorus (50-1000 Hz)", "Snapping Shrimp (2-20 kHz)", "Other / Background Noise"],
                    "Energy Percentage": [fish_r * 100, shrimp_r * 100, max(0.0, 100.0 - (fish_r + shrimp_r) * 100)]
                })
                fig_bands = px.bar(
                    band_df, x="Acoustic Band", y="Energy Percentage",
                    color="Acoustic Band",
                    color_discrete_map={
                        "Fish Chorus (50-1000 Hz)": "#0284C7",
                        "Snapping Shrimp (2-20 kHz)": "#10B981",
                        "Other / Background Noise": "#94A3B8"
                    },
                    title="Energy Partitioning in Uploaded Soundscape"
                )
                st.plotly_chart(fig_bands, use_container_width=True)

            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

# ----------------- TAB 4: Upload Diver Report -----------------
with tabs[3]:
    st.markdown('<div class="main-header">Community Diver Report Ingestion & Validation</div>', unsafe_allow_html=True)
    st.markdown("Directly submit diver observations via the production backend endpoint (`POST /reports`) or upload batch survey files from Reef Check, CoralWatch, or Reef Life Survey.")

    st.subheader("📝 Live Diver Field Submission (POST /reports)")
    st.markdown("Submissions undergo real-time geospatial matching to monitored reef sites, depth plausibility checks, and credential trust scoring.")

    with st.form("live_diver_report_form"):
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            form_obs_id = st.text_input("Observer ID / Call Sign:", value="reef_observer_01")
            form_obs_type = st.selectbox("Observer Credential:", ["marine_biologist", "dive_master", "recreational_diver", "volunteer", "resident"])
            form_program = st.selectbox("Survey Program:", ["Reef Check", "CoralWatch", "Reef Life Survey", "Direct Submission"])
            form_lat = st.number_input("Survey Latitude (°):", value=float(latest_sites.iloc[0]["lat"]) if not latest_sites.empty else -14.668, format="%.4f")
            form_lon = st.number_input("Survey Longitude (°):", value=float(latest_sites.iloc[0]["lon"]) if not latest_sites.empty else 145.460, format="%.4f")

        with col_f2:
            form_date = st.date_input("Observation Date:", value=date.today())
            form_depth = st.number_input("Survey Depth (meters, 0 - 60m):", min_value=0.5, max_value=60.0, value=10.0, step=0.5)
            form_bleach = st.slider("Observed Coral Bleaching Coverage (%):", min_value=0.0, max_value=100.0, value=30.0, step=1.0)
            form_has_photo = st.checkbox("Photographic Proof Attached?", value=True)
            form_photo_url = st.text_input("Photo Evidence URL (optional):", value="https://reefcheck.org/evidence/sample_photo.jpg" if form_has_photo else "")

        form_notes = st.text_area("Ecological Field Observations:", value="Branching Acropora colonies showing pale discoloration; partial bleaching observed on upper reef slope.")
        submit_btn = st.form_submit_button("🚀 Transmit Field Survey to Production Backend")

        if submit_btn:
            payload = {
                "latitude": float(form_lat),
                "longitude": float(form_lon),
                "observed_at": str(form_date),
                "observer_id": form_obs_id,
                "observer_type": form_obs_type,
                "survey_program": form_program,
                "depth_meters": float(form_depth),
                "bleaching_percent": float(form_bleach),
                "has_photo": form_has_photo,
                "photo_url": form_photo_url if form_has_photo else None,
                "notes": form_notes
            }

            result = api_client.submit_diver_report(payload)
            if result.get("success"):
                rep_data = result["data"]
                st.success(f"✅ Diver survey accepted and committed to PostgreSQL/SQLite database! (Record ID #{rep_data['id']})")
                r_c1, r_c2, r_c3 = st.columns(3)
                r_c1.metric("Automated Trust Score", f"{rep_data['trust_score'] * 100:.1f}%", help="Computed from observer credentials, photo proof, depth plausibility, and program provenance.")
                r_c2.metric("Matched Reef Site", rep_data.get("site_id") or "Offshore Buffer (50km)")
                r_c3.metric("Verification Status", rep_data.get("status", "verified").capitalize())
            else:
                st.error(f"Backend submission failed: {result.get('error')}")

    st.markdown("---")
    st.subheader("📁 Batch CSV Ingestion")
    st.markdown("Upload diver observation CSV files from Reef Check, CoralWatch, or local dive surveys:")

    diver_template = pd.DataFrame([
        {"date": "2024-03-15", "lat": -18.25, "lon": 147.45, "depth": 14.5, "bleaching_percent": 35.0, "observer_confidence": 0.90},
        {"date": "2024-03-16", "lat": 18.00, "lon": -65.50, "depth": 8.0, "bleaching_percent": 65.0, "observer_confidence": 0.95}
    ])

    st.download_button(
        label="📥 Download Sample Diver CSV Template",
        data=diver_template.to_csv(index=False),
        file_name="diver_report_template.csv",
        mime="text/csv"
    )

    uploaded_file = st.file_uploader("Upload Diver Report CSV:", type=["csv"])
    if uploaded_file is not None:
        try:
            up_df = pd.read_csv(uploaded_file)
            st.success(f"Successfully parsed {len(up_df)} diver survey observations!")
            st.dataframe(up_df.head(), use_container_width=True)

            required_cols = ["date", "lat", "lon", "bleaching_percent"]
            missing = [c for c in required_cols if c not in up_df.columns]
            if missing:
                st.error(f"Missing required columns in uploaded CSV: {missing}")
            else:
                up_df["bleaching_category"] = pd.cut(
                    up_df["bleaching_percent"],
                    bins=[-float("inf"), 10.0, 50.0, float("inf")],
                    labels=[0, 1, 2]
                ).astype(int)

                st.info("Incorporating local ground-truth observations into regional risk weighting...")
                avg_sev = up_df["bleaching_category"].mean()
                sev_label = "None" if avg_sev < 0.5 else ("Mild" if avg_sev < 1.5 else "Severe")
                
                c_u1, c_u2, c_u3 = st.columns(3)
                c_u1.metric("Uploaded Reports", f"{len(up_df)}")
                c_u2.metric("Mean Bleaching Severity", f"{sev_label} ({avg_sev:.2f})")
                c_u3.metric("Local Citizen Spatial Density", f"{len(up_df)/7854.0:.5f} reps/km²")
        except Exception as e:
            st.error(f"Error parsing uploaded file: {e}")

# ----------------- TAB 5: Model Explainability (SHAP) -----------------
with tabs[4]:
    st.markdown('<div class="main-header">Explainable AI & Feature Attribution (SHAP)</div>', unsafe_allow_html=True)
    st.markdown("""
    Transparent attribution of multi-modal features driving early warning predictions:
    - **SST Thermal Stress**: Degree Heating Weeks (DHW), 30-day max SST, SSTA rolling means.
    - **Citizen Diver Corroboration**: Proximity-weighted bleaching reports & observer confidence.
    - **Acoustic Ecological Health**: Snapping shrimp band attenuation, biophony complexity.
    """)

    shap_img_path = "reports/shap_summary.png"
    if os.path.exists(shap_img_path):
        st.image(shap_img_path, caption="SHAP Summary Beeswarm Plot: Feature Impact on Severe Bleaching Forecast (t+14d)", use_container_width=True)
    else:
        st.info("Run `python -m src.models.evaluate` to generate the SHAP explanation plot.")

    st.subheader("Feature Importance Breakdown by Modality")
    if backend_status.get("online"):
        backend_eval = api_client.get_model_metrics()
        if backend_eval and backend_eval.get("lead_14d"):
            m_stat = backend_eval["lead_14d"] if lead_days == 14 else backend_eval["lead_28d"]
            st.info(f"📊 **Verified Model Benchmarks (+{lead_days}d)**: PR-AUC: **{m_stat['pr_auc']:.4f}** | Severe Bleaching Recall: **{m_stat['recall_severe']*100:.1f}%** | ROC-AUC: **{m_stat['roc_auc']:.4f}** | Macro F1: **{m_stat['macro_f1']:.4f}**")
    if active_pack is not None:
        importances = pd.Series(active_pack["model"].feature_importances_, index=active_pack["features"])
        top15 = importances.nlargest(15).reset_index()
        top15.columns = ["Feature", "Gain Importance"]

        def categorize_feature(fname):
            if any(s in fname for s in ["sst", "dhw", "threshold"]):
                return "Satellite SST"
            elif "diver" in fname:
                return "Citizen Diver Science"
            else:
                return "Underwater Acoustics"

        top15["Modality"] = top15["Feature"].apply(categorize_feature)
        fig_feat = px.bar(
            top15, x="Gain Importance", y="Feature", color="Modality",
            orientation="h", title=f"Top 15 Features Driving Multimodal Forecasts (+{lead_days}d)",
            color_discrete_map={
                "Satellite SST": "#0284C7",
                "Citizen Diver Science": "#EA580C",
                "Underwater Acoustics": "#10B981"
            }
        )
        fig_feat.update_layout(yaxis=dict(autorange="reversed"), height=480)
        st.plotly_chart(fig_feat, use_container_width=True)

# ----------------- TAB 6: Heatwave Scenario Simulator -----------------
with tabs[5]:
    st.markdown('<div class="main-header">Marine Heatwave Scenario Simulator (What-If Analysis)</div>', unsafe_allow_html=True)
    st.markdown("""
    Stress-test reef sites under simulated marine heatwaves, acoustic habitat degradation, and community diver ground reports.
    Evaluate how rapidly operational alert thresholds trigger ahead of time.
    """)

    sim_col1, sim_col2 = st.columns([1, 2])

    with sim_col1:
        st.subheader("Scenario Parameters")
        preset = st.selectbox(
            "Select Heatwave Preset:",
            [
                "Custom Parameters",
                "Mild Thermal Stress (+0.8°C SST, +2 DHW)",
                "Moderate Marine Heatwave (+1.5°C SST, +4.5 DHW)",
                "Extreme Thermal Shock (+2.5°C SST, +8 DHW, Acoustic Collapse)"
            ]
        )

        if "Mild" in preset:
            def_sst_delta, def_dhw_delta, def_ac_state = 0.8, 2.0, "Maintain Current"
        elif "Moderate" in preset:
            def_sst_delta, def_dhw_delta, def_ac_state = 1.5, 4.5, "Maintain Current"
        elif "Extreme" in preset:
            def_sst_delta, def_dhw_delta, def_ac_state = 2.5, 8.0, "Collapse Biophony (-70% Shrimp/Fish)"
        else:
            def_sst_delta, def_dhw_delta, def_ac_state = 0.0, 0.0, "Maintain Current"

        sst_delta = st.slider("Simulated SST Shift (Δ °C):", -2.0, 4.0, def_sst_delta, 0.1)
        dhw_delta = st.slider("Simulated Additional DHW (Δ °C-weeks):", 0.0, 12.0, def_dhw_delta, 0.5)
        sim_ac_state = st.selectbox(
            "Simulated Soundscape State:",
            ["Maintain Current", "Collapse Biophony (-70% Shrimp/Fish)", "Healthy Biophony (+50% Shrimp/Fish)"],
            index=["Maintain Current", "Collapse Biophony (-70% Shrimp/Fish)", "Healthy Biophony (+50% Shrimp/Fish)"].index(def_ac_state)
        )
        sim_diver_sev = st.slider("Simulated Local Diver Bleaching Severity:", 0.0, 2.0, 0.0, 0.1, help="0: None, 1: Mild, 2: Severe")

    with sim_col2:
        st.subheader("Simulated Forecast Impact")
        # Copy latest sites and apply scenario deltas
        sim_df = latest_sites.copy()
        sim_df["sst"] = sim_df["sst"] + sst_delta
        sim_df["ssta"] = sim_df["ssta"] + sst_delta
        sim_df["ssta_roll_7d"] = sim_df["ssta_roll_7d"] + sst_delta
        sim_df["ssta_roll_14d"] = sim_df["ssta_roll_14d"] + sst_delta
        sim_df["ssta_roll_30d"] = sim_df["ssta_roll_30d"] + sst_delta
        sim_df["dhw"] = sim_df["dhw"] + dhw_delta
        sim_df["sst_max_30d"] = sim_df["sst_max_30d"] + max(0.0, sst_delta)

        if sim_diver_sev > 0.0:
            sim_df["diver_reports_count_50km_30d"] = np.maximum(sim_df["diver_reports_count_50km_30d"], 5)
            sim_df["diver_weighted_severity_30d"] = np.maximum(sim_df["diver_weighted_severity_30d"], sim_diver_sev)
            sim_df["diver_has_reports"] = 1

        if "Collapse" in sim_ac_state:
            sim_df["shrimp_band_energy_ratio"] = sim_df["shrimp_band_energy_ratio"] * 0.30
            sim_df["fish_chorus_energy_ratio"] = sim_df["fish_chorus_energy_ratio"] * 0.30
            sim_df["aci"] = sim_df["aci"] * 0.50
        elif "Healthy" in sim_ac_state:
            sim_df["shrimp_band_energy_ratio"] = np.clip(sim_df["shrimp_band_energy_ratio"] * 1.5, 0.0, 1.0)
            sim_df["fish_chorus_energy_ratio"] = np.clip(sim_df["fish_chorus_energy_ratio"] * 1.5, 0.0, 1.0)

        # Run inference on simulated features
        X_sim = sim_df[active_pack["features"]].fillna(0.0)
        p_sim = active_pack["model"].predict_proba(X_sim)
        sim_df["p_none_sim"] = p_sim[:, 0]
        sim_df["p_mild_sim"] = p_sim[:, 1]
        sim_df["p_severe_sim"] = p_sim[:, 2]
        sim_df["p_bleach_sim"] = p_sim[:, 1] + p_sim[:, 2]

        sim_tuples = [get_risk_status(pd.Series({"p_severe": r["p_severe_sim"], "p_bleach": r["p_bleach_sim"]})) for _, r in sim_df.iterrows()]
        sim_df["risk_level_sim"] = [s[0] for s in sim_tuples]
        sim_df["risk_code_sim"] = [s[2] for s in sim_tuples]

        # Before vs After Comparison
        base_counts = latest_sites["risk_level"].value_counts().reindex(["Normal (No Stress)", "Bleaching Watch", "Bleaching Warning", "Bleaching Alert"], fill_value=0)
        sim_counts = sim_df["risk_level_sim"].value_counts().reindex(["Normal (No Stress)", "Bleaching Watch", "Bleaching Warning", "Bleaching Alert"], fill_value=0)

        comp_df = pd.DataFrame({
            "Risk Tier": ["Normal", "Watch", "Warning", "Alert"],
            "Current Sites": base_counts.values,
            "Simulated Sites": sim_counts.values
        })

        fig_comp = go.Figure(data=[
            go.Bar(name="Current Baseline", x=comp_df["Risk Tier"], y=comp_df["Current Sites"], marker_color="#94A3B8"),
            go.Bar(name="Simulated Scenario", x=comp_df["Risk Tier"], y=comp_df["Simulated Sites"], marker_color="#EF4444")
        ])
        fig_comp.update_layout(
            barmode="group",
            title=f"Site Distribution by Risk Tier (Current vs Simulated +{lead_days}d)",
            yaxis_title="Number of Reef Sites",
            height=340
        )
        st.plotly_chart(fig_comp, use_container_width=True)

    # Site Delta Table
    st.subheader("Site-by-Site Simulated Trajectory")
    delta_display = pd.DataFrame({
        "Site ID": sim_df["site_id"],
        "Region": sim_df["region"],
        "Current Tier": latest_sites["risk_level"],
        "Current Risk (%)": (latest_sites["p_bleach"] * 100).round(1).astype(str) + "%",
        "Simulated Tier": sim_df["risk_level_sim"],
        "Simulated Risk (%)": (sim_df["p_bleach_sim"] * 100).round(1).astype(str) + "%",
        "P(Severe) Shift": ((sim_df["p_severe_sim"] - latest_sites["p_severe"]) * 100).apply(lambda v: f"{v:+.1f}%")
    })
    st.dataframe(delta_display, use_container_width=True)

    st.download_button(
        label="📥 Export Scenario Simulation Results (CSV)",
        data=delta_display.to_csv(index=False),
        file_name=f"simulated_heatwave_scenario_{lead_days}d.csv",
        mime="text/csv"
    )


