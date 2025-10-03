# app.py
# heavy_metal_dashboard.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from fpdf import FPDF  # fpdf2
import io
import tempfile
import os

# ---------------- Dark/Light Mode Toggle ----------------
if "dark_mode" not in st.session_state:
    st.session_state["dark_mode"] = False

# Use toggle if available; fallback to checkbox for older Streamlit
try:
    st.session_state["dark_mode"] = st.sidebar.toggle(
        "🌙 Dark Mode",
        value=st.session_state["dark_mode"],
    )
except Exception:
    st.session_state["dark_mode"] = st.sidebar.checkbox(
        "🌙 Dark Mode",
        value=st.session_state["dark_mode"],
    )

dark_mode = st.session_state["dark_mode"]
bg_color = "#1e1e1e" if dark_mode else "#ffffff"
text_color = "#ffffff" if dark_mode else "#000000"
plot_bg_color = "#1e1e1e" if dark_mode else "#f0f2f6"

st.markdown(f"""
    <style>
    .stApp {{ background-color: {bg_color}; color: {text_color}; }}
    h1, h2, h3, h4, h5, p, span, label {{ color: {text_color} !important; }}
    </style>
""", unsafe_allow_html=True)

st.title("🌿 HMPI: Heavy Metal Pollution Index Dashboard")

# ---------------- Data Input Module ----------------
st.subheader("Data Input")
input_method = st.radio("Choose input method:", ["Manual Entry", "Upload CSV/Excel", "Use Sample Data"])
metal_filter = ["Pb","Hg","Cd","As","Cr"]

if input_method == "Manual Entry":
    st.info("Enter site data manually")
    site_names = [s.strip() for s in st.text_area(
        "Site Names (comma separated)", "Site A, Site B, Site C"
    ).split(",") if s.strip()]
    num_sites = len(site_names)
    if num_sites == 0:
        st.error("Please enter at least one site name.")
        st.stop()

    data = {"Site": site_names}
    for metal in metal_filter:
        default_vals = ",".join(["0.05"] * num_sites)
        raw = st.text_area(f"{metal} concentrations (comma separated)", default_vals)
        vals = [v.strip() for v in raw.split(",") if v.strip() != ""]
        if len(vals) != num_sites:
            st.error(f"{metal}: Please provide exactly {num_sites} values.")
            st.stop()
        try:
            data[metal] = [float(v) for v in vals]
        except ValueError:
            st.error(f"{metal}: All values must be numeric.")
            st.stop()

    df = pd.DataFrame(data)
    df["Latitude"] = np.random.uniform(21.12, 21.18, num_sites)
    df["Longitude"] = np.random.uniform(78.60, 78.65, num_sites)

elif input_method == "Upload CSV/Excel":
    uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv","xlsx"])
    if uploaded_file:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
    else:
        st.stop()
else:
    st.info("Using built-in sample dataset")
    df = pd.DataFrame({
        "Site":["Site A","Site B","Site C","Site D"],
        "Pb":[0.05,0.07,0.03,0.06],
        "Hg":[0.01,0.02,0.015,0.025],
        "Cd":[0.015,0.02,0.01,0.03],
        "As":[0.004,0.006,0.005,0.003],
        "Cr":[0.02,0.03,0.025,0.022],
        "Latitude":[21.14,21.145,21.15,21.155],
        "Longitude":[78.61,78.62,78.625,78.63]
    })

# ---------------- Pre-processing ----------------
for metal in metal_filter:
    if metal not in df.columns:
        df[metal] = 0
df[metal_filter] = df[metal_filter].apply(pd.to_numeric, errors='coerce').fillna(0)
if "Latitude" not in df.columns:
    df["Latitude"] = np.random.uniform(21.12,21.18,len(df))
if "Longitude" not in df.columns:
    df["Longitude"] = np.random.uniform(78.60,78.65,len(df))

st.subheader("Pre-processed Data")
st.dataframe(df, use_container_width=True)

# ---------------- Calculation Engine ----------------
st.subheader("HPI & HEI Calculation")
safe_limits = {"Pb":0.06,"Hg":0.02,"Cd":0.02,"As":0.005,"Cr":0.025}

def calculate_hpi(row):
    Q_list = []
    W_list = []
    for metal in metal_filter:
        Ci = row[metal]
        Si = safe_limits[metal]
        Qi = (Ci / Si) * 100
        Wi = 1 / Si
        Q_list.append(Qi)
        W_list.append(Wi)
    return sum(q * w for q, w in zip(Q_list, W_list)) / sum(W_list)

def calculate_hei(row):
    return sum(row[m] / safe_limits[m] for m in metal_filter)

df["HPI"] = df.apply(calculate_hpi, axis=1)
df["HEI"] = df.apply(calculate_hei, axis=1)

def risk_level(hpi):
    if hpi <= 50:
        return "Low"
    elif hpi <= 100:
        return "Medium"
    else:
        return "High"

df["Risk Level"] = df["HPI"].apply(risk_level)
st.write(df[["Site"] + metal_filter + ["HPI","HEI","Risk Level"]])

# ---------------- Dashboard Metrics with Safe Limits ----------------
st.subheader("Dashboard Metrics")
safe_limits_text = ", ".join([f"{metal}: {safe_limits[metal]}" for metal in metal_filter])
st.markdown(f"<p style='color:{text_color}; font-weight:bold;'>Safe Limits → {safe_limits_text}</p>", unsafe_allow_html=True)

cols = st.columns(len(metal_filter))
for i, metal in enumerate(metal_filter):
    avg_val = round(df[metal].mean(), 4)
    delta_val = round(df[metal].max() - df[metal].min(), 4)
    if avg_val <= 0.8 * safe_limits[metal]:
        alert = ""
    elif avg_val <= safe_limits[metal]:
        alert = " ⚠️ Near Limit"
    else:
        alert = " ❌ High!"
    cols[i].metric(label=f"{metal} Avg", value=f"{avg_val}{alert}", delta=delta_val)

# ---------------- Charts ----------------
def plot_bar():
    fig = go.Figure()
    for metal in metal_filter:
        colors = ["green" if v <= safe_limits[metal] else "red" for v in df[metal]]
        fig.add_trace(go.Bar(x=df["Site"], y=df[metal], name=metal, marker_color=colors))
        fig.add_hline(y=safe_limits[metal], line_dash="dash", line_color="blue")
    fig.update_layout(
        title="Metal Concentration by Site",
        barmode='group',
        plot_bgcolor=plot_bg_color,
        paper_bgcolor=bg_color,
        font_color=text_color
    )
    return fig

def plot_line():
    fig = go.Figure()
    palette = px.colors.qualitative.Set2
    for metal in metal_filter:
        fig.add_trace(go.Scatter(
            x=df["Site"], y=df[metal], mode='lines+markers', name=metal,
            line=dict(color=np.random.choice(palette))
        ))
        fig.add_hline(y=safe_limits[metal], line_dash="dash", line_color="blue")
    fig.update_layout(
        title="Metal Trends Across Sites",
        plot_bgcolor=plot_bg_color,
        paper_bgcolor=bg_color,
        font_color=text_color
    )
    return fig

bar_fig = plot_bar()
line_fig = plot_line()

st.plotly_chart(bar_fig, use_container_width=True)
st.plotly_chart(line_fig, use_container_width=True)

# Pie chart for first site
pie_fig = px.pie(
    df,
    names=metal_filter,
    values=df.iloc[0][metal_filter],
    title=f"Metal Composition - {df.iloc[0]['Site']}",
    color_discrete_sequence=px.colors.qualitative.Pastel
)
pie_fig.update_layout(plot_bgcolor=plot_bg_color, paper_bgcolor=bg_color, font_color=text_color)
st.plotly_chart(pie_fig, use_container_width=True)

# Heatmap
st.subheader("Correlation Heatmap")
fig_hm, ax = plt.subplots(figsize=(6, 4))
sns.heatmap(df[metal_filter].corr(), annot=True, cmap="coolwarm", linewidths=0.5, ax=ax)
st.pyplot(fig_hm)
plt.close(fig_hm)

# Map
st.subheader("🌎 Site Locations & Risk Map")
fig_map = px.scatter_mapbox(
    df, lat="Latitude", lon="Longitude",
    size="HPI", color="Risk Level",
    hover_name="Site",
    hover_data=metal_filter+["HPI","HEI"],
    color_discrete_map={"Low":"green","Medium":"orange","High":"red"},
    zoom=12, height=500, size_max=20
)
fig_map.update_layout(mapbox_style="carto-darkmatter" if dark_mode else "carto-positron")
st.plotly_chart(fig_map, use_container_width=True)

# ---------------- PDF Export ----------------
def generate_pdf(df_samples, bar_fig_obj, line_fig_obj, pie_fig_obj):
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.add_page()
    pdf.set_font("Arial", "B", 18)
    pdf.cell(200, 10, "🌿 HMPI Environmental Report", ln=True, align="C")
    pdf.set_font("Arial", size=12)
    pdf.ln(5)
    pdf.cell(200, 10, "Key Metrics:", ln=True)

    for metal in metal_filter:
        avg_val = round(df_samples[metal].mean(), 4)
        max_val = round(df_samples[metal].max(), 4)
        pdf.cell(200, 8, f"{metal} Avg: {avg_val} | Max: {max_val}", ln=True)

    temp_paths = []
    try:
        for fig in (bar_fig_obj, line_fig_obj, pie_fig_obj):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                fig.write_image(tmp.name, format="png", scale=2)  # requires plotly[kaleido]
                temp_paths.append(tmp.name)

        pdf.ln(4)
        for path in temp_paths:
            pdf.image(path, x=10, w=180)
            pdf.ln(2)

        pdf_bytes = pdf.output(dest="S").encode("latin1")
        return io.BytesIO(pdf_bytes)
    finally:
        for path in temp_paths:
            try:
                os.remove(path)
            except Exception:
                pass

if st.button("📄 Export PDF Report"):
    try:
        pdf_buffer = generate_pdf(df, bar_fig, line_fig, pie_fig)
        st.download_button("Download PDF", data=pdf_buffer, file_name="hmpi_dashboard_report.pdf", mime="application/pdf")
    except Exception as e:
        st.error(f"Failed to generate PDF report: {e}\nHint: install 'plotly[kaleido]' and 'fpdf2'.")
        st.stop()