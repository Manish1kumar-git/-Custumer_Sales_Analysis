
import streamlit as st
import pandas as pd

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Sales Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- CUSTOM DESIGN ----------------
st.markdown("""
<style>
    .stApp {
        background-color: #F4F7FB;
    }

    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2rem;
    }

    .hero {
        background: linear-gradient(120deg, #10233F, #2457A7);
        padding: 28px;
        border-radius: 18px;
        color: white;
        margin-bottom: 24px;
    }

    .hero h1 {
        color: white;
        font-size: 32px;
        margin-bottom: 8px;
    }

    .hero p {
        color: #DCE8FA;
        font-size: 15px;
        margin-bottom: 0;
    }

    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 14px;
        border: 1px solid #E3EAF4;
        box-shadow: 0 3px 12px rgba(25, 45, 80, 0.05);
        min-height: 125px;
    }

    .metric-label {
        color: #64748B;
        font-size: 14px;
        margin-bottom: 10px;
    }

    .metric-value {
        color: #142B4A;
        font-size: 27px;
        font-weight: 700;
    }

    .section-title {
        color: #142B4A;
        font-size: 22px;
        font-weight: 700;
        margin-top: 18px;
        margin-bottom: 12px;
    }

    .insight-card {
        background: white;
        border-left: 5px solid #3978D4;
        padding: 17px;
        border-radius: 10px;
        margin-bottom: 12px;
        color: #263B53;
    }

    [data-testid="stSidebar"] {
        background-color: #EAF0F9;
    }
</style>
""", unsafe_allow_html=True)


# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    return pd.read_csv("data/cleaned_dataset.csv")


try:
    df = load_data()
except FileNotFoundError:
    st.error(
        "Dataset not found. Please run the data-cleaning script first "
        "and check that data/cleaned_dataset.csv exists."
    )
    st.stop()
except Exception as e:
    st.error(f"Unable to load dataset: {e}")
    st.stop()


# ---------------- VALIDATE COLUMNS ----------------
required = ["Sales", "Region", "Category", "Churn", "AI_Anomaly"]
missing = [col for col in required if col not in df.columns]

if missing:
    st.error(f"Required columns missing from dataset: {', '.join(missing)}")
    st.write("Available columns:", df.columns.tolist())
    st.stop()

df["Sales"] = pd.to_numeric(df["Sales"], errors="coerce")
df["Churn"] = pd.to_numeric(df["Churn"], errors="coerce")
df["AI_Anomaly"] = pd.to_numeric(df["AI_Anomaly"], errors="coerce")

df = df.dropna(subset=["Sales"])


# ---------------- SIDEBAR FILTERS ----------------
st.sidebar.title("🎛️ Dashboard Controls")
st.sidebar.caption("Filter your sales data")

regions = sorted(df["Region"].dropna().astype(str).unique())
categories = sorted(df["Category"].dropna().astype(str).unique())

selected_regions = st.sidebar.multiselect(
    "Select Region",
    regions,
    default=regions
)

selected_categories = st.sidebar.multiselect(
    "Select Category",
    categories,
    default=categories
)

filtered_df = df[
    df["Region"].astype(str).isin(selected_regions)
    & df["Category"].astype(str).isin(selected_categories)
].copy()

st.sidebar.divider()
st.sidebar.write(f"Records displayed: **{len(filtered_df):,}**")

if st.sidebar.button("Reset Filters", use_container_width=True):
    st.cache_data.clear()
    st.rerun()


# ---------------- HEADER ----------------
st.markdown("""
<div class="hero">
    <h1>📊 AI Sales Intelligence</h1>
    <p>Customer analytics, sales performance and churn monitoring</p>
</div>
""", unsafe_allow_html=True)

if filtered_df.empty:
    st.warning("No records match these filters. Please select another region or category.")
    st.stop()


# ---------------- KPI METRICS ----------------
total_sales = filtered_df["Sales"].sum()
total_customers = len(filtered_df)

churn_rate = (
    filtered_df["Churn"].mean() * 100
    if filtered_df["Churn"].notna().any()
    else 0
)

anomalies = int(filtered_df["AI_Anomaly"].fillna(0).sum())

m1, m2, m3, m4 = st.columns(4)

metrics = [
    (m1, "💰", "Total Sales", f"${total_sales:,.0f}"),
    (m2, "👥", "Customer Records", f"{total_customers:,}"),
    (m3, "🔄", "Churn Rate", f"{churn_rate:.1f}%"),
    (m4, "🤖", "AI Anomalies", f"{anomalies:,}")
]

for col, icon, label, value in metrics:
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{icon} {label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """, unsafe_allow_html=True)


# ---------------- SALES ANALYSIS ----------------
st.markdown(
    '<div class="section-title">📈 Sales Performance</div>',
    unsafe_allow_html=True
)

left, right = st.columns(2)

with left:
    st.subheader("Sales by Region")
    region_sales = (
        filtered_df.groupby("Region")["Sales"]
        .sum()
        .sort_values(ascending=False)
    )

    st.bar_chart(region_sales, color="#3978D4")
    st.caption("Compare total sales across regions.")

with right:
    st.subheader("Sales by Category")
    category_sales = (
        filtered_df.groupby("Category")["Sales"]
        .sum()
        .sort_values(ascending=False)
    )

    st.bar_chart(category_sales, color="#18A999")
    st.caption("Identify the categories generating the most sales.")


# ---------------- MONTHLY SALES TREND ----------------
# Automatically find a likely date column if available.
date_candidates = [
    "Date", "Order Date", "Order_Date",
    "Purchase Date", "Transaction Date"
]

date_col = next(
    (col for col in date_candidates if col in filtered_df.columns),
    None
)

if date_col:
    dates = pd.to_datetime(filtered_df[date_col], errors="coerce")

    trend_df = filtered_df.copy()
    trend_df["Month"] = dates.dt.to_period("M").astype(str)
    trend_df = trend_df[trend_df["Month"] != "NaT"]

    if not trend_df.empty:
        monthly_sales = (
            trend_df.groupby("Month")["Sales"]
            .sum()
            .sort_index()
        )

        st.markdown(
            '<div class="section-title">📅 Monthly Sales Trend</div>',
            unsafe_allow_html=True
        )
        st.line_chart(monthly_sales, color="#805AD5")
else:
    st.info(
        "A monthly trend chart can be shown when the dataset contains "
        "a recognized date column."
    )


# ---------------- AI BUSINESS INSIGHTS ----------------
st.markdown(
    '<div class="section-title">✨ AI Business Insights</div>',
    unsafe_allow_html=True
)

best_region = region_sales.idxmax()
best_category = category_sales.idxmax()

i1, i2 = st.columns(2)

with i1:
    st.markdown(f"""
    <div class="insight-card">
        <b>🏆 Highest-Sales Region</b><br><br>
        {best_region}<br>
        <small>Top region within the selected filters.</small>
    </div>
    """, unsafe_allow_html=True)

with i2:
    st.markdown(f"""
    <div class="insight-card">
        <b>🛍️ Best-Selling Category</b><br><br>
        {best_category}<br>
        <small>Top category within the selected filters.</small>
    </div>
    """, unsafe_allow_html=True)

st.markdown(f"""
<div class="insight-card">
    <b>🤖 Anomaly Monitoring</b><br><br>
    {anomalies:,} records are flagged by the existing AI anomaly field.
    Review these records to investigate unusual patterns.
</div>
""", unsafe_allow_html=True)


# ---------------- DATA TABLE & DOWNLOAD ----------------
with st.expander("🔎 Explore Sales Data"):
    st.dataframe(filtered_df, use_container_width=True, hide_index=True)

st.download_button(
    label="⬇️ Download Filtered Data (CSV)",
    data=filtered_df.to_csv(index=False).encode("utf-8"),
    file_name="filtered_sales_data.csv",
    mime="text/csv"
)

st.divider()
st.caption("AI Sales Intelligence | Customer Sales & Churn Analytics")

