
import streamlit as st
import pandas as pd
import plotly.express as px
import sqlite3
from pathlib import Path

# PAGE CONFIGURATION
st.set_page_config(
    page_title="E-Commerce Intelligence System",
    page_icon="🛒",
    layout="wide"
)

# DATABASE LOCATION
PROJECT_DIR = Path(__file__).resolve().parent

DATABASE_LOCATIONS = [
    PROJECT_DIR / "ecommerce_intelligence.db",
    PROJECT_DIR / "notebooks" / "ecommerce_intelligence.db"
]

DB_PATH = next(
    (path for path in DATABASE_LOCATIONS if path.exists()),
    None
)

if DB_PATH is None:
    st.error("Database file not found!")
    st.write("Checked locations:")
    for path in DATABASE_LOCATIONS:
        st.write(str(path))
    st.stop()

# LOAD DATABASE
@st.cache_data
def load_data(database_path):
    connection = sqlite3.connect(database_path)

    try:
        tables = pd.read_sql_query(
            "SELECT name FROM sqlite_master WHERE type='table'",
            connection
        )

        if "sales" not in tables["name"].values:
            return pd.DataFrame()

        data = pd.read_sql_query(
            "SELECT * FROM sales",
            connection
        )
    finally:
        connection.close()

    return data


df = load_data(str(DB_PATH))

if df.empty:
    st.error("Sales table is empty or not available.")
    st.stop()

# DATA PREPROCESSING
required_columns = [
    "InvoiceNo", "StockCode", "Description",
    "Quantity", "InvoiceDate", "CustomerID",
    "Country", "Revenue"
]

missing_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    st.error(f"Missing columns: {missing_columns}")
    st.stop()

df["InvoiceDate"] = pd.to_datetime(
    df["InvoiceDate"], errors="coerce"
)
df["Revenue"] = pd.to_numeric(
    df["Revenue"], errors="coerce"
)
df["Quantity"] = pd.to_numeric(
    df["Quantity"], errors="coerce"
)

df = df.dropna(
    subset=["InvoiceDate", "Revenue", "Country"]
)

# DASHBOARD HEADER
st.title("🛒 E-Commerce Intelligence System")
st.markdown(
    "### Sales Analytics | Customer Insights | Business Intelligence"
)
st.caption(
    "Interactive dashboard built using Python, Pandas, SQL and Plotly"
)
st.divider()

# SIDEBAR FILTERS
st.sidebar.title("Dashboard Filters")

min_date = df["InvoiceDate"].min().date()
max_date = df["InvoiceDate"].max().date()

date_range = st.sidebar.date_input(
    "Select Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

countries = sorted(df["Country"].unique())

selected_countries = st.sidebar.multiselect(
    "Select Countries",
    options=countries,
    default=countries
)

if not selected_countries:
    st.warning("Please select at least one country.")
    st.stop()

if len(date_range) == 2:
    start_date, end_date = date_range

    filtered_df = df[
        (df["InvoiceDate"].dt.date >= start_date) &
        (df["InvoiceDate"].dt.date <= end_date) &
        (df["Country"].isin(selected_countries))
    ].copy()
else:
    filtered_df = df[
        df["Country"].isin(selected_countries)
    ].copy()

if filtered_df.empty:
    st.warning("No data available for selected filters.")
    st.stop()

# KEY PERFORMANCE INDICATORS
st.subheader("📊 Business Overview")

total_revenue = filtered_df["Revenue"].sum()
total_orders = filtered_df["InvoiceNo"].nunique()
total_customers = filtered_df["CustomerID"].nunique()
total_products = filtered_df["StockCode"].nunique()

average_order_value = (
    total_revenue / total_orders if total_orders > 0 else 0
)

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Revenue", f"£{total_revenue:,.2f}")
col2.metric("Total Orders", f"{total_orders:,}")
col3.metric("Total Customers", f"{total_customers:,}")
col4.metric("Total Products", f"{total_products:,}")

st.divider()

# MONTHLY REVENUE ANALYSIS
st.subheader("📈 Monthly Revenue Analysis")

monthly_revenue = (
    filtered_df
    .set_index("InvoiceDate")
    .resample("MS")["Revenue"]
    .sum()
    .reset_index()
)

monthly_revenue.columns = ["Month", "Revenue"]

fig_monthly = px.line(
    monthly_revenue,
    x="Month",
    y="Revenue",
    markers=True,
    title="Monthly Revenue Trend",
    template="plotly_white"
)

st.plotly_chart(fig_monthly, width="stretch")

st.divider()

# COUNTRY ANALYSIS
st.subheader("🌍 Country-wise Revenue Analysis")

country_revenue = (
    filtered_df.groupby("Country")["Revenue"]
    .sum()
    .nlargest(10)
    .reset_index()
    .sort_values("Revenue")
)

fig_country = px.bar(
    country_revenue,
    x="Revenue",
    y="Country",
    orientation="h",
    title="Top 10 Countries by Revenue",
    template="plotly_white"
)

st.plotly_chart(fig_country, width="stretch")

st.divider()

# PRODUCT ANALYSIS
st.subheader("📦 Product Performance")

product_revenue = (
    filtered_df.groupby("Description")["Revenue"]
    .sum()
    .nlargest(10)
    .reset_index()
    .sort_values("Revenue")
)

fig_product = px.bar(
    product_revenue,
    x="Revenue",
    y="Description",
    orientation="h",
    title="Top 10 Products by Revenue",
    template="plotly_white"
)

st.plotly_chart(fig_product, width="stretch")

st.divider()

# CUSTOMER ANALYSIS
st.subheader("👥 Customer Analysis")

customer_df = filtered_df.dropna(
    subset=["CustomerID"]
).copy()

if not customer_df.empty:
    customer_revenue = (
        customer_df.groupby("CustomerID")["Revenue"]
        .sum()
        .nlargest(10)
        .reset_index()
    )

    customer_revenue["CustomerID"] = (
        customer_revenue["CustomerID"].astype(str)
    )

    fig_customer = px.bar(
        customer_revenue,
        x="CustomerID",
        y="Revenue",
        title="Top 10 Customers by Revenue",
        template="plotly_white"
    )

    st.plotly_chart(fig_customer, width="stretch")

st.divider()

# RFM CUSTOMER SEGMENTATION
st.subheader("🎯 RFM Customer Segmentation")

if not customer_df.empty:
    snapshot_date = (
        customer_df["InvoiceDate"].max()
        + pd.Timedelta(days=1)
    )

    rfm = customer_df.groupby("CustomerID").agg(
        Recency=(
            "InvoiceDate",
            lambda x: (snapshot_date - x.max()).days
        ),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("Revenue", "sum")
    ).reset_index()

    def segment_customer(row):
        if row["Frequency"] >= 10 and row["Monetary"] >= 1000:
            return "Champions"
        elif row["Frequency"] >= 5:
            return "Loyal Customers"
        elif row["Recency"] <= 30:
            return "Recent Customers"
        elif row["Recency"] > 90:
            return "At Risk"
        else:
            return "Others"

    rfm["Segment"] = rfm.apply(
        segment_customer, axis=1
    )

    segment_counts = (
        rfm["Segment"].value_counts().reset_index()
    )
    segment_counts.columns = ["Segment", "Customers"]

    fig_rfm = px.pie(
        segment_counts,
        names="Segment",
        values="Customers",
        title="Customer Segmentation Distribution",
        hole=0.4
    )

    st.plotly_chart(fig_rfm, width="stretch")

    st.markdown("#### Customer RFM Details")
    st.dataframe(
        rfm.sort_values("Monetary", ascending=False),
        width="stretch",
        hide_index=True
    )

st.divider()

# BUSINESS INSIGHTS
st.subheader("💡 Business Insights")

best_country = (
    filtered_df.groupby("Country")["Revenue"]
    .sum()
    .idxmax()
)

best_product = (
    filtered_df.groupby("Description")["Revenue"]
    .sum()
    .idxmax()
)

col1, col2 = st.columns(2)

with col1:
    st.info(f"Highest Revenue Country: {best_country}")
    st.info(f"Highest Revenue Product: {best_product}")

with col2:
    st.info(f"Average Order Value: £{average_order_value:,.2f}")
    st.info(f"Total Orders: {total_orders:,}")

st.divider()

# DOWNLOAD DATA
st.subheader("📥 Download Sales Data")

csv_data = filtered_df.to_csv(
    index=False
).encode("utf-8")

st.download_button(
    label="Download Filtered Sales Data",
    data=csv_data,
    file_name="ecommerce_sales_data.csv",
    mime="text/csv"
)

# FOOTER
st.divider()

st.caption(
    "E-Commerce Intelligence System | Python | Pandas | SQL | Plotly | Streamlit"
)
