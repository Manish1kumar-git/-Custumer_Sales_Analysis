import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="AI Sales Analytics", layout="wide")
st.title("AI-Powered Customer Sales & Churn Analytics")

try:
    df = pd.read_csv("data/cleaned_dataset.csv")
except FileNotFoundError:
    st.warning("Run `python src/project.py` first to create cleaned_dataset.csv.")
    st.stop()

c1,c2,c3,c4 = st.columns(4)
c1.metric("Total Sales", f"${df.Sales.sum():,.0f}")
c2.metric("Customers", f"{len(df):,}")
c3.metric("Churn Rate", f"{df.Churn.mean()*100:.1f}%")
c4.metric("AI Anomalies", f"{df.AI_Anomaly.sum():,}")

st.subheader("Sales by Region")
sales_region = df.groupby("Region", as_index=False)["Sales"].sum()
st.bar_chart(sales_region.set_index("Region"))

st.subheader("Sales by Category")
sales_cat = df.groupby("Category", as_index=False)["Sales"].sum()
st.bar_chart(sales_cat.set_index("Category"))

st.subheader("AI Business Insights")
st.write(f"Highest-sales region: **{sales_region.loc[sales_region.Sales.idxmax(),'Region']}**")
st.write(f"Highest-sales category: **{sales_cat.loc[sales_cat.Sales.idxmax(),'Category']}**")
st.write(f"AI flagged **{int(df.AI_Anomaly.sum())}** records as unusual.")
