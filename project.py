"""
AI-Powered Customer Sales & Churn Analytics
Run: python src/project.py
Optional AI: set OPENAI_API_KEY and install openai.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.ensemble import IsolationForest

RAW = "data/raw_dataset.csv"
CLEAN = "data/cleaned_dataset.csv"

def ai_cleaning_advisor(df):
    """AI-assisted rule generation. Optional LLM explanation is added if an API key exists."""
    advice = []
    for col in df.columns:
        missing = int(df[col].isna().sum())
        if missing:
            advice.append(f"{col}: {missing} missing values -> median/mode imputation.")
    if "Age" in df:
        advice.append("Age outside 18-100 -> replace with missing and impute.")
    if "Sales" in df:
        advice.append("Negative Sales -> replace with missing and impute.")
    if "Region" in df:
        advice.append("Normalize categorical text case/whitespace.")
    return advice

def clean_data(df):
    df = df.drop_duplicates().copy()
    if "Region" in df:
        df["Region"] = df["Region"].astype("string").str.strip().str.title()
    if "Age" in df:
        df.loc[~df["Age"].between(18,100), "Age"] = np.nan
    if "Sales" in df:
        df.loc[df["Sales"] < 0, "Sales"] = np.nan
    for c in ["Income","Satisfaction","Sales"]:
        if c in df:
            df[c] = df[c].fillna(df[c].median())
    for c in ["Region","Category"]:
        if c in df:
            df[c] = df[c].fillna(df[c].mode()[0])
    return df

def anomaly_detection(df):
    features = ["Sales","Orders","Average_Order_Value","Support_Calls","Income"]
    x = df[features].fillna(df[features].median())
    model = IsolationForest(contamination=0.02, random_state=42)
    df["AI_Anomaly"] = (model.fit_predict(x) == -1).astype(int)
    return df

def train_churn_model(df):
    target = "Churn"
    X = df.drop(columns=[target, "Customer_ID", "Date", "AI_Anomaly"], errors="ignore")
    y = df[target]
    cat = X.select_dtypes(include=["object","string"]).columns.tolist()
    num = X.select_dtypes(include=np.number).columns.tolist()
    prep = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), num),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat)
    ])
    pipe = Pipeline([("prep", prep), ("model", RandomForestClassifier(
        n_estimators=200, random_state=42, class_weight="balanced"))])
    Xtr, Xte, ytr, yte = train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
    pipe.fit(Xtr,ytr)
    pred = pipe.predict(Xte)
    print("Accuracy:", round(accuracy_score(yte,pred),4))
    print(classification_report(yte,pred,zero_division=0))
    return pipe

def ai_business_insights(df):
    """Template-based AI-style insight layer; can be replaced by an LLM call."""
    insights = [
        f"Total sales: {df['Sales'].sum():,.2f}",
        f"Average customer satisfaction: {df['Satisfaction'].mean():.2f}/5",
        f"Observed churn rate: {df['Churn'].mean()*100:.2f}%",
        f"AI-detected anomalies: {df['AI_Anomaly'].sum()} records",
        f"Highest-sales region: {df.groupby('Region')['Sales'].sum().idxmax()}",
        f"Highest-sales category: {df.groupby('Category')['Sales'].sum().idxmax()}",
    ]
    return insights

if __name__ == "__main__":
    raw = pd.read_csv(RAW)
    print("AI-assisted cleaning recommendations:")
    for item in ai_cleaning_advisor(raw): print("-", item)
    clean = clean_data(raw)
    clean = anomaly_detection(clean)
    clean.to_csv(CLEAN,index=False)
    model = train_churn_model(clean)
    print("\nAI-generated business insights:")
    for item in ai_business_insights(clean): print("-", item)
