import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import plotly.express as px
from openai import OpenAI
import os

st.set_page_config(page_title="CareBank AI", layout="wide")

st.title("💳 CareBank – Agentic AI Financial Intelligence Platform")

# ===============================
# SIDEBAR SETTINGS
# ===============================

st.sidebar.title("⚙ Control Panel")

api_key = st.sidebar.text_input("Enter OpenAI API Key (Optional)", type="password")

food_budget = st.sidebar.number_input("Food Budget", value=4000)
transport_budget = st.sidebar.number_input("Transport Budget", value=2000)
shopping_budget = st.sidebar.number_input("Shopping Budget", value=3000)
other_budget = st.sidebar.number_input("Other Budget", value=2000)

if api_key:
    client = OpenAI(api_key=api_key)
else:
    client = None

# ===============================
# SESSION MEMORY
# ===============================

if "messages" not in st.session_state:
    st.session_state.messages = []

# ===============================
# AGENTS
# ===============================

class SpendingAgent:
    def run(self, df):
        df["Category"] = df["Description"].apply(lambda x:
            "Food" if "swiggy" in str(x).lower() or "zomato" in str(x).lower()
            else "Transport" if "uber" in str(x).lower()
            else "Shopping" if "amazon" in str(x).lower()
            else "Other"
        )
        return df

class RiskAgent:
    def run(self, df):
        if len(df) < 5:
            df["Anomaly"] = 1
            return df[df["Anomaly"] == -1]

        clf = IsolationForest(contamination=0.1, random_state=42)
        df["Anomaly"] = clf.fit_predict(df[["Amount"]])
        return df[df["Anomaly"] == -1]

class BudgetAgent:
    def run(self, df):
        income = df[df["Amount"] > 0]["Amount"].sum()
        expense = abs(df[df["Amount"] < 0]["Amount"].sum())

        if income == 0:
            score = 0
        else:
            score = int(((income - expense) / income) * 100)

        return score, income, expense

class ForecastAgent:
    def run(self, df):
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"])

        monthly = df.groupby(
            pd.Grouper(key="Date", freq="M")
        )["Amount"].sum().reset_index()

        if len(monthly) < 2:
            return None

        monthly["Forecast"] = monthly["Amount"].rolling(2).mean()
        return monthly

class AdvisorAgent:
    def run(self, score):
        if score > 75:
            return "🟢 Strong financial stability. Consider diversified investments."
        elif score > 50:
            return "🟡 Moderate health. Reduce discretionary spending."
        else:
            return "🔴 Financial risk detected. Immediate correction required."

class Orchestrator:
    def __init__(self):
        self.spending = SpendingAgent()
        self.risk = RiskAgent()
        self.budget = BudgetAgent()
        self.forecast = ForecastAgent()
        self.advisor = AdvisorAgent()

    def execute(self, df):
        st.subheader("🧠 Agent Execution Logs")

        df = self.spending.run(df)
        st.write("✔ Spending Agent completed")

        anomalies = self.risk.run(df)
        st.write("✔ Risk Agent completed")

        score, income, expense = self.budget.run(df)
        st.write("✔ Budget Agent completed")

        forecast = self.forecast.run(df)
        st.write("✔ Forecast Agent completed")

        advice = self.advisor.run(score)
        st.write("✔ Advisor Agent completed")

        return df, anomalies, score, advice, income, expense, forecast

# ===============================
# FILE UPLOAD
# ===============================

uploaded_file = st.file_uploader("📂 Upload Transaction CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce")
    df = df.dropna(subset=["Amount"])

    orchestrator = Orchestrator()
    df, anomalies, score, advice, income, expense, forecast = orchestrator.execute(df)

    st.markdown("---")

    # ===============================
    # METRICS
    # ===============================

    col1, col2, col3 = st.columns(3)
    col1.metric("Income", f"₹{income:,.2f}")
    col2.metric("Expense", f"₹{expense:,.2f}")
    col3.metric("Health Score", f"{score}/100")

    # ===============================
    # PIE CHART
    # ===============================

    st.subheader("📊 Spending Distribution")
    category_sum = df.groupby("Category")["Amount"].sum().abs().reset_index()
    fig = px.pie(category_sum, values="Amount", names="Category")
    st.plotly_chart(fig)

    # ===============================
    # FORECAST
    # ===============================

    st.subheader("📈 Cashflow Forecast")

    if forecast is not None:
        fig2 = px.line(
            forecast,
            x="Date",
            y=["Amount", "Forecast"],
            markers=True
        )
        st.plotly_chart(fig2)
    else:
        st.warning("Not enough data available for forecasting.")

    # ===============================
    # BUDGET ALERTS
    # ===============================

    st.subheader("⚠ Budget Monitoring")

    spending = df.groupby("Category")["Amount"].sum().abs()
    budgets = {
        "Food": food_budget,
        "Transport": transport_budget,
        "Shopping": shopping_budget,
        "Other": other_budget
    }

    for cat in budgets:
        if cat in spending:
            if spending[cat] > budgets[cat]:
                st.error(f"{cat} budget exceeded!")
            elif spending[cat] > 0.8 * budgets[cat]:
                st.warning(f"{cat} nearing budget limit.")

    # ===============================
    # ANOMALIES
    # ===============================

    st.subheader("🚨 Anomalies")
    if anomalies.empty:
        st.success("No major anomalies detected.")
    else:
        st.dataframe(anomalies)

    # ===============================
    # ADVISOR
    # ===============================

    st.subheader("🤖 AI Advisor")
    st.success(advice)

    # ===============================
    # CHATBOT WITH FALLBACK
    # ===============================

    st.markdown("---")
    st.subheader("💬 Conversational Financial AI")

    user_input = st.chat_input("Ask about your financial health...", key="chat_input")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})

        spending_dict = spending.to_dict()
        context_summary = f"""
        Income: {income}
        Expense: {expense}
        Health Score: {score}
        Spending Breakdown: {spending_dict}
        """

        reply = None

        if client:
            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a professional financial advisor."},
                        {"role": "system", "content": context_summary},
                        *st.session_state.messages
                    ]
                )
                reply = response.choices[0].message.content
            except Exception:
                reply = None

        if reply is None:
            if score > 75:
                reply = "Strong financial position. Consider increasing investments."
            elif score > 50:
                reply = "Financial health is moderate. Try reducing unnecessary expenses."
            else:
                reply = "Your spending risk is high. Immediate optimization needed."

        st.session_state.messages.append({"role": "assistant", "content": reply})

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # ===============================
    # DOWNLOAD REPORT
    # ===============================

    report = df.to_csv(index=False)

    st.download_button(
        "📥 Download Financial Report",
        report,
        "report.csv",
        key="download_report_btn"
    )

else:
    st.info("Upload a CSV file to activate the AI system.")
