import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import plotly.express as px
from openai import OpenAI
import os

st.set_page_config(page_title="CareBank AI", layout="wide")
st.title("💳 CareBank – Agentic AI Financial Intelligence Platform")

# =========================
# LOAD OPENAI KEY (Cloud Safe)
# =========================
api_key = os.getenv("OPENAI_API_KEY")

if api_key:
    client = OpenAI(api_key=api_key)
else:
    client = None

# =========================
# SIDEBAR CONTROLS
# =========================
st.sidebar.title("⚙ Budget Settings")

food_budget = st.sidebar.number_input("Food Budget", value=4000)
transport_budget = st.sidebar.number_input("Transport Budget", value=2000)
shopping_budget = st.sidebar.number_input("Shopping Budget", value=3000)
other_budget = st.sidebar.number_input("Other Budget", value=2000)

# =========================
# SESSION MEMORY
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = []

# =========================
# AGENTS
# =========================

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
        score = int(((income - expense) / income) * 100) if income > 0 else 0
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
            return "🟡 Moderate financial health. Optimize discretionary spending."
        else:
            return "🔴 Financial risk detected. Immediate expense correction advised."

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

# =========================
# FILE UPLOAD
# =========================

uploaded_file = st.file_uploader("📂 Upload Transaction CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce")
    df = df.dropna(subset=["Amount"])

    orchestrator = Orchestrator()
    df, anomalies, score, advice, income, expense, forecast = orchestrator.execute(df)

    st.markdown("---")

    # Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Income", f"₹{income:,.2f}")
    col2.metric("Expense", f"₹{expense:,.2f}")
    col3.metric("Health Score", f"{score}/100")

    # Pie Chart
    st.subheader("📊 Spending Distribution")
    category_sum = df.groupby("Category")["Amount"].sum().abs().reset_index()
    fig = px.pie(category_sum, values="Amount", names="Category")
    st.plotly_chart(fig)

    # Forecast
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
        st.warning("Not enough data for forecasting.")

    # Budget Alerts
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

    # Anomalies
    st.subheader("🚨 Anomalies")
    if anomalies.empty:
        st.success("No major anomalies detected.")
    else:
        st.dataframe(anomalies)

    # Advisor
    st.subheader("🤖 AI Advisor")
    st.success(advice)

    # =========================
    # CHAT SYSTEM
    # =========================
    st.markdown("---")
    st.subheader("💬 Conversational Financial AI")

    user_input = st.chat_input("Ask about your financial health...")

    if user_input:
        reply = None

        spending_dict = spending.to_dict()
        context_summary = f"""
        Income: {income}
        Expense: {expense}
        Score: {score}
        Category Spending: {spending_dict}
        """

        # Try OpenAI
        if client:
            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a financial advisor."},
                        {"role": "system", "content": context_summary},
                        {"role": "user", "content": user_input}
                    ],
                    temperature=0.7
                )

                reply = response.choices[0].message.content

            except Exception as e:
                st.error(f"OpenAI API Error: {e}")
                reply = None

        # Intelligent Fallback
        if reply is None:
            text = user_input.lower()
            expense_df = df[df["Amount"] < 0]
            category_expense = expense_df.groupby("Category")["Amount"].sum().abs()

            if not category_expense.empty:
                top_category = category_expense.idxmax()
                top_value = category_expense.max()
            else:
                top_category = None

            if "unnecessary" in text or "reduce" in text:
                reply = f"Your highest spending category is {top_category} (₹{top_value:,.0f}). Consider optimizing this."
            elif "score" in text:
                reply = f"Your financial health score is {score}/100 based on income-to-expense ratio."
            else:
                reply = advice

        st.chat_message("user").write(user_input)
        st.chat_message("assistant").write(reply)

    # Download
    report = df.to_csv(index=False)
    st.download_button("📥 Download Report", report, "report.csv", key="download_btn")

else:
    st.info("Upload a CSV file to activate the AI system.")
