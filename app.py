import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import io

st.set_page_config(page_title="📉 Churn Prediction", layout="wide")
st.title("📉 Customer Churn Prediction")
st.markdown("Predict which customers are likely to leave using ML")

@st.cache_data
def generate_sample_data(n=1000):
    np.random.seed(42)
    df = pd.DataFrame({
        "tenure": np.random.randint(1, 72, n),
        "monthly_charges": np.round(np.random.uniform(20, 120, n), 2),
        "total_charges": np.round(np.random.uniform(100, 8000, n), 2),
        "num_support_calls": np.random.randint(0, 10, n),
        "contract_type": np.random.choice(["Month-to-month", "One year", "Two year"], n),
        "payment_method": np.random.choice(["Electronic check", "Mailed check", "Bank transfer", "Credit card"], n),
        "internet_service": np.random.choice(["DSL", "Fiber optic", "No"], n),
    })
    churn_prob = (0.5 - df["tenure"]*0.005 + df["num_support_calls"]*0.05 +
                  (df["contract_type"] == "Month-to-month").astype(float)*0.3)
    df["churn"] = (churn_prob + np.random.normal(0, 0.1, n) > 0.3).astype(int)
    return df

df = generate_sample_data()

tab1, tab2, tab3 = st.tabs(["📊 EDA", "🤖 Train Model", "🔮 Predict"])
with tab1:
    st.subheader("Dataset Overview")
    st.dataframe(df.head(20))
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Customers", len(df))
    col2.metric("Churned", df["churn"].sum())
    col3.metric("Churn Rate", f"{df['churn'].mean():.1%}")
    
    fig, ax = plt.subplots()
    df["churn"].value_counts().plot(kind="bar", ax=ax, color=["green","red"])
    ax.set_title("Churn Distribution"); ax.set_xlabel("Churn"); ax.set_ylabel("Count")
    st.pyplot(fig)

with tab2:
    model_name = st.selectbox("Select Model:", ["Random Forest", "Gradient Boosting", "Logistic Regression"])
    if st.button("🚀 Train Model"):
        le = LabelEncoder()
        df_enc = df.copy()
        for col in ["contract_type", "payment_method", "internet_service"]:
            df_enc[col] = le.fit_transform(df[col])
        X = df_enc.drop("churn", axis=1); y = df_enc["churn"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        models = {"Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
                  "Gradient Boosting": GradientBoostingClassifier(random_state=42),
                  "Logistic Regression": LogisticRegression(max_iter=1000)}
        model = models[model_name]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        auc = roc_auc_score(y_test, model.predict_proba(X_test)[:,1])
        
        st.success(f"✅ {model_name} trained! AUC-ROC: {auc:.3f}")
        st.text(classification_report(y_test, y_pred, target_names=["No Churn","Churn"]))
        st.session_state["model"] = model
        st.session_state["le"] = le

with tab3:
    st.subheader("Predict for a Single Customer")
    c1, c2 = st.columns(2)
    tenure = c1.slider("Tenure (months)", 1, 72, 12)
    monthly = c2.slider("Monthly Charges ($)", 20, 120, 65)
    calls = c1.slider("Support Calls", 0, 10, 2)
    contract = c2.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    
    if st.button("🔮 Predict Churn") and "model" in st.session_state:
        contract_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}
        inp = np.array([[tenure, monthly, monthly*tenure, calls, contract_map[contract], 1, 1]])
        prob = st.session_state["model"].predict_proba(inp)[0][1]
        if prob > 0.6: st.error(f"⚠️ High churn risk: {prob:.1%}")
        elif prob > 0.3: st.warning(f"⚡ Medium churn risk: {prob:.1%}")
        else: st.success(f"✅ Low churn risk: {prob:.1%}")
