import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

st.set_page_config(page_title="🚀 Spacecraft Monitoring", layout="wide")

st.title("🚀 AI-Based Spacecraft Monitoring System")

uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])

if uploaded_file is None:
    st.warning("⚠️ Please upload your dataset")
else:
    data = pd.read_csv(uploaded_file)

    st.subheader("📊 Dataset Preview")
    st.dataframe(data.head())

    # CHECK REQUIRED COLUMNS
    if 'true_label' in data.columns and 'anomaly' in data.columns:

        y_true = data['true_label']
        y_pred = data['anomaly']

        # ✅ METRICS
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)

        st.subheader("📈 Model Performance")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{accuracy:.2f}")
        col2.metric("Precision", f"{precision:.2f}")
        col3.metric("Recall", f"{recall:.2f}")
        col4.metric("F1 Score", f"{f1:.2f}")

        # ✅ STATUS
        if data['anomaly'].iloc[-1] == 1:
            st.error("⚠️ Anomaly Detected!")
        else:
            st.success("✅ System Normal")

        # ✅ CONFUSION MATRIX
        st.subheader("📉 Confusion Matrix")

        cm = confusion_matrix(y_true, y_pred)

        fig, ax = plt.subplots()
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(ax=ax)

        st.pyplot(fig)

    else:
        st.error("❌ Missing columns: true_label or anomaly")

    # 📡 GRAPH
    st.subheader("📡 Sensor Visualization")
    column = st.selectbox("Select Column", data.columns)
    st.line_chart(data[column])
