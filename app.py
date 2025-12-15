import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score

st.set_page_config(page_title="Auto Data Cleaning & ML", layout="wide")

st.title("🤖 Auto Data Cleaning & ML Web App")
st.write("Upload a CSV → Clean → Encode → Train → Download Excel")

# =========================
# Upload CSV
# =========================
uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Raw Data Preview")
    st.dataframe(df.head())

    # =========================
    # Cleaning
    # =========================
    df.columns = (
        df.columns.str.strip().str.lower().str.replace(" ", "_")
    )
    df = df.drop_duplicates()

    for col in df.columns:
        if df[col].dtype in ["int64", "float64"]:
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna("Unknown")

    st.subheader("🧼 Cleaned Data")
    st.dataframe(df.head())

    # =========================
    # Target selection
    # =========================
    target = st.selectbox("🎯 Select target column", df.columns)

    X = df.drop(columns=[target])
    y = df[target]

    # =========================
    # Preprocessing
    # =========================
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = X.select_dtypes(include=["object"]).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
        ]
    )

    X_processed = preprocessor.fit_transform(X)

    # =========================
    # Train Model
    # =========================
    if st.button("🚀 Train Model"):
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y, test_size=0.2, random_state=42
        )

        if y.dtype == "object":
            model = RandomForestClassifier(n_estimators=200, random_state=42)
            task = "Classification"
        else:
            model = RandomForestRegressor(n_estimators=200, random_state=42)
            task = "Regression"

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        st.success(f"✅ {task} model trained successfully")

        # =========================
        # Metrics
        # =========================
        if task == "Classification":
            acc = accuracy_score(y_test, y_pred)
            st.metric("Accuracy", round(acc, 4))
        else:
            rmse = mean_squared_error(y_test, y_pred, squared=False)
            r2 = r2_score(y_test, y_pred)
            st.metric("RMSE", round(rmse, 4))
            st.metric("R² Score", round(r2, 4))

        # =========================
        # Export Excel
        # =========================
        predictions_df = pd.DataFrame({
            "Actual": y_test.values,
            "Predicted": y_pred
        })

        excel_file = "results.xlsx"
        with pd.ExcelWriter(excel_file, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="Cleaned_Data", index=False)
            predictions_df.to_excel(writer, sheet_name="Predictions", index=False)

        with open(excel_file, "rb") as f:
            st.download_button(
                "⬇️ Download Excel Results",
                data=f,
                file_name="model_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
