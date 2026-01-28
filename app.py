import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Try importing XGBoost safely
try:
    from xgboost import XGBRegressor
    xgb_available = True
except:
    xgb_available = False

# ===============================
# Page Configuration
# ===============================
st.set_page_config(
    page_title="Benzene Concentration Prediction",
    layout="wide"
)

# ===============================
# App Title
# ===============================
st.title("Benzene Concentration Prediction in Isomerization Unit")
st.write(
    "End-to-end machine learning application for exploratory data analysis, "
    "feature selection, model training, evaluation, and benzene concentration prediction."
)

# ===============================
# File Upload
# ===============================
uploaded_file = st.file_uploader(
    "Upload Process and Lab Data (CSV)",
    type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # ===============================
    # Data Preview
    # ===============================
    st.subheader("Data Preview")
    st.dataframe(df.head())

    # ===============================
    # Dataset Overview
    # ===============================
    col1, col2 = st.columns(2)
    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])

    # ===============================
    # Missing Data Overview
    # ===============================
    st.subheader("Missing Data Overview")
    missing_values = df.isnull().sum()
    st.write(missing_values)

    total_missing = int(missing_values.sum())
    total_present = df.size - total_missing

    st.bar_chart(
        pd.DataFrame(
            {"Count": [total_present, total_missing]},
            index=["Available Data", "Missing Data"]
        )
    )

    # ===============================
    # Feature Selection
    # ===============================
    st.subheader("Feature Selection")

    numeric_columns = df.select_dtypes(include=np.number).columns.tolist()

    selected_features = st.multiselect(
        "Select input features",
        options=numeric_columns
    )

    target_column = st.selectbox(
        "Select target variable (Benzene concentration)",
        options=numeric_columns
    )

    if not selected_features or not target_column:
        st.warning("Please select features and target variable.")
        st.stop()

    if target_column in selected_features:
        selected_features.remove(target_column)

    # ===============================
    # Feature Comparison
    # ===============================
    st.subheader("Feature Comparison")

    if len(selected_features) >= 2:
        fx = st.selectbox("Feature X", selected_features)
        fy = st.selectbox("Feature Y", selected_features, index=1)

        comp_df = df[[fx, fy]].dropna()

        st.scatter_chart(comp_df, x=fx, y=fy)
        st.line_chart(comp_df)

        corr_val = comp_df[fx].corr(comp_df[fy])
        st.metric("Pearson Correlation", round(corr_val, 4))

        if abs(corr_val) >= 0.8:
            st.success("Strong correlation detected")
        elif abs(corr_val) >= 0.5:
            st.warning("Moderate correlation detected")
        else:
            st.info("ℹWeak correlation detected")

        st.download_button(
            "Download Comparison Data",
            comp_df.to_csv(index=False),
            file_name=f"{fx}_vs_{fy}.csv",
            mime="text/csv"
        )

    # ===============================
    # Model Training
    # ===============================
    st.subheader("Model Training")

    model_choice = st.selectbox(
        "Select ML Model",
        ["Random Forest", "XGBoost (if available)"]
    )

    X = df[selected_features]
    y = df[target_column]

    X = X.fillna(X.mean())
    y = y.fillna(y.mean())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    if st.button("Train Model"):
        if model_choice == "Random Forest":
            model = RandomForestRegressor(
                n_estimators=200,
                random_state=42
            )

        elif model_choice == "XGBoost (if available)":
            if not xgb_available:
                st.error("XGBoost is not installed in this environment.")
                st.stop()
            model = XGBRegressor(
                n_estimators=200,
                learning_rate=0.05,
                random_state=42
            )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # ===============================
        # Model Metrics
        # ===============================
        st.subheader("Model Performance Metrics")

        r2 = r2_score(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        mae = mean_absolute_error(y_test, y_pred)

        m1, m2, m3 = st.columns(3)
        m1.metric("R² Score", round(r2, 4))
        m2.metric("RMSE", round(rmse, 4))
        m3.metric("MAE", round(mae, 4))

        # ===============================
        # Feature Importance
        # ===============================
        st.subheader("Feature Importance")

        if hasattr(model, "feature_importances_"):
            importance_df = pd.DataFrame({
                "Feature": selected_features,
                "Importance": model.feature_importances_
            }).sort_values(by="Importance", ascending=False)

            st.bar_chart(
                importance_df.set_index("Feature")
            )

        # ===============================
        # Final Prediction
        # ===============================
        st.subheader("Final Benzene Concentration Prediction")

        input_data = {}
        for feature in selected_features:
            input_data[feature] = st.number_input(
                f"Enter value for {feature}",
                float(X[feature].mean())
            )

        if st.button("Predict Benzene Concentration"):
            input_df = pd.DataFrame([input_data])
            prediction = model.predict(input_df)[0]
            st.success(
                f"Predicted Benzene Concentration: **{round(prediction, 4)}**"
            )

        st.success("Model training and prediction completed successfully!")

else:
    st.info("Please upload a CSV file to begin.")

