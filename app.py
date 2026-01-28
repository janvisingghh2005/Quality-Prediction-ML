import streamlit as st
import pandas as pd
import numpy as np

# ===============================
# SAFE ML IMPORTS (NO CRASH)
# ===============================
try:
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    sklearn_available = True
except Exception:
    sklearn_available = False

# ===============================
# Page Configuration
# ===============================
st.set_page_config(
    page_title="Benzene Concentration Prediction",
    layout="wide"
)

st.title("Benzene Concentration Prediction in Isomerization Unit")
st.write(
    "End-to-end data analytics and machine learning application for "
    "industrial process data."
)

# ===============================
# File Upload
# ===============================
uploaded_file = st.file_uploader(
    "Upload Process & Lab Data (CSV)",
    type=["csv"]
)

if uploaded_file is None:
    st.info("Please upload a CSV file to begin.")
    st.stop()

# ===============================
# READ & CLEAN DATA (CRITICAL FIX)
# ===============================
df = pd.read_csv(uploaded_file)

cleaned_df = df.copy()

for col in df.columns:
    # Try numeric conversion WITHOUT regex first
    try:
        cleaned_df[col] = pd.to_numeric(df[col])
    except Exception:
        # If it fails, try cleaning units then convert
        try:
            cleaned_df[col] = (
                df[col]
                .astype(str)
                .str.replace(",", "", regex=False)
                .str.replace(r"[^\d\.\-]+", "", regex=True)
                .replace("", np.nan)
                .astype(float)
            )
        except Exception:
            # Leave column as-is (categorical / ID / tag)
            cleaned_df[col] = df[col]


# ===============================
# Data Preview
# ===============================
st.subheader("Data Preview")
st.dataframe(df.head())

# ===============================
# Debug: Show Detected Numeric Columns
# ===============================
st.subheader("Detected Numeric Columns")
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
st.write(numeric_cols)

if len(numeric_cols) == 0:
    st.error(
        "No numeric columns detected even after cleaning.\n\n"
        "Please check your CSV values."
    )
    st.stop()

# ===============================
# Dataset Overview
# ===============================
c1, c2 = st.columns(2)
c1.metric("Rows", df.shape[0])
c2.metric("Columns", df.shape[1])

# ===============================
# Missing Values
# ===============================
st.subheader("Missing Values")
missing = df.isnull().sum()
st.write(missing)

st.bar_chart(
    pd.DataFrame(
        {
            "Count": [
                df.size - missing.sum(),
                missing.sum()
            ]
        },
        index=["Available", "Missing"]
    )
)

# ===============================
# Feature Selection (FIXED)
# ===============================
st.subheader("Feature Selection")

features = st.multiselect(
    "Select input features",
    numeric_cols
)

target = st.selectbox(
    "Select target (Benzene concentration)",
    numeric_cols
)

if not features or not target:
    st.warning("Select at least one feature and a target.")
    st.stop()

if target in features:
    features.remove(target)

# ===============================
# Feature Comparison
# ===============================
if len(features) >= 2:
    st.subheader("Feature Comparison")

    fx = st.selectbox("Feature X", features)
    fy = st.selectbox("Feature Y", features, index=1)

    comp = df[[fx, fy]].dropna()

    st.scatter_chart(comp, x=fx, y=fy)
    st.line_chart(comp)

    corr = comp[fx].corr(comp[fy])
    st.metric("Pearson Correlation", round(corr, 4))

    if abs(corr) >= 0.8:
        st.success("Strong correlation detected")
    elif abs(corr) >= 0.5:
        st.warning("Moderate correlation detected")
    else:
        st.info("Weak correlation detected")

    st.download_button(
        "Download Comparison Data",
        comp.to_csv(index=False),
        file_name=f"{fx}_vs_{fy}.csv"
    )

# ===============================
# MACHINE LEARNING SECTION
# ===============================
st.subheader("Machine Learning")

if not sklearn_available:
    st.error(
        "Machine learning libraries are not available in this cloud environment.\n\n"
        "EDA and feature analysis work correctly.\n"
        "Model training can be demonstrated locally."
    )
    st.stop()

# ===============================
# Model Training
# ===============================
X = df[features].fillna(df[features].mean())
y = df[target].fillna(df[target].mean())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

if st.button("Train Random Forest Model"):
    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    r2 = r2_score(y_test, preds)
    rmse = mean_squared_error(y_test, preds, squared=False)
    mae = mean_absolute_error(y_test, preds)

    m1, m2, m3 = st.columns(3)
    m1.metric("R²", round(r2, 4))
    m2.metric("RMSE", round(rmse, 4))
    m3.metric("MAE", round(mae, 4))

    # ===============================
    # Feature Importance
    # ===============================
    st.subheader("Feature Importance")
    imp_df = pd.DataFrame({
        "Feature": features,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    st.bar_chart(imp_df.set_index("Feature"))

    # ===============================
    # Final Prediction
    # ===============================
    st.subheader("Final Benzene Concentration Prediction")

    input_vals = {}
    for f in features:
        input_vals[f] = st.number_input(
            f"Enter {f}",
            float(X[f].mean())
        )

    if st.button("Predict"):
        pred = model.predict(pd.DataFrame([input_vals]))[0]
        st.success(
            f"Predicted Benzene Concentration: **{round(pred, 4)}**"
        )


