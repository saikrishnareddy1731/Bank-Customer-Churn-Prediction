import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle
import os

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Bank Customer Churn Prediction",
    page_icon="🏦",
    layout="wide"
)

# ── Load model and artifacts ──────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model('model.h5')
    with open('label_encoder_gender.pkl', 'rb') as f:
        label_encoder_gender = pickle.load(f)
    with open('onehot_encoder_geo.pkl', 'rb') as f:
        onehot_encoder_geo = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, label_encoder_gender, onehot_encoder_geo, scaler

model, label_encoder_gender, onehot_encoder_geo, scaler = load_artifacts()

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🏦 Customer Churn Prediction")
st.markdown(
    "This app predicts whether a bank customer is likely to **churn (leave the bank)** "
    "using an Artificial Neural Network trained on 10,000 customer records."
)
st.markdown("---")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["🔮 Predict Churn", "📊 Model Training Insights"])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — Prediction
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Enter Customer Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        geography    = st.selectbox('Geography',        onehot_encoder_geo.categories_[0])
        gender       = st.selectbox('Gender',           label_encoder_gender.classes_)
        age          = st.slider('Age',                 18, 92, 35)
        tenure       = st.slider('Tenure (years)',      0, 10, 3)

    with col2:
        credit_score     = st.number_input('Credit Score',     min_value=300, max_value=900, value=650)
        balance          = st.number_input('Balance',          min_value=0.0, value=50000.0, step=1000.0)
        estimated_salary = st.number_input('Estimated Salary', min_value=0.0, value=60000.0, step=1000.0)

    with col3:
        num_of_products  = st.slider('Number of Products', 1, 4, 1)
        has_cr_card      = st.selectbox('Has Credit Card',  [0, 1],
                                        format_func=lambda x: "Yes" if x else "No")
        is_active_member = st.selectbox('Is Active Member', [0, 1],
                                        format_func=lambda x: "Yes" if x else "No")

    st.markdown("")
    predict_btn = st.button("🔍 Predict", use_container_width=True)

    if predict_btn:
        # Preprocess
        input_data = pd.DataFrame({
            'CreditScore':     [credit_score],
            'Gender':          [label_encoder_gender.transform([gender])[0]],
            'Age':             [age],
            'Tenure':          [tenure],
            'Balance':         [balance],
            'NumOfProducts':   [num_of_products],
            'HasCrCard':       [has_cr_card],
            'IsActiveMember':  [is_active_member],
            'EstimatedSalary': [estimated_salary]
        })

        geo_encoded    = onehot_encoder_geo.transform([[geography]]).toarray()
        geo_encoded_df = pd.DataFrame(
            geo_encoded,
            columns=onehot_encoder_geo.get_feature_names_out(['Geography'])
        )
        input_data        = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)
        input_data_scaled = scaler.transform(input_data)

        # Predict
        prediction_proba = model.predict(input_data_scaled)[0][0]

        st.markdown("---")
        st.subheader("Prediction Result")

        col_prob, col_verdict = st.columns(2)

        with col_prob:
            st.metric(label="Churn Probability", value=f"{prediction_proba:.2%}")
            st.progress(float(prediction_proba))

        with col_verdict:
            if prediction_proba > 0.5:
                st.error("⚠️ This customer is **likely to churn**.")
                st.markdown("**Recommended action:** Contact with a retention offer.")
            else:
                st.success("✅ This customer is **NOT likely to churn**.")
                st.markdown("**Status:** Customer appears satisfied and loyal.")

# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — Model Training Insights
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("📈 TensorBoard Training Logs")
    st.markdown(
        "The model was trained for **100 epochs** using the Adam optimizer with "
        "Binary Crossentropy loss. Training metrics were logged via TensorBoard callbacks."
    )
    st.markdown("---")

    # Evaluation Accuracy
    st.markdown("#### Evaluation Accuracy vs Iterations")
    st.markdown(
        "Best validation accuracy achieved: **~85.8%** (smoothed: 0.8583) across multiple runs."
    )
    if os.path.exists("tensorboard_accuracy.png"):
        st.image("tensorboard_accuracy.png",
                 caption="TensorBoard — evaluation_accuracy_vs_iterations",
                 use_container_width=True)
    else:
        st.warning("tensorboard_accuracy.png not found. Please add it to the repo root.")

    st.markdown("---")

    # Weight Histograms
    st.markdown("#### Weight & Bias Distributions (Histograms)")
    st.markdown(
        "Shows how layer weights evolved over epochs — a healthy spread confirms "
        "no vanishing or exploding gradients."
    )
    if os.path.exists("tensorboard_histograms.png"):
        st.image("tensorboard_histograms.png",
                 caption="TensorBoard — dense layer bias and kernel histograms",
                 use_container_width=True)
    else:
        st.warning("tensorboard_histograms.png not found. Please add it to the repo root.")

    st.markdown("---")

    # Model Summary
    st.markdown("#### Model Architecture Summary")
    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.markdown("**Architecture**")
        st.code(
            "Input  (12 features)\n"
            "Dense  (64, ReLU)\n"
            "Dense  (32, ReLU)\n"
            "Dense  ( 1, Sigmoid)",
            language="text"
        )
    with col_b:
        st.markdown("**Training Config**")
        st.code(
            "Optimizer : Adam\n"
            "Loss      : Binary Crossentropy\n"
            "Epochs    : 100\n"
            "Batch Size: 32",
            language="text"
        )
    with col_c:
        st.markdown("**Test Set Performance**")
        st.code(
            "Accuracy  : 86.05%\n"
            "Loss      : 0.3469\n"
            "Precision : 0.74 (churn)\n"
            "Recall    : 0.44 (churn)",
            language="text"
        )
