# 🏦 Customer Churn Prediction — ANN Classification

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15.0-orange?logo=tensorflow)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)
![Keras](https://img.shields.io/badge/Keras-Sequential-darkred?logo=keras)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Preprocessing-yellow?logo=scikitlearn)

---

## 📌 Project Overview

This is a **complete end-to-end Deep Learning project** that predicts whether a bank customer is likely to **churn (leave the bank)** using an **Artificial Neural Network (ANN)** built with TensorFlow and Keras.

The trained model is served through an interactive **Streamlit web application** and deployed to **Streamlit Cloud** for public access — making it usable by anyone without any coding knowledge.

> **Business Problem:** Banks lose significant revenue when customers close their accounts. Identifying at-risk customers early allows the bank to take proactive retention actions.

---

## 🎯 Problem Statement

| Property | Detail |
|---|---|
| **Task** | Binary Classification |
| **Target** | `Exited` — 1 = Churned, 0 = Stayed |
| **Dataset** | Churn_Modelling.csv |
| **Rows** | 10,000 customer records |
| **Features** | 13 input features (after dropping ID columns) |
| **Class Distribution** | 79.6% Not Churned (7,963) · 20.4% Churned (2,037) |

---

## 🗂️ Repository Structure

```
├── app.py                        # Streamlit web application (main entry point)
├── experiments.ipynb             # Data exploration, preprocessing & model training
├── prediction.ipynb              # Inference and prediction examples
├── model.h5                      # Saved Keras ANN model (TensorFlow .h5 format)
├── scaler.pkl                    # Fitted StandardScaler (saved with Pickle)
├── label_encoder_gender.pkl      # Fitted LabelEncoder for Gender column
├── onehot_encoder_geo.pkl        # Fitted OneHotEncoder for Geography column
├── Churn_Modelling.csv           # Raw dataset (10,000 rows)
├── requirements.txt              # Python dependencies
└── README.md                     # Project documentation
```

---

## 📊 Dataset — Features Used

| Feature | Type | Description |
|---|---|---|
| `CreditScore` | Numerical | Customer's credit score (300–900) |
| `Geography` | Categorical | Country: France, Germany, Spain |
| `Gender` | Categorical | Male / Female |
| `Age` | Numerical | Customer age (18–92) |
| `Tenure` | Numerical | Years as a bank customer (0–10) |
| `Balance` | Numerical | Account balance |
| `NumOfProducts` | Numerical | Number of bank products used (1–4) |
| `HasCrCard` | Binary | Has credit card: 0 / 1 |
| `IsActiveMember` | Binary | Is active member: 0 / 1 |
| `EstimatedSalary` | Numerical | Annual estimated salary |
| **`Exited`** | **Target** | **Churned: 1 · Stayed: 0** |

> Columns dropped before training: `RowNumber`, `CustomerId`, `Surname` (irrelevant identifiers)

---

## 🧹 Data Preprocessing Pipeline

Three preprocessing steps are applied and saved as artifacts to ensure **identical transformation at inference time**:

### 1. Label Encoding — Gender
```python
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])
# Female → 0,  Male → 1
# Saved as: label_encoder_gender.pkl
```

### 2. One-Hot Encoding — Geography
```python
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse_output=False)
geo_encoded = ohe.fit_transform(df[['Geography']])
# France → [1,0,0], Germany → [0,1,0], Spain → [0,0,1]
# Saved as: onehot_encoder_geo.pkl
```

### 3. Feature Scaling — StandardScaler
```python
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test  = sc.transform(X_test)      # fit only on train → prevent data leakage
# Saved as: scaler.pkl
```

> ⚠️ **Important:** The scaler is fitted **only on the training set**. The same fitted scaler is then applied to both test data and new user inputs. This prevents data leakage.

---

## 🧠 Model Architecture — ANN (Sequential)

Built using **Keras Sequential API** with TensorFlow 2.15.0:

```
┌──────────────────────────────┬────────────────────┬─────────┐
│ Layer (type)                 │ Output Shape       │  Params │
├──────────────────────────────┼────────────────────┼─────────┤
│ dense_2  (Dense — ReLU)      │ (None, 64)         │     832 │
│ dense_3  (Dense — ReLU)      │ (None, 32)         │   2,080 │
│ dense_4  (Dense — Sigmoid)   │ (None, 1)          │      33 │
└──────────────────────────────┴────────────────────┴─────────┘
  Total Trainable Parameters: 2,945
```

### Parameter Calculation
```
Input features: 12  (after encoding Geography → 3 cols, dropping IDs)

Layer 1 (Dense 64):   12 × 64 weights  +  64 bias  =   832 params
Layer 2 (Dense 32):   64 × 32 weights  +  32 bias  = 2,080 params
Layer 3 (Dense  1):   32 ×  1 weights  +   1 bias  =    33 params
──────────────────────────────────────────────────────────────────
Total                                               = 2,945 params
```

### Compilation
```python
model.compile(
    optimizer = 'adam',
    loss      = 'binary_crossentropy',    # standard for binary classification
    metrics   = ['accuracy']
)
```

| Hyperparameter | Value | Reason |
|---|---|---|
| Activation (hidden) | ReLU | Fast convergence, avoids vanishing gradient |
| Activation (output) | Sigmoid | Outputs probability between 0 and 1 |
| Optimizer | Adam | Adaptive learning rate, works well out of the box |
| Loss | Binary Crossentropy | Standard loss for binary classification |
| Epochs | 100 | Sufficient for convergence on this dataset size |
| Batch Size | 32 | Good balance of speed and gradient stability |

---

## 📈 Training — TensorBoard Visualization

Training metrics (loss and accuracy) are logged and visualized in **TensorBoard**:

```python
import datetime
from tensorflow.keras.callbacks import TensorBoard

log_dir = 'logs/fit/' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
tb_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

model.fit(
    X_train, y_train,
    epochs          = 100,
    batch_size      = 32,
    validation_data = (X_test, y_test),
    callbacks       = [tb_callback]
)
```

**To launch TensorBoard locally:**
```bash
tensorboard --logdir logs/fit
```
Then open `http://localhost:6006` in your browser.

---

## 💾 Model Saving

Two file formats are saved to enable both inference and deployment:

```python
# Save the Keras model
model.save('model.h5')                   # HDF5 format — includes architecture + weights

# Save preprocessing artifacts
import pickle

with open('label_encoder_gender.pkl', 'wb') as f:
    pickle.dump(label_encoder_gender, f)

with open('onehot_encoder_geo.pkl', 'wb') as f:
    pickle.dump(onehot_encoder_geo, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
```

| File | Format | Contents |
|---|---|---|
| `model.h5` | HDF5 / Keras | Full model: architecture + trained weights + optimizer state |
| `scaler.pkl` | Pickle | StandardScaler fitted on training data |
| `label_encoder_gender.pkl` | Pickle | LabelEncoder for Gender (Female=0, Male=1) |
| `onehot_encoder_geo.pkl` | Pickle | OneHotEncoder for Geography (France, Germany, Spain) |

---

## 🌐 Streamlit Application — `app.py`

The web app loads all saved artifacts and provides a real-time prediction interface:

### How it works
1. User selects inputs via sliders and dropdowns
2. App applies the **same preprocessing** as training (encode → one-hot → scale)
3. Scaled input is passed to the Keras model
4. Churn probability (0.0 – 1.0) is displayed with a clear verdict

### Inference Pipeline (exact `app.py` code)

```python
import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# Load saved artifacts
model                = tf.keras.models.load_model('model.h5')
label_encoder_gender = pickle.load(open('label_encoder_gender.pkl', 'rb'))
onehot_encoder_geo   = pickle.load(open('onehot_encoder_geo.pkl',   'rb'))
scaler               = pickle.load(open('scaler.pkl',               'rb'))

st.title("Customer Churn Prediction")

# ── Collect user inputs ──────────────────────────────────────────────────────
geography        = st.selectbox('Geography',         onehot_encoder_geo.categories_[0])
gender           = st.selectbox('Gender',            label_encoder_gender.classes_)
age              = st.slider('Age',                  18, 92)
balance          = st.number_input('Balance')
credit_score     = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure           = st.slider('Tenure',               0, 10)
num_of_products  = st.slider('Number of Products',   1, 4)
has_cr_card      = st.selectbox('Has Credit Card',   [0, 1])
is_active_member = st.selectbox('Is Active Member',  [0, 1])

# ── Preprocess exactly as training ──────────────────────────────────────────
input_data = pd.DataFrame({
    'CreditScore':       [credit_score],
    'Gender':            [label_encoder_gender.transform([gender])[0]],
    'Age':               [age],
    'Tenure':            [tenure],
    'Balance':           [balance],
    'NumOfProducts':     [num_of_products],
    'HasCrCard':         [has_cr_card],
    'IsActiveMember':    [is_active_member],
    'EstimatedSalary':   [estimated_salary]
})

geo_encoded    = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded,
                     columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

input_data        = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)
input_data_scaled = scaler.transform(input_data)

# ── Predict and display ──────────────────────────────────────────────────────
prediction_proba = model.predict(input_data_scaled)[0][0]
st.write(f'Churn Probability: {prediction_proba:.2f}')

if prediction_proba > 0.5:
    st.write('The customer is likely to churn.')
else:
    st.write('The customer is not likely to churn.')
```

### App UI Inputs
| Input | Widget | Range / Options |
|---|---|---|
| Geography | Selectbox | France · Germany · Spain |
| Gender | Selectbox | Female · Male |
| Age | Slider | 18 – 92 |
| Balance | Number Input | 0.00 – any |
| Credit Score | Number Input | 300 – 900 |
| Estimated Salary | Number Input | 0.00 – any |
| Tenure | Slider | 0 – 10 years |
| Number of Products | Slider | 1 – 4 |
| Has Credit Card | Selectbox | 0 · 1 |
| Is Active Member | Selectbox | 0 · 1 |

---

## 🚀 Running the Project Locally

### Prerequisites
- Python 3.11
- Git

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/ann-customer-churn-prediction.git
cd ann-customer-churn-prediction

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the Streamlit app
streamlit run app.py
```

The app opens automatically at `http://localhost:8501`

---

## ☁️ Deployment — Streamlit Cloud

This app is deployed and publicly accessible via **Streamlit Community Cloud** (free tier).

### Deployment Steps
1. Push the entire repository to **GitHub** (all files including `.pkl`, `.h5`, `requirements.txt`)
2. Go to [share.streamlit.io](https://share.streamlit.io) → Sign in with GitHub
3. Click **"New app"** → Select your repo → Set `app.py` as the main file
4. Click **Deploy** — Streamlit Cloud auto-installs dependencies and serves the app

### requirements.txt
```
tensorflow==2.15.0
pandas
numpy
scikit-learn
tensorboard
matplotlib
streamlit
```

> **Note:** TensorFlow 2.15.0 is pinned to ensure compatibility between the saved `.h5` model and the serving environment.

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Deep Learning Framework | TensorFlow 2.15.0 + Keras |
| Preprocessing | scikit-learn (LabelEncoder, OneHotEncoder, StandardScaler) |
| Data Manipulation | pandas, NumPy |
| Model Persistence | Keras `.h5` + Python Pickle |
| Visualization | TensorBoard, Matplotlib |
| Web Application | Streamlit |
| Deployment | Streamlit Community Cloud |
| Language | Python 3.11 |

---

## 🔑 Key Concepts Demonstrated

| Concept | Implementation |
|---|---|
| **Binary Classification** | Sigmoid output + Binary Crossentropy loss |
| **Feature Engineering** | Label Encoding, One-Hot Encoding, Standardization |
| **Data Leakage Prevention** | Scaler fitted on train set only, applied to test/inference |
| **Preprocessing Persistence** | Encoders + scaler saved with Pickle, reloaded at inference |
| **Model Persistence** | Keras `.h5` format, reloaded with `tf.keras.models.load_model` |
| **Training Monitoring** | TensorBoard callbacks → loss/accuracy curves |
| **Deployment** | Streamlit app served on Streamlit Community Cloud |
| **End-to-End Pipeline** | Raw CSV → preprocessing → training → saving → web app → cloud |

---


## 📊 Model Results — Actual Evaluation on Test Set

Evaluated on **2,000 held-out records** (20% of the dataset, random_state=42):

| Metric | Value |
|---|---|
| **Test Accuracy** | **86.05%** |
| **Test Loss** | 0.3469 |
| **Precision (Churned)** | 0.74 |
| **Recall (Churned)** | 0.44 |
| **F1-Score (Churned)** | 0.56 |
| **Precision (Not Churned)** | 0.88 |
| **Recall (Not Churned)** | 0.96 |
| **F1-Score (Not Churned)** | 0.92 |

### Classification Report
```
              precision    recall  f1-score   support

 Not Churned       0.88      0.96      0.92      1607
     Churned       0.74      0.44      0.56       393

    accuracy                           0.86      2000
   macro avg       0.81      0.70      0.74      2000
weighted avg       0.85      0.86      0.85      2000
```

### Confusion Matrix
```
                  Predicted: Stay    Predicted: Churn
Actual: Stay           1547                  60
Actual: Churn           219                 174
```

### Results Interpretation
- The model achieves **86% overall accuracy** on unseen data.
- It correctly identifies **96% of loyal customers** (high recall for Not Churned) — preventing unnecessary retention campaigns.
- **Recall for churned customers is 44%** — meaning the model catches roughly 4 in 10 at-risk customers. This is expected behavior for an imbalanced dataset (20% churn rate) without oversampling.
- **Potential improvement:** Applying SMOTE (Synthetic Minority Oversampling) or adjusting the classification threshold from 0.5 to ~0.3 would increase churn recall at the cost of more false positives.

---

## 🔄 How It Works — Step-by-Step Flow

```
Raw CSV (10,000 rows)
        │
        ▼
┌──────────────────────────────────────────────────────────┐
│  Feature Engineering                                     │
│  • Drop: RowNumber, CustomerId, Surname                  │
│  • LabelEncoder:  Gender → {Female:0, Male:1}            │
│  • OneHotEncoder: Geography → [France|Germany|Spain]     │
│  • StandardScaler: normalize all 12 numerical features   │
└──────────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────────┐
│  ANN Training (TensorFlow / Keras)                       │
│  Input (12 features)                                     │
│    → Dense(64, ReLU)                                     │
│    → Dense(32, ReLU)                                     │
│    → Dense(1,  Sigmoid)  → Churn Probability [0, 1]      │
│                                                          │
│  Optimizer: Adam   |   Loss: Binary Crossentropy         │
│  Epochs: 100       |   Batch Size: 32                    │
│  Callbacks: TensorBoard (logs training curves)           │
└──────────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────────┐
│  Save Artifacts                                          │
│  • model.h5                   (Keras model)              │
│  • scaler.pkl                 (StandardScaler)           │
│  • label_encoder_gender.pkl   (LabelEncoder)             │
│  • onehot_encoder_geo.pkl     (OneHotEncoder)            │
└──────────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────────┐
│  Streamlit App (app.py)                                  │
│  User fills form → same preprocessing applied →          │
│  model.predict() → Churn Probability displayed           │
└──────────────────────────────────────────────────────────┘
        │
        ▼
   ☁️  Deployed on Streamlit Community Cloud
```

---

## 🎓 Interview Q&A — Key Concepts

These are questions an interviewer may ask about this project, with answers based on the actual implementation:

---

**Q1: Why did you use an ANN instead of a traditional ML model like Logistic Regression or XGBoost?**

> ANNs can automatically learn non-linear feature interactions without manual feature engineering. For tabular data at this scale (10K rows, 12 features), an ANN with 2 hidden layers is a good starting point that demonstrates deep learning concepts while still being lightweight and fast to train. However, for purely tabular data, XGBoost often outperforms ANNs — this project focuses on demonstrating the full DL workflow rather than optimizing benchmark accuracy.

---

**Q2: What preprocessing steps did you apply and why?**

> Three steps:
> 1. **Label Encoding** for Gender — binary categorical, no ordinality issue, so a simple 0/1 encoding is sufficient.
> 2. **One-Hot Encoding** for Geography — 3 categories (France, Germany, Spain) with no ordinal relationship. One-hot prevents the model from assuming France < Germany < Spain.
> 3. **StandardScaler** — brings all features to mean=0, std=1. Neural networks are sensitive to feature magnitude differences; without scaling, a feature like Balance (0–250K) would dominate over Tenure (0–10).

---

**Q3: How did you prevent data leakage with the scaler?**

> The StandardScaler is fitted **only on the training set** using `fit_transform()`. The test set and new user inputs use only `transform()`. This ensures the model never sees any statistical information from the test set during training. All three fitted preprocessing objects are saved with Pickle and reloaded in the Streamlit app to apply the **exact same transformation** to new inputs.

---

**Q4: Why is recall for churned customers only 44%? Is that a problem?**

> The dataset is imbalanced — only 20.4% of customers churned. Without handling imbalance, the model learns to predict "Not Churned" more often since that minimizes loss. A 44% churn recall means it catches 174 out of 393 true churners. Potential fixes: (1) SMOTE oversampling on the training set, (2) `class_weight` parameter in `model.fit()`, (3) lowering the classification threshold from 0.5 to 0.3–0.4. The trade-off is more false positives (flagging loyal customers as at-risk).

---

**Q5: Why did you use Binary Crossentropy as the loss function?**

> Binary Crossentropy is the standard loss function for binary classification with a sigmoid output. It penalizes confident wrong predictions heavily (e.g., predicting 0.95 when the true label is 0 incurs a very high loss), which pushes the model to be appropriately uncertain. Mean Squared Error is not ideal for classification because it doesn't account for the probabilistic nature of the sigmoid output.

---

**Q6: What does the Sigmoid activation in the output layer do?**

> Sigmoid maps any real number to the range (0, 1), making it interpretable as a probability. If the output is 0.73, it means the model predicts a 73% probability that the customer will churn. The threshold of 0.5 converts this to a binary decision: ≥0.5 → Churned, <0.5 → Not Churned.

---

**Q7: Why save the model in .h5 format? What does it contain?**

> `.h5` (HDF5) is Keras's legacy serialization format. It stores: (1) the model **architecture** — layer types, connections, activations; (2) the trained **weights** — all 2,945 parameter values; (3) the **optimizer state** — allows resuming training. The newer alternative is `.keras` format, but `.h5` is still widely supported and required for compatibility with TensorFlow 2.x deployments.

---

**Q8: What is Dropout and why didn't you use it in this project?**

> Dropout randomly sets neuron outputs to 0 during training, forcing the network to learn redundant representations and preventing overfitting. This project uses a relatively simple architecture (64→32→1) on 10K rows — the model doesn't appear to heavily overfit, so Dropout wasn't added. In a deeper network or with fewer data, adding `Dropout(0.2)` between layers would be a natural next step.

---

**Q9: How does the Streamlit app apply the same preprocessing as training?**

> By loading the **fitted** scaler and encoders from Pickle files. At inference time, the app runs:
> 1. `label_encoder_gender.transform([gender])` — same fitted encoder
> 2. `onehot_encoder_geo.transform([[geography]])` — same fitted OHE
> 3. `scaler.transform(input_data)` — same fitted scaler with training mean/std
>
> This guarantees the input features are in the exact same numerical range and encoding the model was trained on.

---

**Q10: How would you improve this project further?**

> Several directions:
> - **Address class imbalance:** SMOTE, class weights, or threshold tuning to improve churn recall
> - **Hyperparameter tuning:** Keras Tuner to search optimal number of layers, neurons, learning rate
> - **Feature engineering:** Adding interaction terms like Balance/Salary ratio, age groups
> - **Model comparison:** Benchmark against XGBoost and LightGBM (often better for tabular data)
> - **Experiment tracking:** MLflow or Weights & Biases for systematic experiment logging
> - **REST API:** Wrap the model in a FastAPI endpoint for programmatic access alongside the Streamlit UI

---

## 📦 Project Learnings

Building this project covered these practical skills end-to-end:

| Skill | How It Was Applied |
|---|---|
| Pandas / NumPy | Data loading, cleaning, feature selection, DataFrame manipulation |
| scikit-learn Preprocessing | LabelEncoder, OneHotEncoder, StandardScaler, train_test_split |
| Keras Sequential API | Building, compiling, training, evaluating ANN |
| Callback — TensorBoard | Logging training metrics to a folder for visualization |
| Pickle | Serializing and deserializing sklearn objects |
| Keras `.save()` / `load_model()` | Persisting and restoring a trained neural network |
| Streamlit | Building an interactive web UI in pure Python |
| Streamlit Cloud | Free cloud deployment directly from a GitHub repository |
| Inference pipeline | Applying training-time preprocessing to new, unseen inputs |

---

## 📬 Contact

**Sai Krishna Reddy Ragula**
Software Engineer — American Express | Deep Learning & NLP Enthusiast

🔗 [LinkedIn](https://www.linkedin.com/in/saikrishnareddyragula)
🐙 [GitHub](https://github.com/saikrishnareddy1731)

---

## 📝 License

This project is open-source under the [MIT License](LICENSE).
