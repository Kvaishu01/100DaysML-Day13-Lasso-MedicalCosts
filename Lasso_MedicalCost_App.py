import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Load Dataset
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv"
    return pd.read_csv(url)

data = load_data()

# Preprocessing
categorical_features = ["sex", "smoker", "region"]
data = pd.get_dummies(data, columns=categorical_features, drop_first=True)

X = data.drop("charges", axis=1)
y = data["charges"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Lasso model
lasso = Lasso(alpha=1.0)
lasso.fit(X_train, y_train)

# Streamlit UI
st.title("💊 Medical Insurance Cost Prediction using Lasso Regression")
st.write("Enter patient details to predict medical charges.")

# User input
age = st.slider("Age", 18, 100, 30)
bmi = st.slider("BMI (Body Mass Index)", 10.0, 50.0, 25.0)
children = st.slider("Number of Children", 0, 5, 0)
sex = st.selectbox("Sex", ["male", "female"])
smoker = st.selectbox("Smoker", ["yes", "no"])
region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

# Convert input to dataframe
input_data = pd.DataFrame({
    "age": [age],
    "bmi": [bmi],
    "children": [children],
    "sex_male": [1 if sex == "male" else 0],
    "smoker_yes": [1 if smoker == "yes" else 0],
    "region_northwest": [1 if region == "northwest" else 0],
    "region_southeast": [1 if region == "southeast" else 0],
    "region_southwest": [1 if region == "southwest" else 0]
})

# Standardize input
input_scaled = scaler.transform(input_data)

# Prediction
if st.button("Predict Medical Cost"):
    prediction = lasso.predict(input_scaled)[0]
    st.success(f"💰 Estimated Medical Cost: ${prediction:,.2f}")
