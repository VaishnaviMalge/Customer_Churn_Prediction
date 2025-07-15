import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the trained model
with open("final_gb_classifier_model.pkl", "rb") as file:
    model = pickle.load(file)

# Function to preprocess input data
# pickle file only contains predictive model, Hence we again have to perform preprocessing 
def preprocess_input(data):
    
    df = pd.DataFrame(data, index=[0])                       # Convert input data to DataFrame
    
    # Convert categorical variables to numeric
    df['InternetService'] = df['InternetService'].map({'DSL': 0, 'Fiber optic': 1, 'No': 2})
    df['Contract'] = df['Contract'].map({'Month-to-month': 0, 'One year': 1, 'Two year': 2})
    df['PaymentMethod'] = df['PaymentMethod'].map({'Electronic check': 0, 'Mailed check': 1, 'Bank transfer (automatic)': 2, 'Credit card (automatic)': 3})
    
    column1 = ['OnlineSecurity', 'OnlineBackup', 'TechSupport', 'DeviceProtection']
    for x in column1:
        df[x] = df[x].map({'No':0,'Yes':1,'No Internet Service':2})
    
    columns = ['gender','SeniorCitizen','Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'StreamingTV', 'StreamingMovies', 'PaperlessBilling']
    for i in columns:
        df[i] = df[i].map({'No':0,'Yes':1})
    
    return df

# Streamlit User Interface
st.title("Customer Churn Prediction")

# Collect user inputs
gender = st.radio("Gender", ['No', 'Yes'])
senior_citizen = st.radio("Senior Citizen", ['No', 'Yes'])
partner = st.radio("Partner", ['No', 'Yes'])
dependents = st.radio("Dependents", ['No', 'Yes'])
phone_service = st.radio("Phone Service", ['No', 'Yes'])
multiple_lines = st.radio("Multiple Lines", ['No', 'Yes'])
internet_service = st.selectbox("Internet Service", ['DSL', 'Fiber optic', 'No'])
online_security = st.radio("Online Security", ['No','Yes','No Internet Service'])
online_backup = st.radio("Online Backup", ['No','Yes','No Internet Service'])
device_protection = st.radio("Device Protection", ['No','Yes','No Internet Service'])
tech_support = st.radio("Tech Support", ['No','Yes','No Internet Service'])
streaming_tv = st.radio("Streaming TV", ['No', 'Yes'])
streaming_movies = st.radio("Streaming Movies", ['No', 'Yes'])
contract = st.selectbox("Contract", ['Month-to-month', 'One year', 'Two year'])
paperless_billing = st.radio("Paperless Billing", ['No', 'Yes'])
payment_method = st.selectbox("Payment Method", ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
monthly_charges = st.number_input("Monthly Charges", value=0.0)
total_charges = st.number_input("Total Charges", value=0.0)
tenure_bins = st.number_input("Tenure Bins", value=0)

# Make prediction
if st.button("Predict"):
    # When Predict button is pressed Create dictionary from user inputs
    user_data = {
        'gender': gender,
        'SeniorCitizen': senior_citizen,
        'Partner': partner,
        'Dependents': dependents,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'tenure_bins': tenure_bins
    }
    
    # Preprocess input data
    processed_data = preprocess_input(user_data)
    
    # Make prediction
    prediction = model.predict(processed_data)
    
    # Display prediction result
    if prediction[0] == 1:
        st.write("The customer is likely to churn.")
    else:
        st.write("The customer is likely to stay.")
