import streamlit as st
import joblib
import pandas as pd
import numpy as np
import random
# load data 
st.set_page_config(
    page_title="Fraud Guard AI", 
    page_icon="🛡️", 
    layout="centered"
)

# 2. CUSTOM SIDEBAR (Project ki details ke liye)
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2889/2889676.png", width=100)
st.sidebar.title("About Project")
st.sidebar.info(
    "This is  Machine Learning based Fraud Detection Engine. "
    "My model blocks real-time, high-risk transactions using the Random Forest algorithm."
)
st.sidebar.markdown("---")
st.sidebar.write("**Developed by:** Gaurav")

@st.cache_resource
def load_data():
    pipeline=joblib.load('pipeline.pkl')
    model=joblib.load('model_file.pkl')
    return pipeline,model

pipeline,model=load_data()

st.title("FRAUD-DETECTION ENGINE")
st.write("Enter the below details carefully")
if 'total_scanned' not in st.session_state:
    st.session_state.total_scanned = 1432
if 'fraud_detected' not in st.session_state:
    st.session_state.fraud_detected = 24

st.write("### 📊 System Overview")
m1, m2, m3 = st.columns(3)

m1.metric(label="Transactions Scanned Today", value=f"{st.session_state.total_scanned:,}", delta="Active")
m2.metric(label="Fraud Detected", value=f"{st.session_state.fraud_detected:,}", delta="Updating...", delta_color="inverse")
m3.metric(label="System Status", value="Secured", delta="Online")
st.divider()

st.write("### 📝 Enter Transaction Details")

col1,col2=st.columns(2)
with col1:
    failed_trans=st.selectbox('Any faild transaction',['Yes','No'])
    unusal_amt=st.selectbox('Any doubetable amount',['Yes','No'])
    unusal_loc=st.selectbox('strange location',['Yes','No'])
    
with col2:
    mlt_trns=st.selectbox('Any multiple transactions short time',['Yes','No'])
    vel_flag=st.selectbox('Any velocity_flag',['Yes','No'])
    device=st.selectbox("Enter Device Type",['Mobile','Desktop','Tablet'])

if st.button("Evaluate Transaction"):
    mapping={'Yes':1,'No':0}
    input_data=pd.DataFrame({
        'unusual_location_flag': [mapping[unusal_loc]],
        'unusual_amount_flag': [mapping[unusal_amt]],
        'failed_transaction_count_24h': [mapping[failed_trans]],
        'velocity_flag': [mapping[vel_flag]],
        'device_type': [device],
        'multiple_transactions_short_time':[mapping[mlt_trns]]
    })
    try:
        transform_data=pipeline.transform(input_data)
        predict=model.predict(transform_data)
        background_transactions = random.randint(20, 50)
        st.session_state.total_scanned += background_transactions 
        
        if predict == 2:
            st.session_state.fraud_detected += 1 # Fraud detect hone par count badhega
            st.error("🚨 RISK IS HIGH (Class 2) - Transaction Blocked!")
        elif predict == 1:
            st.session_state.fraud_detected += 1 # Medium risk par bhi count badhega
            st.warning("⚠️ RISK IS MEDIUM (Class 1) - Manual Review Required.")
        else:
            st.success("✅ NO RISK (Class 0) - Transaction Approved!")
    except Exception as e:
        st.error(f"Processing Error: Ensure all inputs match expected types. Details: {e}")
    
    
        



    
    

