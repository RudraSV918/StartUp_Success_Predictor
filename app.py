## this file will be actual frontend 

import streamlit as st
import pandas as pd
import pickle
import json

from Tabs.Prediction_tab import show_prediction
from Tabs.Analytics_tab import show_analytics
from Tabs.Downloads_tab import show_downloads

from Auth.auth import signup_user,login_user

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False


st.set_page_config(layout="wide")
st.markdown("<h1 style='text-align: center; white-space: nowrap;'>Startup Success Prediction App</h1>", unsafe_allow_html=True)

def load_models():
    log_model = pickle.load(open("Models/logistic_model.pkl","rb"))  ## rb means read binary 
    knn_model = pickle.load(open("Models/knn_model.pkl","rb"))
    rf_model = pickle.load(open("Models/random_forest_model.pkl","rb"))
    svm_model = pickle.load(open("Models/svm_model.pkl","rb"))
    return log_model,knn_model,rf_model,svm_model

log_model,knn_model,rf_model,svm_model = load_models()

def load_data():
    return pd.read_csv("Data/startup_dataset.csv")

df = load_data()

def load_metrics():
    with open("Models/logistic_metrics.json") as f:
        log_metrics = json.load(f)
    with open("Models/knn_metrics.json") as f:
        knn_metrics = json.load(f)
    with open("Models/random_forest_metrics.json") as f:
        rf_metrics = json.load(f)
    with open("Models/svm_metrics.json") as f:
        svm_metrics = json.load(f)

    return log_metrics,knn_metrics,rf_metrics,svm_metrics


log_metrics,knn_metrics,rf_metrics,svm_metrics = load_metrics()


st.sidebar.header("Startup Parameters")
experience = st.sidebar.slider("Founder Experience ",0,20,5)
team = st.sidebar.slider("Team Size",1,60,10)
funding = st.sidebar.slider("Funding Amount",0.1,50.0,5.0)
market = st.sidebar.slider("Market Size",1,10,5)
innovation = st.sidebar.slider("Innovation Score",1,10,6)
marketing = st.sidebar.slider("Marketing Budget",0.1,20.0,3.0)
competition = st.sidebar.slider("Competition Level",1,10,5)
revenue = st.sidebar.slider("Revenue Growth",-10,100,20)

industry = 1
education =1
stage =2

input_data = pd.DataFrame([{
    "FounderExperience": experience,
    "TeamSize": team,
    "FundingAmount": funding,
    "MarketSize": market,
    "InnovationScore": innovation,
    "MarketingBudget": marketing,
    "CompetitionLevel": competition,
    "IndustryType": industry,
    "FounderEducation": education,
    "ProductStage": stage,
    "RevenueGrowth": revenue
}])

prob_log = log_model.predict_proba(input_data)[0][1] 
prob_knn = knn_model.predict_proba(input_data)[0][1]
prob_rf = rf_model.predict_proba(input_data)[0][1]
prob_svm = svm_model.predict_proba(input_data)[0][1]
## login-Signup,Starts
st.sidebar.title("USER AUTHENTICATION")

menu = ["Login","Sign Up"]
choice = st.sidebar.selectbox("Select Option",menu)
if not st.session_state.logged_in:    
    if choice == "Login":
        username = st.sidebar.text_input("Username")
        password = st.sidebar.text_input("Password",type="password")
        if st.sidebar.button("Login"):
            if login_user(username,password):
                st.session_state.logged_in = True
                st.session_state.username = username
                # st.rerun()
            else:
                st.error("Invaild Credentials.")
    elif choice == "Sign Up":
        new_user = st.sidebar.text_input("New Username")
        new_pass = st.sidebar.text_input("New Password",type="password")
        if st.sidebar.button("Create Account"):
            if signup_user(new_user,new_pass):
                st.success("Account created successfully! Please login.")
            else:
                st.error("Username already exists. Please choose a different one.")


if st.session_state.logged_in:
    st.sidebar.markdown(f"Logged in as **{st.session_state.username}**") 
    if st.sidebar.button("logout"):
        st.session_state.logged_in = False
        st.session_state.username = ""

if st.session_state.logged_in: ##no login

    
    tab1,tab2,tab3 = st.tabs(["Prediction","Analytics","Downloads"])

    with tab1:
        show_prediction(prob_log,prob_knn,prob_rf,prob_svm,log_model,log_metrics,knn_metrics,rf_metrics,svm_metrics,[experience,team,funding,market,innovation,marketing,competition])

    with tab2:
        show_analytics(df,knn_model,input_data)

    with tab3:
        show_downloads(prob_log,input_data)

else:
    st.warning("Please login or sign up to access the Dashboard.")