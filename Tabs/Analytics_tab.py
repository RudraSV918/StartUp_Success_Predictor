import streamlit as st
import plotly.express as px
import pandas as pd
import pickle

def show_analytics(df,knn_model,input_data):
    st.subheader("Similar Startups")
    knn_model = pickle.load(open("Models/knn_model.pkl","rb"))
    df = pd.read_csv("Data/startup_dataset.csv")
    distances,indices = knn_model.kneighbors(input_data)
    similar = df.iloc[indices[0]]
    st.dataframe(similar)
