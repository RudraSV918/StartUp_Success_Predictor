import streamlit as st
import plotly.express as px
import pandas as pd
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
import os


def generate_report(prob_log,input_data):
        st.subheader("DOWNLOAD Your PDF Here...")

        doc = SimpleDocTemplate("Report.pdf")
        styles = getSampleStyleSheet()

        content = []
        content.append(Paragraph("Startup Success Prediction Report", styles['Title']))
        content.append(Paragraph(f"Success Probability: {prob_log*100:.2f}%", styles['Normal']))
        content.append(Paragraph("Input details:", styles['Heading2']))

        for col in input_data.columns:
            value = input_data.iloc[0][col]
            content.append(Paragraph(f"{col}: {value}", styles['Normal']))
        doc.build(content)
        return "Report.pdf"

def show_downloads(prob_log,input_data):
    if st.button("Download Report"):
        pdf_file = generate_report(prob_log,input_data)

        with open(pdf_file, "rb") as f:
            st.download_button(
                label="Download PDF",
                data=f,
                file_name="Startup_Success_Report.pdf",
                mime="application/pdf"
            )

    def save_data(input_data,prob_log):
        input_data["Prediction"]=prob_log
        BASE_DIR = os.path.dirname(os.path.dirname(__file__))
        file_path = os.path.join(BASE_DIR, "Data", "Client_history.csv")

        if os.path.exists(file_path):
            input_data.to_csv(file_path,mode="a",header=False,index=False)
        else:
            input_data.to_csv(file_path,index=False)

    if st.button("Save Data"):
        save_data(input_data.copy(),prob_log)
        st.success("Data saved successfully!")

    if os.path.exists("Data/Client_history.csv"):
        st.subheader("Past Prediction History")
        history = pd.read_csv("Data/Client_history.csv")
        st.dataframe(history.tail(10))