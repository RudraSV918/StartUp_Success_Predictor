import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

def show_prediction(prob_log,prob_knn,prob_rf,prob_svm,log_model,log_metrics,knn_metrics,rf_metrics,svm_metrics,input_values):
    st.subheader("Prediction Result")
    colA,colB,colC,colD = st.columns(4)
    colA.metric("Logistic Probability",f"{prob_log*100:.2f}%")
    colB.metric("KNN Probability",f"{prob_knn*100:.2f}%")
    colC.metric("Random Forest Probability",f"{prob_rf*100:.2f}%")
    colD.metric("SVM Probability",f"{prob_svm*100:.2f}%")

    st.subheader("Model Accuracy (Train vs Test)")

    df_acc = pd.DataFrame({
        "Model": ["Logistic", "KNN", "Random Forest", "SVM"],
        "Train Accuracy": [log_metrics["train_accuracy"],knn_metrics["train_accuracy"],rf_metrics["train_accuracy"],svm_metrics["train_accuracy"]],
        "Test Accuracy": [log_metrics["test_accuracy"], knn_metrics["test_accuracy"], rf_metrics["test_accuracy"],svm_metrics["test_accuracy"]]
    })

    st.dataframe(df_acc)

    if prob_log > 0.5:
        st.success("Startup is likely to succeed!")
    else :
        st.error("Startup is likely to fail. Consider revising your strategy.")

    col1,col2 = st.columns(2)

    ## -------left:prediction
    with col1: 

        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = prob_log*100,
            title = {'text': "Success Probability"},
            gauge = {'axis': {'range': [0,100]}}
        ))
        st.plotly_chart(fig)

    

    features = [
                "Experience","Team","Funding","Market",
                "Innovation","Marketing","Competition",
                "Industry","Education","Stage","Revenue"
            ]
    importance = log_model.coef_[0] ##Gets weights (coefficients) from Logistic Regression

    df_imp = pd.DataFrame({
                "Feature": features,
                "Importance": importance
            })

    with col2:
        st.subheader("Feature Importance")
        fig2 = px.bar(df_imp, x="Importance", y="Feature", orientation="h")
        st.plotly_chart(fig2)
