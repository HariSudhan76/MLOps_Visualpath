import streamlit as st
import joblib
import pandas as pd 
import numpy as np 
import seaborn as sns

st.set_page_config(page_title="Titanic Survival Predictor", layout="centered")

st.title("Titanic Survival Prediction")
st.write("Enter passenger details and click **Predict Survival**"
         "Model trained on seaborn's Titanic dataset with RandomForest classifier")

#Load model
@st.cache_resource
def load_model():
    return joblib.load("artifacts/titanic_model.pkl")
model = load_model()

# Input fields

st.subheader("Passenger details")
pclass = st.selectbox("Passenger Class (1=First, 2=Second, 3=Third)",[1,2,3],index=2)
gender = st.selectbox("Gender",["Male","Female"])
age = st.slider("Age",0,100,30)
sibsp = st.number_input("Siblings / Spouse aboard (SibSP)",min_value=0,max_value=10,value=0)
parch = st.number_input("Parent / Children aboard (Parch)",min_value=0,max_value=10,value=0)
fare = st.number_input("Ticket Fare",min_value=0.0,max_value=1000.0, value=32.2,format="%.2f")
embarked = st.selectbox("Port of Embarkation",["S","C","Q"],index=0)


if st.button("Predict Survival"):
    input_df = pd.DataFrame([{
        "pclass":pclass,
        "sex": gender,
        "age": age,
        "sibsp":sibsp,
        "parch":parch,
        "fare":fare,
        "embarked": embarked
    }])

    pred = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][1]

    if pred == 1:
        st.success(f"Likely to survive (probability = {proba:.2f})")
    else:
        st.error(f"Not likely to survive (probability = {proba:.2f})")


##Optional show sample data and metric
if st.checkbox("show training sample and metrics"):
    st.subheader("Sample from dataset")
    df = sns.load_dataset("titanic")[["pclass","sex","age","sibsp","parch","fare","embarked","survived"]].head(10)
    st.write(df)

    st.subheader("Saved metrics")

    import json,os 
    if os.path.exists("artifacts\metric.json"):
        with open("artifacts\metric.json") as f:
            metrics = json.load(f)
        st.json(metrics)
    
    else:
        st.info("No Metrics found")
