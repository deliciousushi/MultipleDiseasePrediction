import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu

# Set page configuration
st.set_page_config(page_title="Health Assistant", layout="wide", page_icon="🧑‍⚕️")

# Getting the working directory
working_dir = os.path.dirname(os.path.abspath(__file__))

# Load trained models
diabetes_model = pickle.load(open(f'{working_dir}/diabetes_model.sav', 'rb'))
heart_disease_model = pickle.load(open(f'{working_dir}/heart_disease_model.sav', 'rb'))
parkinsons_model = pickle.load(open(f'{working_dir}/parkinsons_model.sav', 'rb'))

# Load doctor data
doctor_file = f'{working_dir}/doctors_list.csv'
if os.path.exists(doctor_file):
    doctor_data = pd.read_csv(doctor_file)
else:
    doctor_data = pd.DataFrame(columns=["Doctor Name", "Specialty", "Location", "Contact", "Availability"])

# Sidebar navigation
with st.sidebar:
    selected = option_menu('Multiple Disease Prediction System',
                           ['Diabetes Prediction', 'Heart Disease Prediction', "Parkinson's Prediction"],
                           menu_icon='hospital-fill',
                           icons=['activity', 'heart', 'person'],
                           default_index=0)

# Function to display doctor booking option
def show_doctor_booking(specialty):
    st.subheader("Book an Appointment")
    available_doctors = doctor_data[doctor_data['Specialty'] == specialty]
    if not available_doctors.empty:
        for _, row in available_doctors.iterrows():
            st.write(f"**{row['Doctor Name']}** - {row['Location']}\n")
            st.write(f"📞 Contact: {row['Contact']}\n")
            st.write(f"🕒 Availability: {row['Availability']}\n")
            st.button(f"Book Appointment with {row['Doctor Name']}")
    else:
        st.warning("No available doctors for this specialty.")

# Diabetes Prediction
if selected == 'Diabetes Prediction':
    st.title('Diabetes Prediction using ML')
    Pregnancies = st.number_input('Pregnancies', min_value=0, step=1)
    Glucose = st.number_input('Glucose Level', min_value=0)
    BloodPressure = st.number_input('Blood Pressure', min_value=0)
    SkinThickness = st.number_input('Skin Thickness', min_value=0)
    Insulin = st.number_input('Insulin Level', min_value=0)
    BMI = st.number_input('BMI', min_value=0.0, step=0.1)
    DiabetesPedigreeFunction = st.number_input('Diabetes Pedigree Function', min_value=0.0, step=0.01)
    Age = st.number_input('Age', min_value=0, step=1)
    
    if st.button('Diabetes Test Result'):
        user_input = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]
        prediction = diabetes_model.predict([user_input])[0]
        if prediction == 1:
            st.error('The person is diabetic.')
            show_doctor_booking("Diabetologist")
        else:
            st.success('The person is not diabetic.')

# Heart Disease Prediction
if selected == 'Heart Disease Prediction':
    st.title('Heart Disease Prediction using ML')
    age = st.number_input('Age', min_value=0, max_value=120, step=1)
    sex = st.selectbox('Sex', ['Male', 'Female'])
    cp = st.selectbox('Chest Pain Type', ['typical angina', 'atypical angina', 'non-anginal pain', 'asymptomatic'])
    trestbps = st.number_input('Resting Blood Pressure', min_value=0)
    chol = st.number_input('Serum Cholesterol', min_value=0)
    thalach = st.number_input('Maximum Heart Rate', min_value=0)
    oldpeak = st.number_input('ST Depression', step=0.1)
    
    if st.button('Heart Disease Test Result'):
        user_input = [age, 1 if sex == 'Male' else 0, trestbps, chol, thalach, oldpeak]
        prediction = heart_disease_model.predict([user_input])[0]
        if prediction == 1:
            st.error('The person has heart disease.')
            show_doctor_booking("Cardiologist")
        else:
            st.success('The person does not have heart disease.')

# Parkinson's Prediction
if selected == "Parkinson's Prediction":
    st.title("Parkinson's Disease Prediction using ML")
    fo = st.number_input('MDVP:Fo(Hz)', min_value=0.0, step=0.1)
    fhi = st.number_input('MDVP:Fhi(Hz)', min_value=0.0, step=0.1)
    flo = st.number_input('MDVP:Flo(Hz)', min_value=0.0, step=0.1)
    jitter_percent = st.number_input('MDVP:Jitter(%)', min_value=0.0, step=0.1)
    shimmer = st.number_input('MDVP:Shimmer', min_value=0.0, step=0.1)
    hnr = st.number_input('HNR', min_value=0.0, step=0.1)
    spread1 = st.number_input('Spread1', min_value=0.0, step=0.1)
    spread2 = st.number_input('Spread2', min_value=0.0, step=0.1)
    dfa = st.number_input('DFA', min_value=0.0, step=0.1)
    d2 = st.number_input('D2', min_value=0.0, step=0.1)
    ppe = st.number_input('PPE', min_value=0.0, step=0.1)
    
    if st.button("Parkinson's Test Result"):
        user_input = [fo, fhi, flo, jitter_percent, shimmer, hnr, spread1, spread2, dfa, d2, ppe]
        prediction = parkinsons_model.predict([user_input])[0]
        if prediction == 1:
            st.error("The person has Parkinson's disease.")
            show_doctor_booking("Neurologist")
        else:
            st.success("The person does not have Parkinson's disease.")
