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
kidney_disease_model = pickle.load(open(f'{working_dir}/kidney_model.sav', 'rb'))

# Load doctor data
doctor_file = f'{working_dir}/doctors_list.csv'
if os.path.exists(doctor_file):
    doctor_data = pd.read_csv(doctor_file)
else:
    doctor_data = pd.DataFrame(columns=["Doctor Name", "Specialty", "Location", "Contact", "Availability"])

# Sidebar navigation
with st.sidebar:
    selected = option_menu('Multiple Disease Prediction System',
                           ['Diabetes Prediction', 'Heart Disease Prediction', "Parkinson's Prediction", "Kidney Disease Prediction"],
                           menu_icon='hospital-fill',
                           icons=['activity', 'heart', 'person', 'droplet'],
                           default_index=0)

# Function to display doctor booking option
def show_doctor_booking(specialty):
    st.subheader("Book an Appointment")
    available_doctors = doctor_data[doctor_data['Specialty'] == specialty]
    if not available_doctors.empty:
        for _, row in available_doctors.iterrows():
            st.write(f"**{row['Doctor Name']}** - {row['Location']}")
            st.write(f"📞 Contact: {row['Contact']}")
            st.write(f"🕒 Availability: {row['Availability']}")
            st.button(f"Book Appointment with {row['Doctor Name']}")
    else:
        st.warning("No available doctors for this specialty.")

# Kidney Disease Prediction
if selected == "Kidney Disease Prediction":
    st.title("Kidney Disease Prediction using ML")
    age = st.number_input('Age', min_value=0, step=1)
    bp = st.number_input('Blood Pressure', min_value=0)
    sg = st.number_input('Specific Gravity', min_value=1.000, max_value=1.030, step=0.001)
    al = st.number_input('Albumin', min_value=0, step=1)
    su = st.number_input('Sugar', min_value=0, step=1)
    rbc = st.selectbox('Red Blood Cells', ['Normal', 'Abnormal'])
    pc = st.selectbox('Pus Cell', ['Normal', 'Abnormal'])
    pcc = st.selectbox('Pus Cell Clumps', ['Present', 'Not Present'])
    ba = st.selectbox('Bacteria', ['Present', 'Not Present'])
    pcv = st.number_input('Packed Cell Volume', min_value=0)
    wc = st.number_input('White Blood Cell Count', min_value=0)
    rc = st.number_input('Red Blood Cell Count', min_value=0.0, step=0.1)
    htn = st.selectbox('Hypertension', ['Yes', 'No'])
    dm = st.selectbox('Diabetes Mellitus', ['Yes', 'No'])
    cad = st.selectbox('Coronary Artery Disease', ['Yes', 'No'])
    appet = st.selectbox('Appetite', ['Good', 'Poor'])
    pe = st.selectbox('Pedal Edema', ['Yes', 'No'])
    ane = st.selectbox('Anemia', ['Yes', 'No'])
    
    if st.button("Kidney Disease Test Result"):
        user_input = [age, bp, sg, al, su, 1 if rbc == 'Abnormal' else 0, 1 if pc == 'Abnormal' else 0,
                      1 if pcc == 'Present' else 0, 1 if ba == 'Present' else 0, pcv, wc, rc,
                      1 if htn == 'Yes' else 0, 1 if dm == 'Yes' else 0, 1 if cad == 'Yes' else 0,
                      1 if appet == 'Poor' else 0, 1 if pe == 'Yes' else 0, 1 if ane == 'Yes' else 0]
        prediction = kidney_disease_model.predict([user_input])[0]
        if prediction == 1:
            st.error("The person has kidney disease.")
            show_doctor_booking("Nephrologist")
        else:
            st.success("The person does not have kidney disease.")
if selected == 'Diabetes Prediction':
    st.title('Diabetes Prediction using ML')
    Pregnancies = st.number_input('Pregnancies', min_value=0, step=1)
    Glucose = st.number_input('Glucose Level (mg/dL)', min_value=0)
    BloodPressure = st.number_input('Blood Pressure (mm Hg)', min_value=0)
    SkinThickness = st.number_input('Skin Thickness (mm)', min_value=0)
    Insulin = st.number_input('Insulin Level', min_value=0)
    BMI = st.number_input('BMI (kg/m²)', min_value=0.0, step=0.1)
    DiabetesPedigreeFunction = st.number_input('Diabetes Pedigree Function', min_value=0.0, step=0.01)
    Age = st.number_input('Age', min_value=0, step=1)
    
    if st.button('Diabetes Test Result'):
        user_input = np.array([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]).reshape(1, -1)
        diab_prediction = diabetes_model.predict(user_input)
        confidence = diabetes_model.predict_proba(user_input)[0][1] * 100
        
        if diab_prediction[0] == 1:
            st.error(f'The person is diabetic (Confidence: {confidence:.2f}%)')
            show_doctor_booking("Diabetes")
        else:
            st.success(f'The person is not diabetic (Confidence: {100-confidence:.2f}%)')

if selected == 'Heart Disease Prediction':
    st.title('Heart Disease Prediction using ML')
    age = st.number_input('Age', min_value=0, max_value=120, step=1)
    sex = st.selectbox('Sex', ['Male', 'Female'])
    cp = st.selectbox('Chest Pain Type', ['typical angina', 'atypical angina', 'non-anginal pain', 'asymptomatic'])
    trestbps = st.number_input('Resting Blood Pressure (mm Hg)', min_value=0)
    chol = st.number_input('Cholesterol Level (mg/dL)', min_value=0)
    fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dL', ['True', 'False'])
    restecg = st.selectbox('Resting ECG', ['normal', 'st-T wave abnormality', 'left ventricular hypertrophy'])
    thalach = st.number_input('Max Heart Rate', min_value=0)
    exang = st.selectbox('Exercise Induced Angina', ['True', 'False'])
    oldpeak = st.number_input('ST Depression', step=0.1)
    slope = st.selectbox('ST Slope', ['upsloping', 'flat', 'downsloping'])
    ca = st.number_input('Major Vessels Colored', min_value=0, max_value=3)
    thal = st.selectbox('Thalassemia', ['normal', 'fixed defect', 'reversible defect'])
    
    if st.button('Heart Disease Test Result'):
        sex_encoded = 1 if sex == 'Male' else 0
        fbs_encoded = 1 if fbs == 'True' else 0
        exang_encoded = 1 if exang == 'True' else 0
        restecg_encoded = {'normal': 0, 'st-T wave abnormality': 1, 'left ventricular hypertrophy': 2}[restecg]
        thal_encoded = {'normal': 0, 'fixed defect': 1, 'reversible defect': 2}[thal]
        cp_encoded = {'typical angina': 0, 'atypical angina': 1, 'non-anginal pain': 2, 'asymptomatic': 3}[cp]
        slope_encoded = {'upsloping': 0, 'flat': 1, 'downsloping': 2}[slope]
        
        user_input = np.array([age, sex_encoded, cp_encoded, trestbps, chol, fbs_encoded, restecg_encoded, thalach, exang_encoded, oldpeak, slope_encoded, ca, thal_encoded]).reshape(1, -1)
        heart_prediction = heart_disease_model.predict(user_input)
        confidence = heart_disease_model.predict_proba(user_input)[0][1] * 100
        
        if heart_prediction[0] == 1:
            st.error(f'The person has heart disease (Confidence: {confidence:.2f}%)')
            show_doctor_booking("Cardiology")
        else:
            st.success(f'The person does not have heart disease (Confidence: {100-confidence:.2f}%)')

if selected == "Parkinsons Prediction":
    st.title("Parkinson's Disease Prediction using ML")
    features = [st.number_input(label, min_value=0.0, step=0.1) for label in ['MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'Jitter(%)', 'Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP', 'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5', 'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA', 'spread1', 'spread2', 'D2', 'PPE']]
    
    if st.button("Parkinson's Test Result"):
        user_input = np.array(features).reshape(1, -1)
        parkinsons_prediction = parkinsons_model.predict(user_input)
        confidence = parkinsons_model.predict_proba(user_input)[0][1] * 100
        
        if parkinsons_prediction[0] == 1:
            st.error(f"The person has Parkinson's disease (Confidence: {confidence:.2f}%)")
            show_doctor_booking("Neurology")
        else:
            st.success(f"The person does not have Parkinson's disease (Confidence: {100-confidence:.2f}%)")
