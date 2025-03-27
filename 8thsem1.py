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
