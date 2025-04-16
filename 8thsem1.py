import os
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
import streamlit as st
from streamlit_option_menu import option_menu
from tensorflow import keras

# Page config
st.set_page_config(page_title="Health Assistant", layout="wide", page_icon="👩‍⚕️")

# Working directory
working_dir = os.path.dirname(os.path.abspath(__file__))

# Load models
diabetes_model = pickle.load(open(f'{working_dir}/diabetes_model.sav', 'rb'))
heart_disease_model = pickle.load(open(f'{working_dir}/heart_disease_model.sav', 'rb'))
parkinsons_model = pickle.load(open(f'{working_dir}/parkinsons_model.sav', 'rb'))
with open(f"{working_dir}/kidney_model (2).sav", "rb") as f:
    kidney_model = pickle.load(f)

# Load doctor data
doctor_file = f'{working_dir}/doctors_list (1).csv'
doctor_data = pd.read_csv(doctor_file) if os.path.exists(doctor_file) else pd.DataFrame(columns=["Doctor Name", "Specialty", "Location", "Contact", "Availability"])

# Create appointment log file if missing
appointments_file = f"{working_dir}/appointments.csv"
if not os.path.exists(appointments_file):
    with open(appointments_file, "w") as f:
        f.write("Patient Name,Age,Contact,Doctor Name,Specialization,Date,Time\n")

# Ensure session state
if "doctor_data" not in st.session_state:
    st.session_state["doctor_data"] = doctor_data
if "selected_specialty" not in st.session_state:
    st.session_state["selected_specialty"] = None
if "appointment_details" not in st.session_state:
    st.session_state["appointment_details"] = {}

# Time conversion helpers
def convert_to_24hr(time_str):
    time_str = time_str.strip().lower()
    if "am" in time_str:
        hour = int(time_str.replace("am", "").strip())
        return 0 if hour == 12 else hour
    elif "pm" in time_str:
        hour = int(time_str.replace("pm", "").strip())
        return hour if hour == 12 else hour + 12
    return None

def extract_availability(availability_str):
    try:
        days_part, time_part = availability_str.split(" ")
        available_days = days_part.split("-")
        start_time, end_time = time_part.split("-")
        return available_days, convert_to_24hr(start_time), convert_to_24hr(end_time)
    except:
        return None, None, None

def is_available_on_date(date, days, start_hr, end_hr):
    day = date.strftime('%a')
    hour = date.hour
    return (day in days) and (start_hr <= hour < end_hr)

def save_appointment(name, age, contact, doctor, specialty, date, time):
    with open(appointments_file, "a") as f:
        f.write(f"{name},{age},{contact},{doctor},{specialty},{date},{time}\n")

def show_patient_details_form():
    st.subheader("Enter Patient Details")

    with st.form("patient_form"):
        name = st.text_input("Full Name")
        age = st.number_input("Age", min_value=0, step=1)
        contact = st.text_input("Contact Number")
        confirm = st.form_submit_button("Confirm Appointment")

        if confirm:
            details = st.session_state["appointment_details"]
            save_appointment(name, age, contact, details["doctor"], details["specialty"], details["date"], details["time"])
            st.success(f"✅ Appointment confirmed with Dr. {details['doctor']} on {details['date']} at {details['time']}.")
            st.session_state["appointment_details"] = {}  # reset

def show_doctor_booking():
    st.subheader("Book an Appointment")

    specialty = st.session_state.get("selected_specialty")
    doctor_data = st.session_state.get("doctor_data")

    if not specialty or doctor_data.empty:
        st.warning("No specialty selected or doctor data missing.")
        return

    available_doctors = doctor_data[doctor_data["Specialty"] == specialty]

    if available_doctors.empty:
        st.warning("No doctors available for this specialty.")
        return

    for idx, row in available_doctors.iterrows():
        doctor = row["Doctor Name"]
        location = row["Location"]
        contact = row["Contact"]
        avail_str = row["Availability"]

        st.write(f"**{doctor}** - {location}")
        st.write(f"📞 Contact: {contact}")

        days, start_hr, end_hr = extract_availability(avail_str)

        if not days:
            st.error(f"Invalid availability format for {doctor}.")
            continue

        st.write(f"🗓️ **Days:** {', '.join(days)}")
        st.write(f"⏰ **Time:** {start_hr}:00 - {end_hr}:00")

        with st.form(key=f"form_{idx}"):
            appt_date = st.date_input("Select date", min_value=datetime.today().date(), key=f"date_{idx}")
            appt_time = st.time_input("Select time", key=f"time_{idx}")
            submitted = st.form_submit_button("Book Appointment")

            if submitted:
                dt_combined = datetime.combine(appt_date, appt_time)
                if is_available_on_date(dt_combined, days, start_hr, end_hr):
                    st.session_state["appointment_details"] = {
                        "doctor": doctor,
                        "specialty": specialty,
                        "date": appt_date,
                        "time": appt_time.strftime('%H:%M')
                    }
                    show_patient_details_form()
                else:
                    st.error("Doctor not available on selected date/time")

with st.sidebar:
    selected = option_menu(
        'Multiple Disease Prediction System',
        ['Diabetes Prediction', 'Heart Disease Prediction', "Parkinson's Prediction", "Kidney Disease Prediction"],
        menu_icon='hospital-fill',
        icons=['activity', 'heart', 'person', 'droplet'],
        default_index=0
    )


import numpy as np

# Kidney Disease Prediction
if selected == "Kidney Disease Prediction":
    st.title("Kidney Disease Prediction using ML")
    age = st.number_input('Age (years)', min_value=0, step=1)
    bp = st.number_input('Blood Pressure (mm Hg)', min_value=0)
    sg = st.number_input('Specific Gravity', min_value=1.000, max_value=1.030, step=0.001, format="%.3f")
    al = st.number_input('Albumin (mg/dL)', min_value=0, step=1)
    su = st.number_input('Sugar (mg/dL)', min_value=0, step=1)
    rbc = st.selectbox('Red Blood Cells', ['Normal', 'Abnormal'])
    pc = st.selectbox('Pus Cell', ['Normal', 'Abnormal'])
    pcc = st.selectbox('Pus Cell Clumps', ['Present', 'Not Present'])
    ba = st.selectbox('Bacteria', ['Present', 'Not Present'])
    bgr = st.number_input('Blood Glucose Random (mg/dL)', min_value=0)
    bu = st.number_input('Blood Urea (mg/dL)', min_value=0)
    sc = st.number_input('Serum Creatinine (mg/dL)', min_value=0.0, step=0.1)
    sod = st.number_input('Sodium (mEq/L)', min_value=0)
    pot = st.number_input('Potassium (mEq/L)', min_value=0.0, step=0.1)
    hemo = st.number_input('Hemoglobin (g/dL)', min_value=0.0, step=0.1)
    pcv = st.number_input('Packed Cell Volume (%)', min_value=0)
    wc = st.number_input('White Blood Cell Count (cells/cu mm)', min_value=0)
    rc = st.number_input('Red Blood Cell Count (million cells/cu mm)', min_value=0.0, step=0.1)
    htn = st.selectbox('Hypertension', ['Yes', 'No'])
    dm = st.selectbox('Diabetes Mellitus', ['Yes', 'No'])
    cad = st.selectbox('Coronary Artery Disease', ['Yes', 'No'])
    appet = st.selectbox('Appetite', ['Good', 'Poor'])
    pe = st.selectbox('Pedal Edema', ['Yes', 'No'])
    ane = st.selectbox('Anemia', ['Yes', 'No'])
    
    if st.button("Kidney Disease Test Result"):
        if kidney_model:
            user_input = np.array([[age, bp, sg, al, su, 1 if rbc == 'Abnormal' else 0, 1 if pc == 'Abnormal' else 0,
                          1 if pcc == 'Present' else 0, 1 if ba == 'Present' else 0, bgr, bu, sc, sod, pot, hemo, pcv, wc, rc,
                          1 if htn == 'Yes' else 0, 1 if dm == 'Yes' else 0, 1 if cad == 'Yes' else 0,
                          1 if appet == 'Poor' else 0, 1 if pe == 'Yes' else 0, 1 if ane == 'Yes' else 0]])
            prediction = kidney_model.predict(user_input)[0]
            if prediction == 1:
                st.error("The person has kidney disease.")
                st.session_state["selected_specialty"] = "Nephrologist"
                show_doctor_booking()

            else:
                st.success("The person does not have kidney disease.")
        else:
            st.error("Kidney Disease Prediction model is not available.")


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

    if st.button('Diabetes Test Result'):  # Fixed indentation
        user_input = np.array([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]).reshape(1, -1)
        diab_prediction = diabetes_model.predict(user_input)
        confidence = diabetes_model.predict_proba(user_input)[0][1] * 100

        if diab_prediction[0] == 1:
            st.error(f'The person is diabetic (Confidence: {confidence:.2f}%)')
            show_doctor_booking("Diabetes",doctor_data)
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
            show_doctor_booking("Cardiology",doctor_data)
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
            show_doctor_booking("Neurology",doctor_data)
        else:
            st.success(f"The person does not have Parkinson's disease (Confidence: {100-confidence:.2f}%)")
