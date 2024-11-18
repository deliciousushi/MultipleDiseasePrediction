import os
import pickle
import numpy as np
import streamlit as st
from streamlit_option_menu import option_menu

# Set page configuration
st.set_page_config(page_title="Health Assistant",
                   layout="wide",
                   page_icon="🧑‍⚕️")

    
# getting the working directory of the main.py
working_dir = os.path.dirname(os.path.abspath(__file__))

# loading the saved models

diabetes_model = pickle.load(open(f'{working_dir}/diabetes_model.sav', 'rb'))

heart_disease_model = pickle.load(open(f'{working_dir}/heart_disease_model.sav', 'rb'))

parkinsons_model = pickle.load(open(f'{working_dir}/parkinsons_model.sav', 'rb'))

# sidebar for navigation
with st.sidebar:
    selected = option_menu('Multiple Disease Prediction System',

                           ['Diabetes Prediction',
                            'Heart Disease Prediction',
                            'Parkinsons Prediction'],
                           menu_icon='hospital-fill',
                           icons=['activity', 'heart', 'person'],
                           default_index=0)


# Diabetes Prediction Page
if selected == 'Diabetes Prediction':
    # Page title
    st.title('Diabetes Prediction using ML')

    # User inputs
    col1, col2, col3 = st.columns(3)

    with col1:
        Pregnancies = st.number_input('Pregnancies: Number of times pregnant', min_value=0, step=1)

    with col2:
        Glucose = st.number_input('Glucose: Plasma glucose concentration (mg/dL)', min_value=0)

    with col3:
        BloodPressure = st.number_input('Blood Pressure: Diastolic (mm Hg)', min_value=0)

    with col1:
        SkinThickness = st.number_input('Skin Thickness: Triceps fold thickness (mm)', min_value=0)

    with col2:
        Insulin = st.number_input('Insulin: 2-Hour serum insulin (mu U/ml)', min_value=0)

    with col3:
        BMI = st.number_input('BMI: Body Mass Index (kg/m²)', min_value=0.0, step=0.1)

    with col1:
        DiabetesPedigreeFunction = st.number_input('Diabetes Pedigree Function', min_value=0.0, step=0.01)

    with col2:
        Age = st.number_input('Age: Years', min_value=0, step=1)

    # Diabetes Prediction
    diab_diagnosis = ''

    if st.button('Diabetes Test Result'):
        # Preprocessing user input
        user_input = [
            Pregnancies, Glucose, BloodPressure, SkinThickness,
            Insulin, BMI, DiabetesPedigreeFunction, Age
        ]

        # Ensure all inputs are valid and scale/normalize if necessary
        # Assuming the model was trained on normalized data:
        scaler_mean = {
            'Pregnancies': 3.8, 'Glucose': 120.0, 'BloodPressure': 69.1,
            'SkinThickness': 20.5, 'Insulin': 80.0, 'BMI': 32.0,
            'DiabetesPedigreeFunction': 0.47, 'Age': 33.0
        }
        scaler_std = {
            'Pregnancies': 3.3, 'Glucose': 31.0, 'BloodPressure': 19.3,
            'SkinThickness': 15.0, 'Insulin': 115.2, 'BMI': 7.9,
            'DiabetesPedigreeFunction': 0.33, 'Age': 11.8
        }

        # Scale the inputs
        user_input_scaled = [
            (user_input[i] - scaler_mean[key]) / scaler_std[key]
            for i, key in enumerate(scaler_mean)
        ]

        # Make prediction
        diab_prediction = diabetes_model.predict([user_input_scaled])

        if diab_prediction[0] == 1:
            diab_diagnosis = 'The person is diabetic'
        else:
            diab_diagnosis = 'The person is not diabetic'

    st.success(diab_diagnosis)


# Heart Disease Prediction Page
if selected == 'Heart Disease Prediction':
    # Page title
    st.title('Heart Disease Prediction using ML')

    # User inputs
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input('Age', min_value=0, max_value=120, step=1)

    with col2:
        sex = st.selectbox('Sex', ['Male', 'Female'])

    with col3:
        cp = st.selectbox('Chest Pain Types', ['typical angina', 'atypical angina', 'non-anginal pain', 'asymptomatic'])

    with col1:
        trestbps = st.number_input('Resting Blood Pressure (mm Hg)', min_value=0)

    with col2:
        chol = st.number_input('Serum Cholestoral in mg/dl', min_value=0)

    with col3:
        fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', ['True', 'False'])

    with col1:
        restecg = st.selectbox('Resting Electrocardiographic Results', ['normal', 'st-T wave abnormality', 'left ventricular hypertrophy'])

    with col2:
        thalach = st.number_input('Maximum Heart Rate Achieved', min_value=0)

    with col3:
        exang = st.selectbox('Exercise Induced Angina', ['True', 'False'])

    with col1:
        oldpeak = st.number_input('ST Depression Induced by Exercise')

    with col2:
        slope = st.selectbox('Slope of the Peak Exercise ST Segment', ['upsloping', 'flat', 'downsloping'])

    with col3:
        ca = st.number_input('Number of Major Vessels Colored by Flourosopy', min_value=0, max_value=3)

    with col1:
        thal = st.selectbox('Thalassemia', ['normal', 'fixed defect', 'reversible defect'])

    # Heart Disease Prediction
    heart_diagnosis = ''

    if st.button('Heart Disease Test Result'):
        # Preprocessing
        sex_encoded = 1 if sex == 'Male' else 0
        fbs_encoded = 1 if fbs == 'True' else 0
        exang_encoded = 1 if exang == 'True' else 0
        restecg_encoded = {'normal': 0, 'st-T wave abnormality': 1, 'left ventricular hypertrophy': 2}[restecg]
        thal_encoded = {'normal': 0, 'fixed defect': 1, 'reversible defect': 2}[thal]
        cp_encoded = {'typical angina': 0, 'atypical angina': 1, 'non-anginal pain': 2, 'asymptomatic': 3}[cp]
        slope_encoded = {'upsloping': 0, 'flat': 1, 'downsloping': 2}[slope]

        user_input = [age, sex_encoded, cp_encoded, trestbps, chol, fbs_encoded, restecg_encoded, thalach, exang_encoded, oldpeak, slope_encoded, ca, thal_encoded]

        heart_prediction = heart_disease_model.predict([user_input])

        if heart_prediction[0] == 1:
            heart_diagnosis = 'The person has heart disease'
        else:
            heart_diagnosis = 'The person does not have heart disease'

    st.success(heart_diagnosis)


if selected == "Parkinsons Prediction":

    # Page title
    st.title("Parkinson's Disease Prediction using ML")

    # Create columns for user input
    col1, col2, col3, col4, col5 = st.columns(5)

    # User input fields for various features related to Parkinson's disease
    with col1:
        fo = st.number_input('MDVP:Fo(Hz)', min_value=0.0, step=0.1)

    with col2:
        fhi = st.number_input('MDVP:Fhi(Hz)', min_value=0.0, step=0.1)

    with col3:
        flo = st.number_input('MDVP:Flo(Hz)', min_value=0.0, step=0.1)

    with col4:
        Jitter_percent = st.number_input('MDVP:Jitter(%)', min_value=0.0, step=0.1)

    with col5:
        Jitter_Abs = st.number_input('MDVP:Jitter(Abs)', min_value=0.0, step=0.1)

    with col1:
        RAP = st.number_input('MDVP:RAP', min_value=0.0, step=0.1)

    with col2:
        PPQ = st.number_input('MDVP:PPQ', min_value=0.0, step=0.1)

    with col3:
        DDP = st.number_input('Jitter:DDP', min_value=0.0, step=0.1)

    with col4:
        Shimmer = st.number_input('MDVP:Shimmer', min_value=0.0, step=0.1)

    with col5:
        Shimmer_dB = st.number_input('MDVP:Shimmer(dB)', min_value=0.0, step=0.1)

    with col1:
        APQ3 = st.number_input('Shimmer:APQ3', min_value=0.0, step=0.1)

    with col2:
        APQ5 = st.number_input('Shimmer:APQ5', min_value=0.0, step=0.1)

    with col3:
        APQ = st.number_input('MDVP:APQ', min_value=0.0, step=0.1)

    with col4:
        DDA = st.number_input('Shimmer:DDA', min_value=0.0, step=0.1)

    with col5:
        NHR = st.number_input('NHR', min_value=0.0, step=0.1)

    with col1:
        HNR = st.number_input('HNR', min_value=0.0, step=0.1)

    with col2:
        RPDE = st.number_input('RPDE', min_value=0.0, step=0.1)

    with col3:
        DFA = st.number_input('DFA', min_value=0.0, step=0.1)

    with col4:
        spread1 = st.number_input('spread1', min_value=0.0, step=0.1)

    with col5:
        spread2 = st.number_input('spread2', min_value=0.0, step=0.1)

    with col1:
        D2 = st.number_input('D2', min_value=0.0, step=0.1)

    with col2:
        PPE = st.number_input('PPE', min_value=0.0, step=0.1)

    # Code for Prediction
    parkinsons_diagnosis = ''

    # Button for Prediction
    if st.button("Parkinson's Test Result"):

        # Collect user inputs into a list
        user_input = [
            fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ, DDP, Shimmer,
            Shimmer_dB, APQ3, APQ5, APQ, DDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE
        ]

        # Convert inputs to float for prediction
        user_input = [float(x) for x in user_input]

        # Make prediction using the Parkinson's disease model
        parkinsons_prediction = parkinsons_model.predict([user_input])

        # Diagnosis based on the prediction
        if parkinsons_prediction[0] == 1:
            parkinsons_diagnosis = "The person has Parkinson's disease"
        else:
            parkinsons_diagnosis = "The person does not have Parkinson's disease"

    # Display the result
    st.success(parkinsons_diagnosis)
