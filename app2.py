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

# Streamlit App for Diabetes Prediction
if selected == 'Diabetes Prediction':

    # Page title
    st.title('Diabetes Prediction using ML')

    # User inputs
    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox('Gender', ['Male', 'Female', 'Other'])
        age = st.number_input('Age', min_value=0, max_value=120, step=1, value=25)

    with col2:
        hypertension = st.selectbox('Hypertension', [0, 1], help="0: No, 1: Yes")
        heart_disease = st.selectbox('Heart Disease', [0, 1], help="0: No, 1: Yes")

    with col3:
        smoking_history = st.selectbox(
            'Smoking History',
            ['non-smoker', 'current', 'past_smoker']
        )
        bmi = st.number_input('BMI', min_value=10.0, max_value=50.0, step=0.1, value=25.0)

    col4, col5 = st.columns(2)

    with col4:
        hba1c_level = st.number_input('HbA1c Level', min_value=3.0, max_value=15.0, step=0.1, value=5.0)

    with col5:
        blood_glucose_level = st.number_input('Blood Glucose Level', min_value=50, max_value=300, step=1, value=120)

    # Preprocessing
    # Encode gender
    gender_encoded = 1 if gender == 'Male' else (0 if gender == 'Female' else 2)

    # Encode smoking history
    smoking_history_encoded = {
        'non-smoker': 0,
        'current': 1,
        'past_smoker': 2
    }[smoking_history]

    # Scale numerical features to match model training
    scaler_mean = {'age': 50.0, 'bmi': 25.0, 'HbA1c_level': 5.5, 'blood_glucose_level': 120.0}
    scaler_std = {'age': 20.0, 'bmi': 5.0, 'HbA1c_level': 1.0, 'blood_glucose_level': 30.0}

    def scale(value, mean, std):
        return (value - mean) / std

    age_scaled = scale(age, scaler_mean['age'], scaler_std['age'])
    bmi_scaled = scale(bmi, scaler_mean['bmi'], scaler_std['bmi'])
    hba1c_scaled = scale(hba1c_level, scaler_mean['HbA1c_level'], scaler_std['HbA1c_level'])
    glucose_scaled = scale(blood_glucose_level, scaler_mean['blood_glucose_level'], scaler_std['blood_glucose_level'])

    # Input features array
    input_data = np.array([
        gender_encoded, age_scaled, hypertension, heart_disease,
        smoking_history_encoded, bmi_scaled, hba1c_scaled, glucose_scaled
    ]).reshape(1, -1)

    # Prediction
    if st.button('Predict'):
        prediction = diabetes_model.predict(input_data)
        prediction_label = 'Diabetes Detected' if prediction[0] == 1 else 'No Diabetes Detected'

        # Display result
        st.success(prediction_label)


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


if selected == "Parkinson's Prediction":

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
