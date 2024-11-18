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


# Heart Disease Prediction Page
if selected == 'Heart Disease Prediction':

    # page title
    st.title('Heart Disease Prediction using ML')

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.text_input('Age')

    with col2:
        sex = st.text_input('Sex')

    with col3:
        cp = st.text_input('Chest Pain types')

    with col1:
        trestbps = st.text_input('Resting Blood Pressure')

    with col2:
        chol = st.text_input('Serum Cholestoral in mg/dl')

    with col3:
        fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl')

    with col1:
        restecg = st.text_input('Resting Electrocardiographic results')

    with col2:
        thalach = st.text_input('Maximum Heart Rate achieved')

    with col3:
        exang = st.text_input('Exercise Induced Angina')

    with col1:
        oldpeak = st.text_input('ST depression induced by exercise')

    with col2:
        slope = st.text_input('Slope of the peak exercise ST segment')

    with col3:
        ca = st.text_input('Major vessels colored by flourosopy')

    with col1:
        thal = st.text_input('thal: 0 = normal; 1 = fixed defect; 2 = reversable defect')

    # code for Prediction
    heart_diagnosis = ''

    # creating a button for Prediction

    if st.button('Heart Disease Test Result'):

        user_input = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]

        user_input = [float(x) for x in user_input]

        heart_prediction = heart_disease_model.predict([user_input])

        if heart_prediction[0] == 1:
            heart_diagnosis = 'The person is having heart disease'
        else:
            heart_diagnosis = 'The person does not have any heart disease'

    st.success(heart_diagnosis)

# Parkinson's Prediction Page
if selected == "Parkinsons Prediction":

    # page title
    st.title("Parkinson's Disease Prediction using ML")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        fo = st.text_input('MDVP:Fo(Hz)')

    with col2:
        fhi = st.text_input('MDVP:Fhi(Hz)')

    with col3:
        flo = st.text_input('MDVP:Flo(Hz)')

    with col4:
        Jitter_percent = st.text_input('MDVP:Jitter(%)')

    with col5:
        Jitter_Abs = st.text_input('MDVP:Jitter(Abs)')

    with col1:
        RAP = st.text_input('MDVP:RAP')

    with col2:
        PPQ = st.text_input('MDVP:PPQ')

    with col3:
        DDP = st.text_input('Jitter:DDP')

    with col4:
        Shimmer = st.text_input('MDVP:Shimmer')

    with col5:
        Shimmer_dB = st.text_input('MDVP:Shimmer(dB)')

    with col1:
        APQ3 = st.text_input('Shimmer:APQ3')

    with col2:
        APQ5 = st.text_input('Shimmer:APQ5')

    with col3:
        APQ = st.text_input('MDVP:APQ')

    with col4:
        DDA = st.text_input('Shimmer:DDA')

    with col5:
        NHR = st.text_input('NHR')

    with col1:
        HNR = st.text_input('HNR')

    with col2:
        RPDE = st.text_input('RPDE')

    with col3:
        DFA = st.text_input('DFA')

    with col4:
        spread1 = st.text_input('spread1')

    with col5:
        spread2 = st.text_input('spread2')

    with col1:
        D2 = st.text_input('D2')

    with col2:
        PPE = st.text_input('PPE')

    # code for Prediction
    parkinsons_diagnosis = ''

    # creating a button for Prediction    
    if st.button("Parkinson's Test Result"):

        user_input = [fo, fhi, flo, Jitter_percent, Jitter_Abs,
                      RAP, PPQ, DDP,Shimmer, Shimmer_dB, APQ3, APQ5,
                      APQ, DDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE]

        user_input = [float(x) for x in user_input]

        parkinsons_prediction = parkinsons_model.predict([user_input])

        if parkinsons_prediction[0] == 1:
            parkinsons_diagnosis = "The person has Parkinson's disease"
        else:
            parkinsons_diagnosis = "The person does not have Parkinson's disease"

    st.success(parkinsons_diagnosis)
