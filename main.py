import streamlit as st
import pandas as pd
import joblib

st.write("""
    ### Heart Disease Classification App
    ### This app predicts the occurance of a heart disease based on several medical conditions
""")

st.image('heart-disease.jpeg')

st.sidebar.header('Input patient information')


def gather_patient_data():
    age = st.sidebar.number_input("Enter the patient age: ", step=1)
    sex_select = st.sidebar.selectbox('Sex', ('Male', 'Female'))
    sex = 1 if sex_select == 'Male' else 0

    cp_select = st.sidebar.selectbox('Chest pain type', ('asymptomatic', 'atypical angina',
                                                         'non-anginal pain', 'typical angina'))
    if cp_select == 'asymptomatic':
        cp = 0
    elif cp_select == 'atypical angina':
        cp = 1
    elif cp_select == 'non-anginal pain':
        cp = 2
    else:
        cp = 3

    tres = st.sidebar.number_input('Resting blood pressure: ', step=1)
    chol = st.sidebar.number_input('Serum cholesterol in mg/dl: ', step=1)
    fbs_select = st.sidebar.selectbox('Is your fasting blood sugar greater than 120 mg/dl?', ('Yes', 'No'))
    fbs = 1 if fbs_select == 'Yes' else 0

    res_select = st.sidebar.selectbox('Resting electrocardiographic results', ('left ventricular hypertrophy',
                                                                               'normal',
                                                                               'has ST-T wave abnormality'))
    if res_select == 'left ventricular hypertrophy':
        res = 0
    elif res_select == 'normal':
        res = 1
    else:
        res = 2

    tha = st.sidebar.number_input('Maximum heart rate achieved of the patient: ', step=1)
    exa_select = st.sidebar.selectbox('Does the patient have exercise induced angina?', ('Yes', 'No'))
    exa = 1 if exa_select == 'Yes' else 0

    old = st.sidebar.number_input('ST depression induced by exercise relative to rest: ', step=0.1)
    slope_select = st.sidebar.selectbox('The slope of the peak exercise ST segment',
                                        ('Downsloping', 'Flat', 'Upsloping'))
    if slope_select == 'Downsloping':
        slope = 0
    elif slope_select == 'Flat':
        slope = 1
    else:
        slope = 2

    ca = st.sidebar.selectbox('Number of major blood vessels', (0, 1, 2, 3))
    thal_select = st.sidebar.selectbox('Thalassemia condition of the patient',
                                       ('fixed defect', 'normal blood flow', 'reversible defect'))
    if thal_select == 'fixed defect':
        thal = 1
    elif thal_select == 'normal blood flow':
        thal = 2
    else:
        thal = 3

    data = {
        'age': age,
        'sex': sex,
        'cp': cp,
        'trestbps': tres,
        'chol': chol,
        'fbs': fbs,
        'restecg': res,
        'thalach': tha,
        'exang': exa,
        'oldpeak': old,
        'slope': slope,
        'ca': ca,
        'thal': thal
    }

    features = pd.DataFrame(data, index=[0])
    return features


input_df = gather_patient_data()

model = joblib.load('model/random-forest-heart-disease-classifier.joblib')

mean = input_df.mean()
std = input_df.std()

pred = model.predict(input_df)
if st.button("Predict"):
    if pred[0] == 0:
        st.error('Warning! You have high risk of getting a heart attack!')
    else:
        st.success('You have lower risk of getting a heart disease!')

