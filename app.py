import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('titanic_model.pkl')

# Title of the Streamlit app
st.title('Titanic Survival Prediction')

# User input for model features
st.write('Please provide the following details:')

pclass = st.selectbox('Passenger Class', [1, 2, 3])
sex = st.selectbox('Sex', ['male', 'female'])
age = st.slider('Age', 0, 80, 25)
fare = st.slider("Ticket Fare", 0.0, 10000.0, 50.0)
embarked = st.selectbox('Port of Embarkation', ['Southampton', 'Cherbourg', 'Queenstown'])

# Convert categorical input into numerical format
sex = 0 if sex == 'Male' else 1
embarked = {'Southampton': 0, 'Cherbourg': 1, 'Queenstown': 2}[embarked]

# Prepare the feature vector for prediction
features = pd.DataFrame({
 'pclass': [pclass],
 'sex': [sex],
 'age': [age],
 'fare': [fare],
 'embarked': [embarked]
})

# Make the prediction
if st.button('Predict'):
    prediction = model.predict(features)[0]
    result = 'Survived' if prediction == 1 else 'Not Survived'
    st.write(f'The model predicts: {result}')

# Optionally show the input features
st.write('Input features:')
st.write(features)

