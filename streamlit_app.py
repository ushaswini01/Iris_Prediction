import streamlit as st
import joblib
import pandas as pd

# Load the trained model
model = joblib.load('model.joblib')

# Streamlit app title
st.title("Iris Species Prediction")

# Input fields for user to enter data
sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, max_value=10.0, value=5.0)
sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, max_value=10.0, value=3.0)
petal_length = st.number_input("Petal Length (cm)", min_value=0.0, max_value=10.0, value=1.5)
petal_width = st.number_input("Petal Width (cm)", min_value=0.0, max_value=10.0, value=0.2)

# Make a prediction when the user clicks the button
if st.button("Predict"):
    input_data = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]],
                              columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
    prediction = model.predict(input_data)[0]
    st.write(f"The predicted species is: {prediction}")
