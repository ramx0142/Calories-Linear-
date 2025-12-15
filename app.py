import streamlit as st
import joblib
import numpy as np

# 1. Load the trained model
# Make sure 'exercise_model.pkl' is in the same folder as this app.py file
try:
    model = joblib.load('exercise_calories.pkl')
except FileNotFoundError:
    st.error("Model file not found. Please export 'exercise_model.pkl' from your notebook and place it here.")
    st.stop()

# 2. App Interface
st.set_page_config(page_title="Calorie Burn Predictor", page_icon="🔥")

st.title("🔥 Exercise Calorie Predictor")
st.write("Enter the Day and Minutes Exercised to predict calories burned.")

# 3. Input Fields
# Your model was trained on 2 features: ['day', 'minutes_exercised']

col1, col2 = st.columns(2)

with col1:
    day = st.number_input(
        "Day", 
        min_value=1, 
        max_value=365, 
        value=10, 
        step=1,
        help="The day number (e.g., Day 1, Day 2...)"
    )

with col2:
    minutes = st.number_input(
        "Minutes Exercised", 
        min_value=0, 
        max_value=300, 
        value=45, 
        step=5,
        help="Duration of exercise in minutes"
    )

# 4. Prediction Logic
if st.button("Predict Calories"):
    # The model expects a 2D array: [[day, minutes]]
    input_data = np.array([[day, minutes]])
    
    try:
        prediction = model.predict(input_data)
        result = prediction[0]
        
        st.markdown("---")
        st.subheader("Results:")
        st.success(f"You are estimated to burn **{result:.2f}** calories.")
        
    except Exception as e:
        st.error(f"An error occurred: {e}")

# Footer
st.caption("Model: Linear Regression")
