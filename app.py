import streamlit as st
import joblib
import numpy as np

# 1. Page Configuration
st.set_page_config(page_title="Multiple Disease Prediction", layout="wide", page_icon="🏥")

# 2. Load the Trained Models
@st.cache_resource
def load_models():
    p_model = joblib.load('parkinsons_model.pkl')
    p_scaler = joblib.load('parkinsons_scaler.pkl')
    k_model = joblib.load('kidney_model.pkl')
    l_model = joblib.load('liver_model.pkl')
    return p_model, p_scaler, k_model, l_model

try:
    parkinsons_model, parkinsons_scaler, kidney_model, liver_model = load_models()
    models_loaded = True
except Exception as e:
    st.error(f"Error loading models. Please ensure Phase 2 completed successfully. Details: {e}")
    models_loaded = False

# 3. Sidebar Navigation
st.sidebar.title("Predictive Healthcare System")
selected_disease = st.sidebar.radio("Select Disease to Predict:", 
                                    ["Parkinson's Disease", "Kidney Disease", "Liver Disease"])

# 4. Main Application Logic
if models_loaded:
    st.title(f"🩺 {selected_disease} Predictor")
    st.markdown("Enter the patient's lab test results below to generate an AI-powered risk assessment.")

    # ==========================================
    # PARKINSON'S PREDICTOR
    # ==========================================
    if selected_disease == "Parkinson's Disease":
        st.info("Please enter the 22 acoustic vocal measurement values separated by commas.")
        st.write("**Example input:** 119.992, 157.302, 74.997, 0.00784, 0.00007, 0.0037, 0.00554, 0.01109, 0.04374, 0.426, 0.02182, 0.0313, 0.02971, 0.06545, 0.02211, 21.033, 0.414783, 0.815285, -4.813031, 0.266482, 2.301442, 0.284654")
        
        user_input = st.text_input("Enter Test Results:")
        
        if st.button("Run Parkinson's Prediction"):
            if user_input:
                try:
                    # Convert input string to numeric array
                    input_list = [float(x.strip()) for x in user_input.split(',')]
                    if len(input_list) == 22:
                        input_array = np.asarray(input_list).reshape(1, -1)
                        # Scale the data using the saved scaler
                        scaled_data = parkinsons_scaler.transform(input_array)
                        prediction = parkinsons_model.predict(scaled_data)
                        
                        st.divider()
                        if prediction == 1:
                            st.error("⚠️ **Prediction:** High Risk of Parkinson's Disease. Please consult a neurologist.")
                        else:
                            st.success("✅ **Prediction:** Low Risk of Parkinson's Disease.")
                    else:
                        st.warning(f"Expected 22 values, but got {len(input_list)}. Please check your input.")
                except ValueError:
                    st.warning("Please ensure all inputs are valid numbers separated by commas.")

    # ==========================================
    # KIDNEY DISEASE PREDICTOR
    # ==========================================
    elif selected_disease == "Kidney Disease":
        st.info("Please enter the 24 clinical feature values separated by commas (including age, blood pressure, specific gravity, etc.).")
        
        user_input = st.text_input("Enter Test Results:")
        
        if st.button("Run Kidney Prediction"):
            if user_input:
                try:
                    input_list = [float(x.strip()) for x in user_input.split(',')]
                    if len(input_list) == 24:
                        input_array = np.asarray(input_list).reshape(1, -1)
                        prediction = kidney_model.predict(input_array)
                        
                        st.divider()
                        if prediction == 1:
                            st.error("⚠️ **Prediction:** High Risk of Chronic Kidney Disease (CKD).")
                        else:
                            st.success("✅ **Prediction:** Low Risk of Chronic Kidney Disease.")
                    else:
                        st.warning(f"Expected 24 values, but got {len(input_list)}. Please check your input.")
                except ValueError:
                    st.warning("Please ensure all inputs are valid numbers separated by commas.")

    # ==========================================
    # LIVER DISEASE PREDICTOR
    # ==========================================
    elif selected_disease == "Liver Disease":
        st.info("Please enter the 10 liver function test values separated by commas (Age, Gender [1=Male, 0=Female], Total Bilirubin, Direct Bilirubin, etc.).")
        st.write("**Example input:** 65, 0, 0.7, 0.1, 187, 16, 18, 6.8, 3.3, 0.9")
        
        user_input = st.text_input("Enter Test Results:")
        
        if st.button("Run Liver Prediction"):
            if user_input:
                try:
                    input_list = [float(x.strip()) for x in user_input.split(',')]
                    if len(input_list) == 10:
                        input_array = np.asarray(input_list).reshape(1, -1)
                        prediction = liver_model.predict(input_array)
                        
                        st.divider()
                        if prediction == 1:
                            st.error("⚠️ **Prediction:** High Risk of Liver Disease.")
                        else:
                            st.success("✅ **Prediction:** Low Risk of Liver Disease.")
                    else:
                        st.warning(f"Expected 10 values, but got {len(input_list)}. Please check your input.")
                except ValueError:
                    st.warning("Please ensure all inputs are valid numbers separated by commas.")
                