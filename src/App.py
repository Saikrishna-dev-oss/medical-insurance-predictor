import streamlit as st
import joblib
import pandas as pd
import warnings
import os

warnings.filterwarnings("ignore", category = UserWarning)

st.set_page_config(page_title="HealthQuote AI", page_icon="🏥", layout="centered")

st.markdown("""
    <style>
    /* Hide Streamlit default menu and footer for a clean app feel */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Style the main predict button */
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        font-weight: bold;
        background-color: #0066cc;
        color: white;
        transition: all 0.3s ease;
        padding: 10px 0px;
    }
    .stButton>button:hover {
        background-color: #004c99;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    /* Custom CSS card for the final price output */
    .result-card {
        background-color: #f8f9fa;
        padding: 25px;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 10px rgba(0,0,0,0.08);
        margin-top: 20px;
        border: 1px solid #e1e4e8;
    }
    
    /* Ensure the card looks good in Dark Mode too */
    @media (prefers-color-scheme: dark) {
        .result-card {
            background-color: #1e1e1e;
            border: 1px solid #333;
        }
    }
    </style>
""", unsafe_allow_html=True)

# --- Model Loading ---
@st.cache_resource
def load_model():
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    model_path = os.path.join(current_dir, "..", "saved_models", "linear_regression_model.pkl")
    
    return joblib.load(model_path)

model = load_model()

# ---  App Header ---
st.markdown("<h1 style='text-align: center;'>🏥 HealthQuote AI</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray; margin-bottom: 30px;'>Intelligent Medical Insurance Cost Prediction Engine</p>", unsafe_allow_html=True)

# --- 5. Input Section (Inside a sleek bordered container) ---
with st.container(border=True):
    st.markdown("### Patient Vitals & Demographics")
    st.write("") # Spacer
    
    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age", min_value=18, max_value=100, value=25)
        gender_input = st.selectbox("Gender", options=["Male", "Female"])
        bmi = st.number_input("Body Mass Index (BMI)", min_value=10.0, max_value=50.0, value=22.5, step=0.1)

    with col2:
        children = st.number_input("Number of Children", min_value=0, max_value=10, value=0)
        smoker_input = st.radio("Smoker Status", options=["No", "Yes"])

    st.divider()
    currency = st.radio("Select Output Currency", options=["INR (₹)", "USD ($)"], horizontal=True)

# --- 6. Prediction Logic ---
sex = 1 if gender_input == 'Male' else 0
smoker = 1 if smoker_input == 'Yes' else 0

if st.button("Generate Precision Quote"):
    input_features = pd.DataFrame({
            'age': [age], 
            'sex':[sex],
            'bmi': [bmi], 
            'children': [children], 
            'smoker': [smoker],
            'region_northwest': [0],
            'region_southeast': [0],   
            'region_southwest': [0],
            'bmi_smoker_interaction': [bmi * smoker]
         })
    
    predicted_cost_usd = model.predict(input_features)[0]
    
    # --- 7. Render the Premium Output Card ---
    if currency == "INR (₹)":
        predicted_cost_inr = predicted_cost_usd * 84.0
        display_text = f"₹ {predicted_cost_inr:,.2f}"
    else:
        display_text = f"$ {predicted_cost_usd:,.2f}"
        
    st.markdown(f"""
        <div class="result-card">
            <p style="margin-bottom: 5px; color: gray; font-size: 18px;">Estimated Yearly Premium</p>
            <h1 style="margin-top: 0px; color: #0066cc; font-size: 42px;">{display_text}</h1>
        </div>
    """, unsafe_allow_html=True)