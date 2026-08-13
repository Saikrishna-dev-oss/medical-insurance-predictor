import joblib
import pandas as pd
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

def load_model():
    """Load the serialized Machine learning model from the 'model' directory."""

    model_path = "../saved_models/linear_regression_model.pkl"
    try:
        return joblib.load(model_path)
    except FileNotFoundError:
        print(f"Error: Model not found at {model_path}. Did you run the notebook?")
        exit()

def get_user_input():
    """Prompt the user for input features and return them as a DataFrame."""
    
    print("\n--- Medical Insurance Quote Generator ---")
    age = float(input("Enter Age (e.g., 25): "))
    sex_input = input("Enter Gender (male/female): ").strip().lower()
    sex = 1 if sex_input == 'male' else 0
    bmi = float(input("Enter BMI (e.g., 28.5): "))
    children = int(input("Enter Number of Children (e.g., 0): "))

    smoker_input = input("Are you a smoker? (yes/no): ").strip().lower()
    smoker = 1 if smoker_input == 'yes' else 0


    return age, sex, bmi, children, smoker

def predict_insurance_cost(model, age, sex, bmi, children, smoker):
    """Predict the insurance cost based on user input features."""
    
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
    predicted_cost = model.predict(input_features)
    return predicted_cost

if  __name__ == "__main__":
    model = load_model()
    age, sex, bmi, children, smoker = get_user_input()

    predicted_cost = predict_insurance_cost(model, age, sex, bmi, children, smoker)
    print(f"\n>> Estimated Yearly Medical Charges: ${predicted_cost[0]:,.2f} <<\n")