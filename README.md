# Check out the live demo of the application <u>[here](https://medical-insurance-predictor-msk.streamlit.app/)
# 🏥 HealthQuote AI: Medical Insurance Cost Predictor

An end-to-end machine learning pipeline and interactive web application designed to predict yearly medical insurance premiums based on patient demographics and vitals. 

This project bridges the gap between raw statistical data analysis and a production-ready, locally deployable application.

---

## 🚀 Project Overview

The core objective of this project is to apply Multiple Linear Regression to a medical dataset, moving through the entire data science lifecycle:
1. **Exploratory Data Analysis (EDA):** Identifying severe right-skewness in medical charges and isolating the driving factors (smoking status).
2. **Data Preprocessing:** Applying Binary and One-Hot Encoding to categorical variables.
3. **Feature Engineering:** Designing a specific mathematical interaction term to capture non-linear relationships.
4. **Model Deployment:** Serializing the trained weights and serving them through a modern Streamlit web interface with dual-currency (USD/INR) support.

## 🧠 The Mathematical Optimization

Standard linear models often fail to capture compounding variables. During the visual EDA phase, it was discovered that a high Body Mass Index (BMI) only drastically inflates medical costs if the patient is **also a smoker**. 

To solve this, a custom `bmi_smoker_interaction` feature was engineered before training:
* **Without Interaction Feature:** The baseline model achieved an R-squared score of ~78%.
* **With Interaction Feature:** By mathematically multiplying BMI and Smoker status, the model successfully recognized the compounding risk, pushing the R-squared accuracy above **84%**.

## 🛠️ Tech Stack & Architecture

* **Core Language:** Python 3
* **Data Processing & Math:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn (Ordinary Least Squares Linear Regression)
* **Visualizations:** Matplotlib, Seaborn
* **Frontend/UI:** Streamlit
* **Serialization:** Joblib

### Directory Structure
```text
medical-insurance-predictor/
│
├── data/
│   └── raw/insurance.csv             # The raw Kaggle dataset
├── notebooks/
│   └── 01_data_exploration.ipynb     # EDA, preprocessing, and model training sandbox
├── saved_models/
│   └── linear_regression_model.pkl   # The serialized model weights
├── src/
│   ├── predict.py                    # Terminal-based inference engine
│   └── app.py                        # Streamlit web application
├── .gitignore
└── README.md
```

---


### 💻 Installation & Usage

To run this application locally on your machine:

**1. Clone the repository**
```bash
git clone [https://github.com/Saikrishna-dev-oss/medical-insurance-predictor](https://github.com/Saikrishna-dev-oss/medical-insurance-predictor)
cd medical-insurance-predictor

```
**2. Create a virtual environment**
```bash
if using conda:

    conda create -n aiml_env1 python=3.10 -y
    conda activate aiml_env1

if using venv:
    python -m venv aiml_env1

    source aiml_env1/bin/activate  # Linux/Mac
    env\Scripts\activate  # Windows
```
***3. Install dependencies**
```bash
pip install pandas numpy scikit-learn matplotlib seaborn streamlit joblib

**4. Run the Streamlit application**
```bash
    cd src
    streamlit run app.py
```

# Author

Mangali Sai Krishna (MSK)

Artificial Intelligence & Machine Learning Engineering

This project was developed to establish core development intuition in algorithmic modeling, feature engineering, and application deployment.
