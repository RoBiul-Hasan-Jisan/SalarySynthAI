import streamlit as st
import pandas as pd
import joblib

# Load model and encoders
model = joblib.load(r"D:\AI JOBS MARKET\model\best_rf_model.pkl")
label_encoders = joblib.load(r"D:\AI JOBS MARKET\model\label_encoders.pkl")

cat_cols = [
    "job_title", "experience_level", "education_required",
    "country", "remote_work", "company_size", "industry"
]

# Streamlit UI
st.title(" AI Job Salary Predictor (2025-2026)")

# Use encoder classes for dropdowns
job_title = st.selectbox("Job Title", label_encoders["job_title"].classes_)
experience_level = st.selectbox("Experience Level", label_encoders["experience_level"].classes_)
education_required = st.selectbox("Education Required", label_encoders["education_required"].classes_)
country = st.selectbox("Country", label_encoders["country"].classes_)
remote_work = st.selectbox("Remote Work", label_encoders["remote_work"].classes_)
company_size = st.selectbox("Company Size", label_encoders["company_size"].classes_)
industry = st.selectbox("Industry", label_encoders["industry"].classes_)

# Numeric inputs
years_of_experience = st.number_input("Years of Experience", min_value=0, max_value=50, value=3)
salary_range = st.number_input("Salary Range ($)", min_value=0, value=20000)
salary_mid = st.number_input("Salary Mid ($)", min_value=0, value=120000)
skill_count = st.number_input("Number of Required Skills", min_value=0, value=5)
demand_score = st.number_input("Demand Score", min_value=0, max_value=100, value=80)

# Predict
if st.button("Predict Salary"):
    input_df = pd.DataFrame([{
        "job_title": job_title,
        "experience_level": experience_level,
        "education_required": education_required,
        "country": country,
        "remote_work": remote_work,
        "company_size": company_size,
        "industry": industry,
        "years_of_experience": years_of_experience,
        "salary_range": salary_range,
        "salary_mid": salary_mid,
        "skill_count": skill_count,
        "demand_score": demand_score
    }])

    # Encode categorical columns
    for col in cat_cols:
        input_df[col] = label_encoders[col].transform(input_df[col].astype(str))

    # Predict
    pred_salary = model.predict(input_df)[0]
    st.success(f" Predicted Annual Salary: ${pred_salary:,.0f}")