from flask import Flask, render_template, request
import pandas as pd
import joblib
import os
import logging

app = Flask(__name__)

# ---------------------------
# Logging (production ready)
# ---------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------
# Load model safely
# ---------------------------
MODEL_PATH = os.path.join("model", "best_rf_model.pkl")
ENCODER_PATH = os.path.join("model", "label_encoders.pkl")

try:
    model = joblib.load(MODEL_PATH)
    label_encoders = joblib.load(ENCODER_PATH)
    logger.info("Model and encoders loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    model = None
    label_encoders = None


cat_cols = [
    "job_title",
    "experience_level",
    "education_required",
    "country",
    "remote_work",
    "company_size",
    "industry"
]


# ---------------------------
# Home page
# ---------------------------
@app.route("/")
def home():

    if label_encoders is None:
        return "Model not loaded properly"

    return render_template(
        "index.html",
        job_titles=label_encoders["job_title"].classes_,
        experience_levels=label_encoders["experience_level"].classes_,
        educations=label_encoders["education_required"].classes_,
        countries=label_encoders["country"].classes_,
        remote_options=label_encoders["remote_work"].classes_,
        company_sizes=label_encoders["company_size"].classes_,
        industries=label_encoders["industry"].classes_,
    )


# ---------------------------
# Prediction route
# ---------------------------
@app.route("/predict", methods=["POST"])
def predict():

    try:

        input_data = {
            "job_title": request.form.get("job_title"),
            "experience_level": request.form.get("experience_level"),
            "education_required": request.form.get("education_required"),
            "country": request.form.get("country"),
            "remote_work": request.form.get("remote_work"),
            "company_size": request.form.get("company_size"),
            "industry": request.form.get("industry"),
            "years_of_experience": float(request.form.get("years_of_experience", 0)),
            "salary_range": float(request.form.get("salary_range", 0)),
            "salary_mid": float(request.form.get("salary_mid", 0)),
            "skill_count": float(request.form.get("skill_count", 0)),
            "demand_score": float(request.form.get("demand_score", 0)),
        }

        input_df = pd.DataFrame([input_data])

        # Encode categorical columns
        for col in cat_cols:
            input_df[col] = label_encoders[col].transform(input_df[col].astype(str))

        # Prediction
        pred_salary = model.predict(input_df)[0]

        prediction = f"${pred_salary:,.0f}"

        logger.info("Prediction successful")

    except Exception as e:

        logger.error(f"Prediction error: {e}")
        prediction = "Prediction failed"

    return render_template(
        "index.html",
        prediction=prediction,
        job_titles=label_encoders["job_title"].classes_,
        experience_levels=label_encoders["experience_level"].classes_,
        educations=label_encoders["education_required"].classes_,
        countries=label_encoders["country"].classes_,
        remote_options=label_encoders["remote_work"].classes_,
        company_sizes=label_encoders["company_size"].classes_,
        industries=label_encoders["industry"].classes_,
    )


# ---------------------------
# Run locally
# ---------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)