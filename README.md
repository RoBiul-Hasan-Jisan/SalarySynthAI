# SalarySynthAI 

Predict AI job salaries for 2025–2026 using Machine Learning.  

SalarySynthAI helps **job seekers, recruiters, and companies** estimate expected salaries for AI roles globally using real market data.

---

##  Project Highlights

- **Total Jobs:** 1,500  
- **Average Salary:** $194,892  
- **Highest Paying Role:** AI Solutions Architect  
- **Top Country Salary:** USA  
- **Remote Jobs Percentage:** 75.4%  
- **Best Model R² Score:** 0.847  

---

##  Dataset

SalarySynthAI uses the **AI job market dataset for 2025–2026**, containing:

- Job Title  
- Experience Level  
- Education Required  
- Country  
- Remote Work Option  
- Company Size  
- Industry  
- Years of Experience  
- Salary Range & Mid Salary  
- Number of Required Skills  
- Demand Score  
- Annual Salary (USD)  

---

##  Features

**Categorical Inputs:**  
`job_title`, `experience_level`, `education_required`, `country`, `remote_work`, `company_size`, `industry`  

**Numeric Inputs:**  
`years_of_experience`, `salary_range`, `salary_mid`, `skill_count`, `demand_score`  

---

##  Model Leaderboard

| Model               | MAE      | R² Score | CV R² Score |
|--------------------|----------|----------|-------------|
| Gradient Boosting   | 18,904   | 0.847    | 0.874       |
| XGBoost             | 18,204   | 0.847    | 0.887       |
| Random Forest       | 24,144   | 0.712    | 0.794       |
| Linear Regression   | 40,762   | 0.384    | 0.473       |

**Best Model:** Gradient Boosting (R² = 0.847)  

**Top Features:** `years_of_experience`, `salary_mid`, `job_title`, `skill_count`, `demand_score`  

**Grid Search Parameters (Random Forest Example):**

```python
param_grid = {
    "n_estimators": [200, 300],
    "max_depth": [10, 15, 20]
}

```

## Visualizations

Salary Distribution

Salary by Experience Level

Top 15 Highest Paying AI Roles

Feature Correlation Heatmap

Feature Importance

Actual vs Predicted Salary Scatter Plot

## Technologies

Languages: Python 3.10

Libraries: pandas, numpy, scikit-learn, xgboost, matplotlib, seaborn, joblib, Flask, gunicorn

Deployment: Render / Local Flask server

---

## Usage
---
1. Clone the Repository
```bash
git clone https://github.com/yourusername/SalarySynthAI.git
cd SalarySynthAI
```

2. Install Dependencies
```bash
pip install -r requirements.txt
```
3. Run Locally
```bash
python app.py

Open in browser: http://127.0.0.1:5000

```

4. Deploy on Render
```bash
Build Command: pip install -r requirements.txt
Start Command: gunicorn app:app

```
---
**Live Demo:** https://salarysynthai.onrender.com/




## Project Structure
```bash
SalarySynthAI/
│
├── app.py                  # Flask app
├── requirements.txt        # Python dependencies
├── model/
│   ├── best_rf_model.pkl   # Trained Random Forest model
│   └── label_encoders.pkl  # Encoders for categorical features
├── templates/
│   └── index.html          # Frontend HTML
├── static/
│   └── style.css           
└── README.md               # Documentation

```
