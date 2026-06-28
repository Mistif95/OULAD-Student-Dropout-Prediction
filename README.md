# 🎓 Student Dropout Early Warning System
**GWE 2026 Data Science Challenge — Grow With EDM Gen 7**

An end-to-end Data Science project that leverages the **Open University Learning Analytics Dataset (OULAD)** to predict student dropout risk using machine learning. The system serves as an **Early Warning System** for educators to identify at-risk students before they disengage.

---

## 📌 Problem Statement

Student dropout is a critical challenge in higher education. By analyzing demographics, assessment performance, and Virtual Learning Environment (VLE) interaction patterns, this project builds a predictive model that flags students at risk of withdrawal — enabling timely, targeted intervention.

---

## 🎯 Objectives

1. **Data Integration** — Merge student demographics, course info, assessment records, and VLE click data into a unified feature set.
2. **Exploratory Analysis** — Identify behavioral differences between students who complete vs. withdraw from courses.
3. **Predictive Modeling** — Train a LightGBM classifier to predict dropout risk with high accuracy.
4. **Actionable Insights** — Generate automated intervention recommendations for high-risk students.

---

## 🗂️ Repository Structure

```
├── README.md
├── requirements.txt
├── notebooks/
│   └── analysis.ipynb          # Full EDA + ML pipeline
├── src/
│   └── app.py                  # Streamlit deployment app
├── data/                       # OULAD CSV files (not tracked by git)
├── models/
│   ├── lgbm_dropout_model.pkl
│   ├── feature_list.pkl
│   └── thresholds.pkl
└── presentation/
    └── slides.pdf
```

---

## 📦 Dataset

This project uses the **Open University Learning Analytics Dataset (OULAD)**, available on [Kaggle](https://www.kaggle.com/) and the [Open University website](https://analyse.kmi.open.ac.uk/open_dataset).

| File | Description |
|---|---|
| `studentInfo.csv` | Demographics and final results per student |
| `studentRegistration.csv` | Registration and unregistration dates |
| `studentAssessment.csv` | Assessment scores and submission dates |
| `assessments.csv` | Assessment metadata (type, weight, deadline) |
| `vle.csv` | VLE material metadata |
| `courses.csv` | Module duration information |
| `studentVle2M.csv` | VLE click interactions (~2M rows, trimmed from original 10.6M) |

> **Note on `studentVle2M.csv`:** Rows were randomly removed across the full dataset (not filtered by student), so all ~32,593 students remain represented with fewer interaction records. Relative patterns between students are preserved, maintaining model validity.

---

## 🧠 Methodology

### Features Engineered

**Demographic:** `gender`, `disability`, `age_band`, `imd_band`, `highest_education`, `num_of_prev_attempts`

**Assessment:** `mean_score`, `std_score`, `miss_ratio`, `score_trend`, `num_assessments`, `avg_submission_delay`

**VLE Engagement:** `total_clicks`, `active_days`, `avg_daily_clicks`, `engagement_rate`, `click_trend`, `material_diversity`, `activity_span`

**Registration:** `early_unregistration`, `days_until_unreg`

### Model

- **Algorithm:** LightGBM (chosen for superior performance on tabular/imbalanced data and native categorical support)
- **Imbalance handling:** SMOTE oversampling
- **Validation:** Stratified K-Fold cross-validation
- **Interpretability:** SHAP (SHapley Additive exPlanations)
- **Baselines compared:** Logistic Regression, Decision Tree, Random Forest, Gradient Boosting

### Target

Binary classification: `Withdrawn` vs. `Non-Withdrawn` from `final_result`.

---

## 🚀 Deployment

The app is deployed on **Streamlit Community Cloud** and publicly accessible at:

**🔗 https://oulad-student-dropout-prediction-mistif95.streamlit.app/**

### App Features

- **Home** — Project overview, problem background, and navigation
- **EDA Dashboard** — Interactive visualizations and data insights
- **Prediction** — Real-time dropout risk prediction per student
- **About** — Model explanation, evaluation metrics, and team info

### Two Modes

| Mode | Description |
|---|---|
| **Mode A — Saved Model** | Loads `lgbm_dropout_model.pkl` directly for instant prediction |
| **Mode B — Raw Dataset** | Upload OULAD CSVs to re-run feature engineering and predict |

---

## ⚙️ Installation

It is highly recommended to create a virtual environment before installing the dependencies. This ensures that the project packages do not conflict with your system-wide Python installations.

## 1. Clone Repo or download the files
```bash
git clone https://github.com/Mistif95/OULAD-Student-Dropout-Prediction.git
cd OULAD-Student-Dropout-Prediction
pip install -r requirements.txt
```

## 2. Create a virtual environment
```bash
python -m venv venv
```
## 3. Activate the virtual environment
```bash
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

## 4. Install required dependencies
```bash
pip install -r requirements.txt
```
## 5. Running the project
To run the app locally:

```bash
streamlit run src/app.py
```

Place the OULAD CSV files in the `data/` directory before running.

---

## 📊 Model Performance LightGBM 

| Metric | Score |
|---|---|
| Accuracy | 0.9967 |
| ROC-AUC | 0.9997 |
| F1-Score | 0.9951 |
| Recall | 0.9903 |
| Precision | 1.0000 |

---

## 🛠️ Tech Stack

- **Python** — pandas, numpy
- **Visualization** — matplotlib, seaborn, plotly
- **ML** — scikit-learn, LightGBM, imbalanced-learn (SMOTE)
- **Interpretability** — SHAP
- **Deployment** — Streamlit

---

## 👥 Team Runner

| Name | Role |
|---|---|
| Rayhan Aliffio Wibowo | Member |

---

## ⚖️ Ethics & Attribution

- Dataset: [OULAD — Open University Learning Analytics Dataset](https://analyse.kmi.open.ac.uk/open_dataset)
- AI tools used in development: **Claude (Anthropic)**
- All code and analysis are original work by the Team Runner.

---

## 🤖 AI Usage Acknowledgement
In accordance with the guidelines of the GWE 2026 Data Science Challenge, we transparently declare the use of Artificial Intelligence tools during the development of this end-to-end Data Science project.

- Claude (Anthropic): Leveraged for code optimization, troubleshooting complex debugging issues, and refining the structure of our Streamlit deployment.

## 📄 License

This project is submitted as part of the **GWE 2026 Data Science Challenge**. All rights reserved by the Team Runner.
