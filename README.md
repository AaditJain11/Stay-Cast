# StayCast • In-Hospital Length of Stay (LOS) Predictor

**StayCast** is an end-to-end machine learning web application that predicts how long a patient is likely to stay in the hospital. The project uses a two-stage modeling approach to handle the highly right-skewed nature of hospital Length of Stay (LOS) data.

The project is based on the research methodology proposed by Xu et al. (2022) in *Predicting in-hospital length of stay: a two-stage modeling approach to account for highly skewed data*, published in BMC Medical Informatics and Decision Making.

[Read the research paper](https://bmcmedinformdecismak.biomedcentral.com/articles/10.1186/s12911-022-01990-2)

The main idea behind StayCast is to first identify whether a patient is likely to have a short or prolonged hospital stay and then estimate the duration for patients classified as short-stay cases.

---

## 📌 Problem Overview

Hospital Length of Stay is an important factor in healthcare planning. It helps hospitals manage beds, schedule procedures, and allocate resources more effectively.

However, LOS data is usually heavily right-skewed. Most patients stay in the hospital for a relatively short period, while a smaller number of patients require much longer stays.

This makes prediction challenging. A traditional regression model trained on raw LOS values may struggle to balance predictions for short stays and prolonged hospitalizations.

**StayCast addresses this challenge using a two-stage machine learning approach.**

### How the model works

**Stage 1 — Short-Stay vs. Prolonged-Stay Classification**

A balanced Random Forest Classifier determines whether a patient's predicted hospital stay falls into one of two categories:

* Short stay: Less than 7 days
* Prolonged stay: 7 days or more

The classifier is tuned to prioritize sensitivity for short-stay cases, helping identify patients who are likely to be discharged within the first 7 days.

**Stage 2 — Length of Stay Regression**

For patients classified as short-stay cases, a Random Forest Regressor predicts the expected LOS using a logarithmically transformed target:

$$
\log(1+\text{LOS})
$$

The prediction is then converted back into days using the exponential inverse transformation, `expm1()`.

For patients classified as prolonged-stay cases, the model reports 7 days as the predicted duration, corresponding to the classification threshold. This is a threshold-based output rather than an estimate of the patient's actual prolonged stay.

---

## 🚀 Live Demo

Explore the deployed StayCast application:

**[Visit StayCast](https://stay-cast.onrender.com)**

| Component        | Technology                     |
| ---------------- | ------------------------------ |
| Backend          | Python, Flask                  |
| Machine Learning | Scikit-learn                   |
| Data Processing  | Pandas, NumPy                  |
| Frontend         | HTML, Tailwind CSS, JavaScript |
| Visualizations   | Chart.js                       |
| Icons            | Lucide Icons                   |
| Deployment       | Render                         |

---

## 🛠 Features & Architecture

### 1. Leakage-Free Preprocessing

The model uses features that are available before the procedure, ensuring that predictions do not depend on information that would only become available after the patient's hospital stay.

Encounter identifiers and admission or discharge timestamps are excluded from the model inputs to avoid target leakage.

### 2. Categorical and Numerical Data Processing

The preprocessing pipeline handles different types of patient information:

* One-hot encoding is used for hospital facility locations (`facid`).
* Historical readmission counts (`rcount`) are converted into numerical values.
* Numerical patient measurements are processed for model training.

### 3. Logarithmic Target Transformation

The LOS target is transformed using `log1p()` before training the second-stage regressor.

This reduces the influence of larger LOS values during model fitting and helps the model learn from a highly skewed distribution.

After prediction, `expm1()` converts the output back into the original LOS scale, measured in days.

### 4. Research-Based Model Evaluation

StayCast includes evaluation metrics designed to examine different aspects of LOS prediction.

**Customized Truncated Loss**

A modified loss metric that does not penalize predictions when both the actual and predicted LOS fall within the prolonged-stay category of 7 days or more.

**Calibration Slope**

Measures the relationship between actual and predicted LOS. A slope of 1.0 is the reference value for perfect calibration under this measure.

**Stratified Mean Absolute Error (MAE)**

Evaluates prediction errors across different LOS categories:

* 0–2 days
* 2–4 days
* 4–7 days
* 7 days or more

This provides a more detailed view of model performance across different patient stay durations.

### 5. Interactive Healthcare Dashboard

The web application provides a simple interface for entering patient information and viewing LOS predictions.

Key features include:

* Patient information input form
* Predicted hospital stay in days
* Short-stay or prolonged-stay classification
* Visual indicators showing the predicted stay in relation to the selected cohort
* Interactive charts and status badges
* Responsive layout for different screen sizes

---

## 📊 Dataset & Feature Matrix

The model is trained and validated using a public dataset containing approximately 100,000 patient encounters, with LOS values bounded at 17 days.

The following features are used as model inputs:

| Feature Category             | Features                                            |
| ---------------------------- | --------------------------------------------------- |
| Administrative & Demographic | `gender`, `facid`, `rcount`                         |
| Vital Signs                  | `pulse`, `respiration`, `bmi`                       |
| Laboratory Measurements      | `hematocrit`, `sodium`, `glucose`, `bloodureanitro` |
| Target Variable              | `lengthofstay`                                      |

### Feature descriptions

| Feature          | Description                              |
| ---------------- | ---------------------------------------- |
| `gender`         | Patient gender                           |
| `facid`          | Hospital facility identifier             |
| `rcount`         | Historical readmission count             |
| `pulse`          | Pulse rate in beats per minute           |
| `respiration`    | Respiratory rate                         |
| `bmi`            | Body Mass Index in kg/m²                 |
| `hematocrit`     | Hematocrit measurement                   |
| `sodium`         | Blood sodium level in mEq/L              |
| `glucose`        | Blood glucose level in mg/dL             |
| `bloodureanitro` | Blood urea nitrogen measurement in mg/dL |
| `lengthofstay`   | Hospital Length of Stay in days          |

---

## 📂 Repository Structure

```text
StayCast/
│
├── app.py
│   # Flask application, prediction endpoints, and web interface
│
├── stage1_classifier.pkl
│   # Trained Random Forest Classifier
│
├── stage2_regressor.pkl
│   # Trained Random Forest Regressor
│
├── requirements.txt
│   # Required Python packages
│
└── README.md
    # Project documentation
```

---

## 🎯 Project Objective

The objective of StayCast is to explore how machine learning can be used to estimate hospital Length of Stay and support better hospital resource planning.

By combining classification and regression, the project provides a structured way to handle both short and prolonged hospital stays while making the prediction process accessible through a web-based interface.

**StayCast is a research-oriented prototype for LOS prediction and resource planning, not a clinical decision-making or diagnostic tool.**
