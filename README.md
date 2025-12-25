# 🧠 Migraine Clinical Dashboard

A **Python Flask** web application for predicting visual auras and analyzing patient feature contributions using EHR data. The system includes a complete **ETL pipeline** from raw clinical data to an interactive dashboard for clinicians and researchers.



## 🎯 Project Overview

This project demonstrates **full-stack Python development**, showcasing data processing, machine learning integration, and web application deployment.

### **Key Features**

🟢 **Patient Predictions** – Predict migraine auras for individual patients.  
🟢 **Feature Contributions** – Visualize how each feature influences individual predictions.  
🟢 **Global Feature Importance** – View odds ratios and coefficients for all features across the dataset.  
🟢 **Metrics Tracking** – Monitor model performance over time, including **Accuracy, AUC, PSI**.  
🟢 **User Management** – Admin interface for model retraining and version metrics.  
🟢 **ETL Pipeline** – Automated extraction, transformation, and loading of raw EHR data.  
🟢 **Model Integration** – Dynamic loading and predictions using **scikit-learn** models.  

---

## 🔙 Backend

- **Python 3.12** – Core application logic.  
- **Flask** – Web framework for the interactive dashboard.  
- **pandas & numpy** – Data processing and ETL operations.  
- **scikit-learn** – Machine learning prediction models.  
- **joblib** – Model serialization and loading.  
- **SQLAlchemy** – Database access and ORM support.  

---

## 🎨 Frontend

- **Jinja2** – Template engine for dynamic HTML rendering.  
- **Plotly** – Interactive visualization of metrics and feature contributions.  
- **Bootstrap 5** – Responsive UI framework for tables and dashboards.  

---

## 🏗️ Architecture & Patterns

- **ETL Pipeline** – Raw data → Processed dataset → Model predictions.  
- **Clean Code Practices** – Modular, testable Python code for maintainability.  
- **Role-Based Access** – Admin vs general user dashboard access.  
- **Metrics Logging** – Track and display model performance across versions.

---

## ⏯️ Getting Started

### Clone the Repository

```bash
git clone https://github.com/DuztyMusketear/migraine-clinical-dashboard.git
cd migraine-clinical-dashboard
```
### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the ETL Pipeline

```bash
python etl/run_etl.py
```
### Start Flask Application

```bash
python webapp/app.py
```
The dashboard will be available at:
  * `http://127.0.0.1:5000` (or URL shown in terminal)

----


## 🔍Background

During a migraine, up to 30% of individuals experience a transitory disturbance in
neurological function referred to as a migraine aura (Joppeková et al.,2025). Migraine auras
can affect vision, hearing, language, touch, and motor skills. Visual auras, in particular, have
been associated with cortical spreading depression (CSD) (Kitamura & Imai, 2024). CSD is a
depolarization of neuronal and glial cells, leading to neuronal swelling followed by a
depression of electrical activity (Joppeková et al.,2025). Previous studies have focused on the
neurophysiology and genetic factors underlying migraine auras; however, there is limited
research on predicting the manifestation of an aura during a migraine. To address this gap,
we performed an exploratory data analysis (EDA) on a dataset comprised of 400 clinical
observations of patients experiencing migraines. Through our EDA, we developed a
predictive model capable of classifying whether a patient will experience a visual aura. This
model will be integrated into a dashboard, enabling healthcare professionals to visualize risk
factors and prediction outcomes.

---

### License

Source code is licensed under the [MIT license](https://opensource.org/license/mit).

---
### Sources

* Joppeková, Ľ., Pinto, M. J., da Costa, M. D., Boček, R., Berman, G., Salim, Y., Akhtanova, D.,
Abzalbekova, A., MaassenVanDenBrink, A., & Lampl, C. (2025). What does a migraine
aura look like?—A systematic review. The Journal of Headache and Pain, 26(1). https://
doi.org/10.1186/s10194-025-02080-6

* Kitamura, E., & Imai, N. (2024). Molecular and Cellular Neurobiology of Spreading
Depolarization/Depression and Migraine: A Narrative Review. International Journal of
Molecular Sciences, 25(20), 11163. https://doi.org/10.3390/ijms252011163
