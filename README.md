# 🧠 Migraine Clinical Dashboard

A **Python Flask** web application for predicting visual auras and analyzing patient feature contributions using EHR data. The system includes a complete **ETL pipeline** from raw clinical data to an interactive dashboard for clinicians and researchers.



## 🎯 Project Overview

This project demonstrates **full-stack Python development**, showcasing data processing, machine learning integration, and web application deployment. The application allows clinicians to:

### **Key Features**

🟢 **Patient Predictions** – Predict migraine auras for individual patients.  
🟢 **Feature Contributions** – Visualize how each feature influences individual predictions.  
🟢 **Global Feature Importance** – View odds ratios and coefficients for all features across the dataset.  
🟢 **Metrics Tracking** – Monitor model performance over time, including **Accuracy, AUC, PSI**.  
🟢 **User Management** – Admin interface for model retraining and version metrics.  
🟢 **ETL Pipeline** – Automated extraction, transformation, and loading of raw EHR data.  
🟢 **Model Integration** – Dynamic loading and predictions using **scikit-learn** models.  

---

## 🛠️ Backend

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
- 
