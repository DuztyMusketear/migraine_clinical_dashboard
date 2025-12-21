import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
import joblib
import os

#Load cleaned data
df = pd.read_csv("data/migraine_symptom_classification_clean.csv")

X = df.drop("Visual", axis=1)
y = df["Visual"]

#Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Logistic Regression (Hyperparameter Tuning)
param_grid_lr = {
    'C': [0.01, 0.1, 1, 10, 100]
}

log_reg = LogisticRegression(max_iter=1000)
grid_lr = GridSearchCV(log_reg, param_grid_lr, cv=5, scoring='f1')
grid_lr.fit(X_train, y_train)


#Get Best Model
best_lr = grid_lr.best_estimator_

#Save Model
os.makedirs("model", exist_ok=True)
joblib.dump(best_lr, os.path.join("model", "logistic_model.pk1"))

print("Best Logistic Regression Model Saved to: model/logistic_model.pk1")