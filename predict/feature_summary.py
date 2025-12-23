import numpy as np
import pandas as pd

def global_feature_summary(model):
    """Returnd DataFrame of features including coefficients and odds-ratios"""

    FEATURE_COLS = model.feature_names_in_
    coefs = model.coef_[0]
    odds_ratios = np.exp(coefs)
    df = pd.DataFrame({
        "Feature": FEATURE_COLS,
        "Coeffiecient": coefs,
        "Odds_Ratio": odds_ratios
    }).sort_values(by="Odds_Ratio", ascending=False)
    return df

def patient_feature_contribution(model, X_patient):
    """Returns DataFrame showning contribution of each feature"""

    FEATURE_COLS = model.feature_names_in_
    log_odds = X_patient.iloc[0].values * model.coef_[0]
    df = pd.DataFrame({
        "Feature": FEATURE_COLS,
        "Contribution": log_odds
    }).sort_values(by="Contribution", ascending=False)
    return df