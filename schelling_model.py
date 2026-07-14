import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import chi2_contingency

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
    RocCurveDisplay
)


# ------------------------------------------------------------
# 1. GENERATE SYNTHETIC USER DATA
# ------------------------------------------------------------

np.random.seed(42)

N = 5000

data = pd.DataFrame({
    "age": np.random.randint(
        18,
        70,
        N
    ),

    "income": np.random.normal(
        70000,
        25000,
        N
    ),

    "previous_purchases": np.random.poisson(
        3,
        N
    ),

    "discount": np.random.uniform(
        0,
        30,
        N
    ),

    "treatment": np.random.binomial(
        1,
        0.5,
        N
    )
})


# ------------------------------------------------------------
# 2. CREATE A BEHAVIOURAL CONVERSION PROCESS
# ------------------------------------------------------------

# Treatment represents exposure to a default option.

log_odds = (
    -3.0
    +
    0.015 * data["previous_purchases"]
    +
    0.04 * data["discount"]
    +
    0.70 * data["treatment"]
)


probability = (
    1 /
    (
        1 + np.exp(-log_odds)
    )
)


data["conversion"] = np.random.binomial(
    1,
    probability
)


# ------------------------------------------------------------
# 3. COMPARE CONTROL AND TREATMENT
# ------------------------------------------------------------

conversion_rates = (
    data.groupby("treatment")
    ["conversion"]
    .mean()
)

print(
    "Conversion Rates"
)

print(
    conversion_rates
)


control_rate = conversion_rates[0]

treatment_rate = conversion_rates[1]


uplift = (
    treatment_rate
    -
    control_rate
)


print(
    "\nAbsolute Treatment Effect:",
    round(uplift, 4)
)


print(
    "Percentage Point Uplift:",
    round(uplift * 100, 2)
)


# ------------------------------------------------------------
# 4. STATISTICAL SIGNIFICANCE TEST
# ------------------------------------------------------------

contingency_table = pd.crosstab(
    data["treatment"],
    data["conversion"]
)


chi2, p_value, dof, expected = (
    chi2_contingency(
        contingency_table
    )
)


print(
    "\nP-value:",
    p_value
)


# ------------------------------------------------------------
# 5. LOGISTIC REGRESSION
# ------------------------------------------------------------

features = [
    "age",
    "income",
    "previous_purchases",
    "discount",
    "treatment"
]


X = data[features]

y = data["conversion"]


X_train, X_test, y_train, y_test = (
    train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )
)


model = LogisticRegression(
    max_iter=1000
)


model.fit(
    X_train,
    y_train
)


# ------------------------------------------------------------
# 6. PREDICTIONS
# ------------------------------------------------------------

predictions = model.predict(
    X_test
)


probabilities = model.predict_proba(
    X_test
)[:, 1]


# ------------------------------------------------------------
# 7. MODEL PERFORMANCE
# ------------------------------------------------------------

print(
    "\nAccuracy:",
    accuracy_score(
        y_test,
        predictions
    )
)


print(
    "\nROC-AUC:",
    roc_auc_score(
        y_test,
        probabilities
    )
)


print(
    "\nClassification Report"
)

print(
    classification_report(
        y_test,
        predictions
    )
)


# ------------------------------------------------------------
# 8. COEFFICIENT INTERPRETATION
# ------------------------------------------------------------

coefficients = pd.DataFrame({

    "Variable":
        features,

    "Coefficient":
        model.coef_[0],

    "Odds_Ratio":
        np.exp(
            model.coef_[0]
        )

})


print(
    "\nLogistic Regression Results"
)

print(
    coefficients
)


# ------------------------------------------------------------
# 9. ROC CURVE
# ------------------------------------------------------------

RocCurveDisplay.from_predictions(
    y_test,
    probabilities
)

plt.title(
    "ROC Curve: Conversion Prediction Model"
)

plt.show()
