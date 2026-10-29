
# Part 2: Weather Classification Mini-Project
# Task 1: Fetch the Data
# City: Ypsilanti, Michigan

import requests
import pandas as pd

url = "https://archive-api.open-meteo.com/v1/archive"

params = {
    "latitude": 42.24,
    "longitude": -83.61,
    "start_date": "2025-01-01",
    "end_date": "2025-12-31",
    "daily": [
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
        "wind_speed_10m_max",
    ],
    "timezone": "America/New_York",
}

response = requests.get(url, params=params)
response.raise_for_status()

df = pd.DataFrame(response.json()["daily"])
df["date"] = pd.to_datetime(df["time"])
df = df.drop("time", axis=1)

print("Dataset shape:", df.shape)
print(df.head())


# Task 2: Engineer the Label
# Using the assignment's default weather thresholds.

df["good_for_running"] = (
    df["temperature_2m_max"].between(7, 26)
    & (df["temperature_2m_min"] >= 0)
    & (df["precipitation_sum"] < 3)
    & (df["wind_speed_10m_max"] < 30)
).astype(int)

print("\nClass balance:")
print(df["good_for_running"].value_counts())

good_fraction = df["good_for_running"].mean()
print(f"Fraction of good running days: {good_fraction:.2%}")


# Task 2 Reflection:
# 117 of 365 days (32.05%) were classified as good for running.
# This seems reasonable for Ypsilanti's climate, given its cold
# winters and variable precipitation and wind conditions.
# The classes are imbalanced (117 good vs. 248 skip days).
# Precision, recall, and F1 for the good class will be important
# because accuracy alone could hide poor performance on good days.

import requests
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression

import json
import sys
from pathlib import Path

import joblib
import sklearn
from sklearn.metrics import f1_score



# Task 3: Build and Evaluate a KNN Classifier

# Select the four weather features and the target label.
feature_names = [
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "wind_speed_10m_max",
]

X = df[feature_names]
y = df["good_for_running"]

# Split the data: 80% training and 20% testing.
# Stratify to preserve the class proportions in both sets.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print("\n--- Task 3: Train/Test Split ---")
print("Training samples:", len(X_train))
print("Test samples:", len(X_test))
print("Training class balance:")
print(y_train.value_counts())
print("Test class balance:")
print(y_test.value_counts())


# Task 3: KNN Pipeline and Cross-Validation

k_values = [1, 3, 5, 7, 9, 11, 15, 21]
cv_results = {}

for k in k_values:
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=k)),
    ])

    scores = cross_val_score( 
        pipeline,
        X_train,
        y_train,
        cv=5,
        scoring="accuracy",
    )

    cv_results[k] = scores.mean()
    print(f"k={k}: Mean CV accuracy = {scores.mean():.4f}")

best_k = max(cv_results, key=cv_results.get)

print(f"\nBest k based on cross-validation: {best_k}")

# Build and fit the final KNN pipeline using the best k.
knn_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=best_k)),
])

knn_pipeline.fit(X_train, y_train)

# Evaluate the fitted model on the held-out test data.
knn_predictions = knn_pipeline.predict(X_test)

knn_accuracy = accuracy_score(y_test, knn_predictions)

print("\n--- Task 3: KNN Test Evaluation ---")
print(f"Test accuracy: {knn_accuracy:.4f}")
print("Classification report:")
print(classification_report(y_test, knn_predictions))

# Task 4: Build and Evaluate a Logistic Regression Classifier

logreg_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("logreg", LogisticRegression(max_iter=1000)),
])

logreg_pipeline.fit(X_train, y_train)
# Generate predictions using the test data.
logreg_predictions = logreg_pipeline.predict(X_test)

# Calculate evaluation metrics.
logreg_accuracy = accuracy_score(y_test, logreg_predictions)

print("\n--- Task 4: Logistic Regression Evaluation ---")
print(f"Test accuracy: {logreg_accuracy:.4f}")
print("Classification report:")
print(classification_report(y_test, logreg_predictions))

# Task 4: Examine Logistic Regression Feature Coefficients

print("\n--- Task 4: Logistic Regression Coefficients ---")

coefficients = logreg_pipeline.named_steps["logreg"].coef_[0]

for feature, coefficient in zip(feature_names, coefficients):
    print(f"{feature}: {coefficient:.4f}")

# Task 4 Reflection:
# Precipitation has the strongest negative coefficient (-2.2649),
# indicating that wetter conditions most strongly push the model
# toward predicting a skip day.
# Wind speed also has a negative coefficient (-0.0350), but its
# influence is much smaller.
# Maximum temperature has a positive coefficient (0.6217),
# while minimum temperature has almost no linear influence.
# These coefficients reflect the logistic regression model's
# learned relationships, which may not fully capture the
# upper and lower temperature limits in the labeling rules.

# Task 5: Compare and Choose
#
# KNN achieved higher test accuracy (94.52%) than Logistic
# Regression (68.49%). KNN also performed better on precision,
# recall, and F1 for the good-running-day class.
#
# Accuracy alone is not the most important metric for a running
# app. I would prioritize precision for the good class because
# a false positive could send someone running in unsuitable
# weather. Recall also matters because false negatives cause
# runners to miss suitable days.
#
# I would deploy KNN because it substantially outperformed
# Logistic Regression on this dataset.
# KNN requires storing training examples and calculating
# distances during prediction, which can increase model size
# and prediction time. Logistic Regression is generally smaller,
# faster, and more interpretable.
# Both models can provide probability estimates, although KNN
# probabilities are based on neighbor proportions and may be
# less informative as confidence measures.
# Given this small dataset, KNN's better predictive performance
# outweighs those tradeoffs.


# Task 6: Save the selected model and metadata

models_dir = Path(__file__).resolve().parent / "models"
models_dir.mkdir(parents=True, exist_ok=True)

model_path = models_dir / "weather_classifier.pkl"
metadata_path = models_dir / "weather_classifier_metadata.json"

# Save the fitted KNN pipeline.
joblib.dump(knn_pipeline, model_path)

# Evaluate the selected model for metadata.
saved_model_predictions = knn_pipeline.predict(X_test)
saved_model_accuracy = accuracy_score(y_test, saved_model_predictions)
saved_model_f1 = f1_score(y_test, saved_model_predictions)

metadata = {
    "python_version": sys.version.split()[0],
    "sklearn_version": sklearn.__version__,
    "feature_names": feature_names,
    "label_rule": (
        "good_for_running = 1 when temperature_2m_max is between "
        "7 and 26 C inclusive, temperature_2m_min >= 0 C, "
        "precipitation_sum < 3 mm, and wind_speed_10m_max < 30 km/h; "
        "otherwise 0."
    ),
    "city": "Ypsilanti, Michigan",
    "test_accuracy": saved_model_accuracy,
    "test_f1": saved_model_f1
}

with open(metadata_path, "w", encoding="utf-8") as metadata_file:
    json.dump(metadata, metadata_file, indent=4)

print(f"Saved model: {model_path}")
print(f"Saved metadata: {metadata_path}")
