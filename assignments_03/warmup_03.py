import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay, classification_report,
)
import joblib

df = pd.read_csv("weather_classification.csv")
numeric = ["temperature_2m_max", "temperature_2m_min", "precipitation_sum", "wind_speed_10m_max"]
X = df[numeric]
y = df["good_for_running"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- PREPROCESSING ---

# Q1: StandardScaler
# Fit only on training data to prevent data leakage from the test set.

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n--- Preprocessing Q1 ---")
print("Scaled training column means:")
print(X_train_scaled.mean(axis=0))


# Q2: One-hot encode season

encoder = OneHotEncoder(sparse_output=False)
season_encoded = encoder.fit_transform(df[["season"]])

print("\n--- Preprocessing Q2 ---")
print("Encoded season array shape:", season_encoded.shape)
print("Encoded feature names:", encoder.get_feature_names_out())

# The season column becomes four columns because the dataset
# contains four categories: winter, spring, summer, and fall.


# Q3: ColumnTransformer and LogisticRegression Pipeline

features_with_season = numeric + ["season"]

X_full = df[features_with_season]

X_full_train, X_full_test, y_full_train, y_full_test = train_test_split(
    X_full, y, test_size=0.2, random_state=42, stratify=y
)

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", StandardScaler(), numeric),
        ("season", OneHotEncoder(handle_unknown="ignore"), ["season"])
    ]
)

preprocessing_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

preprocessing_pipeline.fit(X_full_train, y_full_train)

print("\n--- Preprocessing Q3 ---")
print("Pipeline test accuracy:",
      preprocessing_pipeline.score(X_full_test, y_full_test))

# The pipeline automatically applies the required preprocessing
# before fitting or predicting, so scaling and encoding do not
# have to be performed manually.

# --- KNN ---

# Q1: Build and evaluate KNN with k=5

knn_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=5))
])

knn_pipeline.fit(X_train, y_train)
knn_predictions = knn_pipeline.predict(X_test)

print("\n--- KNN Q1 ---")
print("Test accuracy:", accuracy_score(y_test, knn_predictions))
print("Classification report:")
print(classification_report(y_test, knn_predictions))


# Q2: Compare k values using 5-fold cross-validation

print("\n--- KNN Q2 ---")

k_values = [1, 3, 5, 7, 9, 11, 15, 21]
cv_results = {}

for k in k_values:
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=k))
    ])

    scores = cross_val_score(
        model, X_train, y_train, cv=5, scoring="accuracy"
    )

    cv_results[k] = scores.mean()
    print(f"k={k}: Mean CV accuracy = {scores.mean():.4f}")

best_k = max(cv_results, key=cv_results.get)

print(f"Best k based on mean CV accuracy: {best_k}")

# I would choose the k with the highest mean cross-validation
# accuracy because it performed best across the training folds.
# If multiple k values tie, the first one in the list is selected.


# Q3: Compare KNN with and without scaling

knn_unscaled = KNeighborsClassifier(n_neighbors=5)
knn_unscaled.fit(X_train, y_train)

unscaled_predictions = knn_unscaled.predict(X_test)

knn_scaled = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=5))
])

knn_scaled.fit(X_train, y_train)
scaled_predictions = knn_scaled.predict(X_test)

unscaled_accuracy = accuracy_score(y_test, unscaled_predictions)
scaled_accuracy = accuracy_score(y_test, scaled_predictions)

print("\n--- KNN Q3 ---")
print(f"Unscaled test accuracy: {unscaled_accuracy:.4f}")
print(f"Scaled test accuracy: {scaled_accuracy:.4f}")

if scaled_accuracy > unscaled_accuracy:
    print("Scaling improved test accuracy.")
elif scaled_accuracy < unscaled_accuracy:
    print("Scaling reduced test accuracy.")
else:
    print("Scaling did not change test accuracy.")

# Scaling matters for KNN because it calculates distances
# between observations. Features with larger numeric ranges
# can otherwise dominate the distance calculation.
# The printed accuracies show whether scaling helped this dataset.

# --- LOGISTIC REGRESSION ---

# Q1: Build and evaluate Logistic Regression

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("logreg", LogisticRegression(max_iter=1000))
])

pipeline.fit(X_train, y_train)
logreg_predictions = pipeline.predict(X_test)

print("\n--- Logistic Regression Q1 ---")
print(f"Test accuracy: {accuracy_score(y_test, logreg_predictions):.4f}")
print(f"Precision (good): {precision_score(y_test, logreg_predictions):.4f}")
print(f"Recall (good): {recall_score(y_test, logreg_predictions):.4f}")
print(f"F1 (good): {f1_score(y_test, logreg_predictions):.4f}")


# Q2: Examine Logistic Regression coefficients

print("\n--- Logistic Regression Q2 ---")

coefficients = pipeline.named_steps["logreg"].coef_[0]

for feature, coefficient in zip(numeric, coefficients):
    print(f"{feature}: {coefficient:.4f}")

# Negative coefficients push predictions toward "skip" (class 0).
# Positive coefficients push predictions toward "good" (class 1).
# The most negative coefficients indicate the strongest influence
# toward "skip" for a one-standard-deviation feature increase.

# Wind speed has the strongest negative coefficient (-2.4438),
# followed by precipitation (-1.8891). Both features push the
# model toward predicting "skip" as their values increase.
# This matches my intuition because strong winds and heavier
# precipitation make outdoor running less suitable.
# Maximum temperature also has a small negative coefficient
# (-0.1212), while minimum temperature has a positive
# coefficient (0.2563).




# Q3: Predict probabilities for the first five test days

print("\n--- Logistic Regression Q3 ---")

good_class_index = list(pipeline.named_steps["logreg"].classes_).index(1)

probabilities = pipeline.predict_proba(X_test)[:5, good_class_index]

for day_number, probability in enumerate(probabilities, start=1):
    print(f"Test day {day_number}: P(good) = {probability:.2f}")

# A probability near 0.5 means the model is relatively uncertain
# whether the day should be classified as "good" or "skip".

# --- EVALUATION ---

# Q1: Logistic Regression confusion matrix

import os

os.makedirs("outputs", exist_ok=True)

cm = confusion_matrix(y_test, logreg_predictions)

print("\n--- Evaluation Q1 ---")
print("Confusion matrix:")
print(cm)

false_positives = cm[0, 1]
false_negatives = cm[1, 0]

print(f"False positives: {false_positives}")
print(f"False negatives: {false_negatives}")

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["skip", "good"]
)

display.plot(cmap="Blues")
plt.title("Logistic Regression Confusion Matrix")
plt.savefig("outputs/logreg_confusion_matrix.png")
plt.close()

print("Confusion matrix saved to outputs/logreg_confusion_matrix.png")

# False positives classify a bad running day as good.
# False negatives classify a good running day as bad.
# For a running app, false positives are generally more concerning
# because they could encourage someone to run in unsuitable weather.

# The confusion matrix shows 9 false positives and 7 false negatives.
# The 9 false positives represent unsuitable days incorrectly
# predicted as good for running. The 7 false negatives represent
# suitable days incorrectly predicted as skip days.
# False positives are more concerning because the app could
# encourage someone to run in unsuitable weather.
# False negatives could cause missed running opportunities,
# but they pose less risk than false positives.


# --- JOBLIB ---

# Q1: Save and reload the fitted Logistic Regression pipeline

os.makedirs("models", exist_ok=True)

joblib.dump(pipeline, "models/warmup_model.pkl")

print("\n--- joblib Q1 ---")
print("Model saved to models/warmup_model.pkl")

loaded = joblib.load("models/warmup_model.pkl")
assert (loaded.predict(X_test) == pipeline.predict(X_test)).all()
print("Predictions match.")

# Saving only the LogisticRegression estimator would omit the
# StandardScaler fitted during training. Predicting directly on
# raw, unscaled data would then produce unreliable predictions
# because the model expects features on the scaled training-data
# scale. Saving the complete pipeline preserves both steps.

