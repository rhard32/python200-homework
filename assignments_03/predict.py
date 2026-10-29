
# Task 7: Load the trained weather classifier and metadata

import json
from pathlib import Path

import joblib
import pandas as pd

# Locate the saved model and metadata files.
models_dir = Path(__file__).resolve().parent / "models"
model_path = models_dir / "weather_classifier.pkl"
metadata_path = models_dir / "weather_classifier_metadata.json"

# Load the fitted classifier and its metadata.
model = joblib.load(model_path)

with open(metadata_path, "r", encoding="utf-8") as metadata_file:
    metadata = json.load(metadata_file)

# Display information about the trained classifier.
print(f"City: {metadata['city']}")
print(f"Features: {metadata['feature_names']}")
print(f"Test accuracy: {metadata['test_accuracy']:.4f}")


# Create four hypothetical days: good, bad, and borderline conditions.
weather_days = pd.DataFrame(
    [
        [20.0, 10.0, 0.0, 10.0],   # Clearly good
        [32.0, 22.0, 12.0, 45.0], # Clearly bad
        [7.0, 0.0, 2.9, 29.0],    # Borderline good
        [26.5, 5.0, 2.0, 20.0],   # Borderline bad
    ],
    columns=metadata["feature_names"],
)

# Predict whether each day is suitable for running.
predictions = model.predict(weather_days)
probabilities = model.predict_proba(weather_days)[:, 1]

# Display inputs, predicted labels, and confidence scores.
print("\n--- Weather Predictions ---")
print(weather_days.to_string(index=False))

for day_number, (prediction, probability) in enumerate(
    zip(predictions, probabilities), start=1
):
    label = "good" if prediction == 1 else "skip"
    print(
        f"Day {day_number}: {label} "
        f"(P(good) = {probability:.2f})"
    )

# Task 8: Reflection
#
# The borderline examples show that model predictions do not
# always agree with the original labeling rules.
# Day 3 received P(good) = 0.33 and was classified as skip,
# although its weather values satisfy the labeling thresholds.
# Day 4 received P(good) = 1.00 and was classified as good,
# although its maximum temperature exceeds 26 C.
#
# If a borderline day received P(good) = 0.52, I would treat
# the prediction as uncertain rather than confidently recommend
# running. The app could display a caution message or require
# additional weather checks before making a recommendation.
#
# Running predict.py before training would fail because the
# saved model and metadata files would not exist. A clearer
# error message could be provided by checking whether both
# files exist and instructing the user to run train_classifier.py
# first.
#
# For an automated Week 10 cloud pipeline, I would add
# command-line arguments for input data and model paths so
# the prediction script could run without manual code changes.
