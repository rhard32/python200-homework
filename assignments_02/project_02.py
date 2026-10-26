
# Mini-Project: Predicting Daily High Temperatures
# Task 1: Fetch one year of historical weather data

import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Ypsilanti, Michigan
latitude = 42.2411
longitude = -83.6130

url = "https://archive-api.open-meteo.com/v1/archive"

params = {
    "latitude": latitude,
    "longitude": longitude,
    "start_date": "2023-01-01",
    "end_date": "2023-12-31",
    "daily": [
        "temperature_2m_min",
        "temperature_2m_max",
        "precipitation_sum",
        "wind_speed_10m_max",
    ],
    "temperature_unit": "celsius",
    "wind_speed_unit": "kmh",
    "timezone": "America/Detroit",
}

response = requests.get(url, params=params, timeout=30)
response.raise_for_status()

daily = response.json()["daily"]
df = pd.DataFrame(daily)

df["time"] = pd.to_datetime(df["time"])

print("--- Task 1: Fetch Weather Data ---")
print("City: Ypsilanti, Michigan")
print("Number of days:", len(df))
print(df.head())
print("\nMissing values:")
print(df.isna().sum())


# Task 2: Explore the dataset

print("\n--- Task 2: Explore the Data ---")

# Summarize the numeric weather measurements.
print("\nDescriptive statistics:")
print(df.describe())

# Confirm whether missing values need handling.
print("\nMissing values by column:")
print(df.isna().sum())

# No rows need to be removed or filled because Task 1
# confirmed that every column has zero missing values.

# Examine correlations with daily maximum temperature.
correlations = (
    df.select_dtypes(include="number")
    .corr()["temperature_2m_max"]
    .drop("temperature_2m_max")
    .sort_values(ascending=False)
)

print("\nCorrelations with maximum temperature:")
print(correlations)

print("\nStrongest correlated feature:", correlations.abs().idxmax())

# Explore the relationship between daily minimum and maximum temperatures.
plt.figure(figsize=(8, 5))
plt.scatter(
    df["temperature_2m_min"],
    df["temperature_2m_max"],
    alpha=0.6,
)

plt.title("Daily Maximum vs Minimum Temperature - Ypsilanti 2023")
plt.xlabel("Minimum Temperature (°C)")
plt.ylabel("Maximum Temperature (°C)")
plt.tight_layout()
plt.savefig("assignments_02/outputs/weather_exploration.png")
plt.close()

print("\nSaved: outputs/weather_exploration.png")

# The scatter plot helps reveal whether warmer daily minimum
# temperatures generally correspond to warmer daily maximums.
# The correlation results quantify the strength of that relationship.

# Task 3: Feature Engineering

print("\n--- Task 3: Feature Engineering ---")

# Create a binary indicator for summer months:
# June, July, and August = 1; all other months = 0.
df["is_summer"] = df["time"].dt.month.isin([6, 7, 8]).astype(int)

print("\nSummer indicator counts:")
print(df["is_summer"].value_counts().sort_index())

print("\nFirst five dates with summer indicator:")
print(df[["time", "is_summer"]].head())

# Summer is expected to have a positive relationship with
# daily maximum temperature because these months are generally
# warmer in Ypsilanti, Michigan.
# This feature allows the full regression model to account
# for seasonal differences.


# Task 4: Baseline Linear Regression

print("\n--- Task 4: Baseline Regression ---")

# Use minimum temperature as the only predictor.
X_baseline = df[["temperature_2m_min"]]
y = df["temperature_2m_max"]

# Reserve 20% of the data for testing.
X_train_base, X_test_base, y_train_base, y_test_base = train_test_split(
    X_baseline,
    y,
    test_size=0.20,
    random_state=42,
)

# Train the baseline model.
baseline_model = LinearRegression()
baseline_model.fit(X_train_base, y_train_base)

# Predict maximum temperatures for the test data.
y_pred_base = baseline_model.predict(X_test_base)

# Evaluate the predictions.
baseline_rmse = np.sqrt(mean_squared_error(y_test_base, y_pred_base))
baseline_r2 = r2_score(y_test_base, y_pred_base)

print("Training rows:", len(X_train_base))
print("Testing rows:", len(X_test_base))
print("Slope:", baseline_model.coef_[0])
print("Intercept:", baseline_model.intercept_)
print("Test RMSE:", baseline_rmse)
print("Test R²:", baseline_r2)

# The slope measures the predicted increase in daily maximum
# temperature for each 1°C increase in minimum temperature.
#
# RMSE measures the typical prediction error in degrees Celsius.
# For example, an RMSE of 3 means predictions are typically
# about 3°C away from the actual maximum temperatures.
#
# R² measures how well the model explains variation in the
# daily maximum temperatures of the test dataset.

# Task 5: Full Regression Model

print("\n--- Task 5: Full Regression Model ---")

features = [
    "temperature_2m_min",
    "precipitation_sum",
    "wind_speed_10m_max",
    "is_summer",
]

X_full = df[features]

X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
    X_full,
    y,
    test_size=0.20,
    random_state=42,
)

full_model = LinearRegression()
full_model.fit(X_train_full, y_train_full)

y_pred_full = full_model.predict(X_test_full)

full_train_r2 = full_model.score(X_train_full, y_train_full)
full_test_r2 = r2_score(y_test_full, y_pred_full)
full_rmse = np.sqrt(mean_squared_error(y_test_full, y_pred_full))

print("Training R²:", full_train_r2)
print("Test R²:", full_test_r2)
print("Test RMSE:", full_rmse)

print("\nModel comparison:")
print("Baseline test R²:", baseline_r2)
print("Full model test R²:", full_test_r2)
print("R² difference:", full_test_r2 - baseline_r2)

print("\nFeature coefficients:")
for feature, coefficient in zip(features, full_model.coef_):
    print(f"{feature}: {coefficient:.4f}")

print("\nTrain-test R² gap:", full_train_r2 - full_test_r2)

# Each coefficient represents the predicted change in maximum
# temperature for a one-unit increase in that feature, while
# holding the other features constant.
#
# A positive coefficient increases predicted maximum temperature.
# A negative coefficient decreases predicted maximum temperature.
#
# Compare baseline and full-model test R² to determine whether
# the additional features improve performance on unseen data.
#
# A large difference between training and test R² can indicate
# that the model does not generalize as well to unseen data.
#
# Coefficient signs may differ from expectations because
# weather features can be correlated with one another.


# Task 6: Predicted vs Actual Plot and Final Interpretation

print("\n--- Task 6: Final Evaluation ---")

plt.figure(figsize=(8, 5))

plt.scatter(
    y_pred_full,
    y_test_full,
    alpha=0.7,
    label="Test predictions",
)

plot_min = min(y_pred_full.min(), y_test_full.min())
plot_max = max(y_pred_full.max(), y_test_full.max())

plt.plot(
    [plot_min, plot_max],
    [plot_min, plot_max],
    "r--",
    label="Perfect prediction",
)

plt.title("Predicted vs Actual (Full Model)")
plt.xlabel("Predicted High Temperature (°C)")
plt.ylabel("Actual High Temperature (°C)")
plt.legend()
plt.tight_layout()

plt.savefig("assignments_02/outputs/predicted_vs_actual_high.png")
plt.close()

print("Saved: outputs/predicted_vs_actual_high.png")

print("\nFinal project summary:")
print("Dataset: 365 daily weather records from Ypsilanti, MI (2023)")
print("Test set: 73 days")
print(f"Full-model test RMSE: {full_rmse:.2f}°C")
print(f"Full-model test R²: {full_test_r2:.4f}")
print(f"Baseline test R²: {baseline_r2:.4f}")

# FINAL INTERPRETATION
#
# The dataset contains 365 days of weather measurements from
# Ypsilanti, Michigan, during 2023. We evaluated the model on
# 73 days that were held out from training.
#
# The full model achieved an RMSE of approximately 3.62°C,
# meaning its predictions have a typical error magnitude
# of approximately 3.62 degrees Celsius.
#
# Its test R² was approximately 0.872, meaning the model
# explained about 87.2% of the variation in test temperatures.
#
# Minimum temperature was the strongest predictor by correlation
# (approximately 0.937 with maximum temperature).
# Its positive coefficient indicates that higher minimum
# temperatures predict higher maximum temperatures.
#
# The summer indicator had a positive coefficient of 1.44,
# suggesting higher predicted maximum temperatures during summer,
# after accounting for the other model features.
#
# A surprising result was that the full model performed
# slightly worse than the baseline on the test dataset.
# Baseline R² was 0.887, compared with 0.872 for the full model.
# Adding features does not guarantee improved generalization.
#
# The train-test R² gap was approximately 0.015,
# which does not suggest a large generalization gap.
#
# In the predicted-versus-actual plot:
# Points above the diagonal represent underpredictions.
# Points below the diagonal represent overpredictions.
# The distance from the diagonal shows the prediction error.
#
# The plot can be inspected to compare error spread across
# lower and higher maximum temperatures.
