
# --- scikit-learn API ---
# Q1: Create, fit, and predict using LinearRegression

import numpy as np
from sklearn.linear_model import LinearRegression

temp_min = np.array([1, 4, 8, 12, 16, 20]).reshape(-1, 1)
temp_max = np.array([8, 10, 15, 18, 23, 27])

# Create the regression model.
model = LinearRegression()

# Train the model using the known low and high temperatures.
model.fit(temp_min, temp_max)

# Predict high temperatures for lows of 6°C and 18°C.
predictions = model.predict(np.array([[6], [18]]))

# Display the regression coefficients and predictions.
print("--- scikit-learn Question 1 ---")
print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)
print("Predicted high for low of 6°C:", predictions[0])
print("Predicted high for low of 18°C:", predictions[1])

# Q2: Reshape a 1D array into a 2D feature matrix

x = np.array([10, 20, 30, 40, 50])

print("\n--- scikit-learn Question 2 ---")
print("Original shape:", x.shape)

x_2d = x.reshape(-1, 1)
print("Reshaped shape:", x_2d.shape)

# scikit-learn expects X to be 2D because each row represents
# one observation and each column represents one feature.
# Here, we have five observations and one feature.

# --- Linear Regression ---
# Shared synthetic weather dataset for Questions 1–5

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

rng = np.random.default_rng(42)
num_days = 120
temp_min = rng.uniform(-5, 22, num_days)
is_rainy = rng.integers(0, 2, num_days).astype(float)
temp_max = 1.0 * temp_min + 6 - 4 * is_rainy + rng.normal(0, 2, num_days)

# Linear Regression Q1: Visualize daily high vs low temperatures

plt.figure(figsize=(8, 5))
plt.scatter(temp_min, temp_max, c=is_rainy, cmap="coolwarm")
plt.title("Daily High vs Low")
plt.xlabel("Minimum Temperature (°C)")
plt.ylabel("Maximum Temperature (°C)")
plt.colorbar(label="Rain Status (0 = Dry, 1 = Rainy)")
plt.tight_layout()
plt.savefig("assignments_02/outputs/high_vs_low.png")
plt.close()

print("\n--- Linear Regression Question 1 ---")
print("Saved: outputs/high_vs_low.png")

# The scatter plot should show two overlapping groups.
# Rainy days generally have lower high temperatures than dry days
# with similar minimum temperatures. This suggests that is_rainy
# is a useful feature for predicting the daily high.

# Linear Regression Q2: Train/test split using minimum temperature

X = temp_min.reshape(-1, 1)
y = temp_max

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

print("\n--- Linear Regression Question 2 ---")
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

# Linear Regression Q3: Fit and evaluate the baseline model

model_baseline = LinearRegression()
model_baseline.fit(X_train, y_train)

y_pred = model_baseline.predict(X_test)

rmse_baseline = np.sqrt(mean_squared_error(y_test, y_pred))
r2_baseline = model_baseline.score(X_test, y_test)

print("\n--- Linear Regression Question 3 ---")
print("Slope:", model_baseline.coef_[0])
print("Intercept:", model_baseline.intercept_)
print("Test RMSE:", rmse_baseline)
print("Test R²:", r2_baseline)

# The slope represents the predicted change in daily high temperature
# for every 1°C increase in minimum temperature.
# This model does not account for whether the day is rainy.

# Linear Regression Q4: Add rain status as a second feature

X_full = np.column_stack([temp_min, is_rainy])

X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
    X_full, temp_max, test_size=0.20, random_state=42
)

model_full = LinearRegression()
model_full.fit(X_train_full, y_train_full)

r2_full = model_full.score(X_test_full, y_test_full)

print("\n--- Linear Regression Question 4 ---")
print("Baseline test R²:", r2_baseline)
print("Two-feature test R²:", r2_full)
print("R² improvement:", r2_full - r2_baseline)
print("temp_min coefficient:", model_full.coef_[0])
print("is_rainy coefficient:", model_full.coef_[1])

# The is_rainy coefficient represents the predicted difference
# between rainy and dry days at the same minimum temperature.
# A negative coefficient means rain lowers the predicted daily high.
# Comparing test R² values shows whether adding rain improves
# predictions on unseen data.

# Linear Regression Q5: Predicted vs Actual plot

y_pred_full = model_full.predict(X_test_full)

plt.figure(figsize=(8, 5))
plt.scatter(y_pred_full, y_test_full, alpha=0.7)

# Draw the diagonal reference line for perfect predictions.
plot_min = min(y_pred_full.min(), y_test_full.min())
plot_max = max(y_pred_full.max(), y_test_full.max())

plt.plot(
    [plot_min, plot_max],
    [plot_min, plot_max],
    "r--",
    label="Perfect Prediction"
)

plt.title("Predicted vs Actual")
plt.xlabel("Predicted High Temperature (°C)")
plt.ylabel("Actual High Temperature (°C)")
plt.legend()
plt.tight_layout()
plt.savefig("assignments_02/outputs/predicted_vs_actual_warmup.png")
plt.close()

print("\n--- Linear Regression Question 5 ---")
print("Saved: outputs/predicted_vs_actual_warmup.png")

# Points above the diagonal indicate underprediction:
# the actual temperature was higher than predicted.
# Points below the diagonal indicate overprediction:
# the actual temperature was lower than predicted.
