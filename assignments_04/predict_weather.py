
# Task 5: Predict Weather Using the Saved Model

from pathlib import Path

from weather_model import WeatherClassifier


def main() -> None:
    """Load the saved model and predict running conditions."""

    model_path = (
        Path(__file__).parent
        / "models"
        / "weather_classifier.pkl"
    )

    classifier = WeatherClassifier(model_path)

    days = [
        {
            "temperature_2m_max": 26.0,
            "temperature_2m_min": 16.0,
            "precipitation_sum": 0.0,
            "wind_speed_10m_max": 5.0,
        },
        {
            "temperature_2m_max": 3.0,
            "temperature_2m_min": -4.0,
            "precipitation_sum": 18.0,
            "wind_speed_10m_max": 38.0,
        },
        {
            "temperature_2m_max": 17.0,
            "temperature_2m_min": 9.0,
            "precipitation_sum": 2.0,
            "wind_speed_10m_max": 14.0,
        },
        {
            "temperature_2m_max": 21.0,
            "temperature_2m_min": 12.0,
            "precipitation_sum": 0.5,
            "wind_speed_10m_max": 12.0,
        },
    ]

    predictions = classifier.predict(days)

    for index, (day, prediction) in enumerate(
        zip(days, predictions), start=1
    ):
        print(
            f"Day {index}: {day} | "
            f"Label: {prediction.label} | "
            f"Probability of good running weather: "
            f"{prediction.probability:.2%}"
        )


# Without this guard, importing predict_weather.py from another
# script would automatically execute main() and print predictions.
if __name__ == "__main__":
    main()


# Task 6: Reflection

# Question 1:
# Selecting features in a fixed order prevents incorrect predictions
# caused by mismatched column positions. For example, if a caller
# supplies precipitation_sum before temperature_2m_max, using the
# caller's order could cause the model to interpret rainfall as
# temperature. Enforcing the training feature order prevents this bug.

# Question 2:
# Training a small model inside a pytest fixture makes the tests
# independent of the real saved model file. The tests remain
# repeatable and can run even if the production model is replaced,
# moved, or unavailable. This also isolates the component's
# behavior from changes to the trained model.

# Question 3:
# The component is ready for batch processing because predict()
# accepts multiple weather records and returns structured Prediction
# objects in the same order.
#
# Before production deployment, I would add logging to record
# prediction failures and help diagnose problems in the pipeline.
