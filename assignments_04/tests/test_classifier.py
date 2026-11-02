
# Task 4: WeatherClassifier Test Suite

import joblib
import pandas as pd
import pytest

from sklearn.linear_model import LogisticRegression

from weather_model import Prediction, WeatherClassifier


FEATURES = [
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "wind_speed_10m_max",
]


@pytest.fixture
def classifier(tmp_path):
    """Train and save a tiny model for independent testing."""
    training_data = pd.DataFrame(
        [
            [25, 15, 0, 5],
            [28, 18, 0, 4],
            [23, 13, 0, 7],
            [30, 20, 0, 3],
            [5, -2, 15, 35],
            [8, 1, 12, 30],
            [2, -5, 20, 40],
            [10, 3, 10, 25],
        ],
        columns=FEATURES,
    )

    labels = [1, 1, 1, 1, 0, 0, 0, 0]

    model = LogisticRegression(max_iter=1000)
    model.fit(training_data, labels)

    model_path = tmp_path / "test_weather_classifier.pkl"
    joblib.dump(model, model_path)

    return WeatherClassifier(model_path)


@pytest.mark.parametrize(
    "day, expected_label",
    [
        (
            {
                "temperature_2m_max": 28,
                "temperature_2m_min": 18,
                "precipitation_sum": 0,
                "wind_speed_10m_max": 4,
            },
            "good",
        ),
        (
            {
                "temperature_2m_max": 2,
                "temperature_2m_min": -5,
                "precipitation_sum": 20,
                "wind_speed_10m_max": 40,
            },
            "skip",
        ),
    ],
)
def test_prediction_labels(classifier, day, expected_label):
    prediction = classifier.predict([day])[0]
    assert prediction.label == expected_label


def test_probability_between_zero_and_one(classifier):
    day = dict(zip(FEATURES, [25, 15, 0, 5]))
    prediction = classifier.predict([day])[0]

    assert 0 <= prediction.probability <= 1


def test_three_predictions_in_order(classifier):
    days = [
        dict(zip(FEATURES, [28, 18, 0, 4])),
        dict(zip(FEATURES, [2, -5, 20, 40])),
        dict(zip(FEATURES, [25, 15, 0, 5])),
    ]

    predictions = classifier.predict(days)

    assert len(predictions) == 3
    assert all(isinstance(p, Prediction) for p in predictions)
    assert [p.label for p in predictions] == ["good", "skip", "good"]


def test_empty_list_returns_empty_list(classifier):
    assert classifier.predict([]) == []


def test_missing_feature_raises_value_error(classifier):
    day = {
        "temperature_2m_max": 25,
        "temperature_2m_min": 15,
        "precipitation_sum": 0,
    }

    with pytest.raises(ValueError, match="wind_speed_10m_max"):
        classifier.predict([day])


def test_missing_model_file_raises_file_not_found(tmp_path):
    missing_path = tmp_path / "missing_model.pkl"

    with pytest.raises(FileNotFoundError, match="not found"):
        WeatherClassifier(missing_path)


# Mutation test:
# Temporarily changed WeatherClassifier.predict() to always return "skip".
# The following tests detected the defect:
# 1. test_prediction_labels[day0-good]
# 2. test_three_predictions_in_order
# Result: 2 failed, 5 passed.
# Restored the original label logic after the mutation test.

