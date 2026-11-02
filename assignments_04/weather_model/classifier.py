
# Task 1: Prediction Dataclass

from dataclasses import dataclass


@dataclass
class Prediction:
    """Represents a weather prediction for running conditions."""

    label: str
    probability: float


# Tasks 1 and 2: Prediction Dataclass and WeatherClassifier

from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd


@dataclass
class Prediction:
    """Represents a weather prediction for running conditions."""

    label: str
    probability: float


class WeatherClassifier:
    """Loads a saved weather model and predicts running conditions."""

    FEATURES = [
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
        "wind_speed_10m_max",
    ]

    def __init__(self, model_path: str | Path) -> None:
        """
        Load a saved weather classification pipeline.

        Args:
            model_path: Path to the saved model file.

        Raises:
            FileNotFoundError: If the model file does not exist.
        """
        model_path = Path(model_path)

        if not model_path.is_file():
            raise FileNotFoundError(
                f"Weather model file not found: {model_path}"
            )

        self.model = joblib.load(model_path)

    def predict(self, days: list[dict[str, float]]) -> list[Prediction]:
        """
        Predict running conditions for a batch of weather records.

        Args:
            days: Weather dictionaries containing all required features.

        Returns:
            A list of Prediction objects in input order.

        Raises:
            ValueError: If any day is missing required features.
        """
        if not days:
            return []

        for index, day in enumerate(days):
            missing = [
                feature for feature in self.FEATURES
                if feature not in day
            ]

            if missing:
                raise ValueError(
                    f"Day {index + 1} missing required features: "
                    f"{', '.join(missing)}"
                )

        data = pd.DataFrame(days)[self.FEATURES]

        labels = self.model.predict(data)
        probabilities = self.model.predict_proba(data)[:, 1]

        return [
            Prediction(
                label="good" if int(label) == 1 else "skip",
                probability=float(probability),
            )
            for label, probability in zip(labels, probabilities)
        ]
