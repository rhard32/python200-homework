
# Warmup Question 1: ModelInfo

import json
from pathlib import Path


class ModelInfo:
    def __init__(self, path):
        with open(path, "r", encoding="utf-8") as file:
            self.data = json.load(file)

    def feature_count(self) -> int:
        return len(self.data["features"])


# Create a sample JSON file.
sample_path = Path(__file__).parent / "sample_model_info.json"

with open(sample_path, "w", encoding="utf-8") as file:
    json.dump(
        {"features": [
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_sum",
            "wind_speed_10m_max",
        ]},
        file,
    )

model_info = ModelInfo(sample_path)
print("Feature count:", model_info.feature_count())

# Loading JSON once in __init__ avoids repeatedly reading the
# file from disk whenever feature_count() is called.


# Warmup Question 2: DayForecast Dataclass

from dataclasses import dataclass


@dataclass
class DayForecast:
    date: str
    high_c: float
    will_rain: bool

    def summary(self) -> str:
        rain_status = "rain expected" if self.will_rain else "no rain expected"
        return f"{self.date}: high {self.high_c:.1f}C, {rain_status}"


forecast1 = DayForecast("2023-06-15", 24.1, True)
forecast2 = DayForecast("2023-06-16", 27.5, False)

print(forecast1.summary())
print(forecast2.summary())

# The @dataclass decorator automatically generates the __init__ method,
# so we do not need to write it manually.


# Warmup Question 3: Temperature Parsing and pytest Tests

import pytest


def parse_temperature(text: str) -> float:
    try:
        return float(text)
    except ValueError:
        raise ValueError(f"Invalid temperature: {text}") from None


def test_parse_temperature_valid():
    assert parse_temperature("24.1") == pytest.approx(24.1)


def test_parse_temperature_bad_input_raises():
    with pytest.raises(ValueError, match="temperature"):
        parse_temperature("hot")


# Using pytest.raises(ValueError) without match would only verify
# the exception type. It would not catch an incorrect error message.
