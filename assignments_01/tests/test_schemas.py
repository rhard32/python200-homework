
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from weatherkit.schemas import HourlyBlock, WeatherResponse


def test_valid_weather_response():
    """The real API response should validate with 168 hourly observations."""

    # Resolve the path relative to this test file because a plain relative
    # path depends on the directory pytest was launched from.
    json_path = Path(__file__).parent.parent / "weather_raw.json"

    with json_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    response = WeatherResponse.model_validate(data)

    assert len(response.hourly.time) == 168


def test_invalid_latitude():
    """Latitude outside the valid range should fail validation."""

    data = {
        "latitude": 200.0,
        "longitude": -80.0,
        "timezone": "GMT",
        "elevation": 200.0,
        "hourly": {
            "time": ["2026-04-08T00:00"],
            "temperature_2m": [12.0],
            "precipitation": [0.0],
        },
    }

    with pytest.raises(ValidationError):
        WeatherResponse.model_validate(data)


def test_mismatched_hourly_lengths():
    """Parallel hourly arrays must have matching lengths."""

    data = {
        "time": [
            "2026-04-08T00:00",
            "2026-04-08T01:00",
            "2026-04-08T02:00",
        ],
        "temperature_2m": [12.0, 13.0],
        "precipitation": [0.0, 0.1, 0.0],
    }

    with pytest.raises(ValidationError):
        HourlyBlock.model_validate(data)


def test_null_temperature():
    """A null temperature should fail strict float validation."""

    data = {
        "time": [
            "2026-04-08T00:00",
            "2026-04-08T01:00",
        ],
        "temperature_2m": [12.0, None],
        "precipitation": [0.0, 0.1],
    }

    with pytest.raises(ValidationError):
        HourlyBlock.model_validate(data)
