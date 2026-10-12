
import pytest

from weatherkit.records import HourlyReading, to_readings
from weatherkit.schemas import WeatherResponse


@pytest.fixture
def sample_response() -> WeatherResponse:
    """Provide a small validated weather response for testing."""
    data = {
        "latitude": 35.2,
        "longitude": -80.8,
        "timezone": "GMT",
        "elevation": 200.0,
        "hourly": {
            "time": [
                "2026-04-08T00:00",
                "2026-04-08T01:00",
                "2026-04-08T02:00",
            ],
            "temperature_2m": [10.0, 12.5, 9.0],
            "precipitation": [0.0, 0.5, 1.0],
        },
    }
    return WeatherResponse.model_validate(data)


def test_reading_count_and_order(sample_response):
    """Conversion preserves the number and order of observations."""
    readings = to_readings(sample_response)

    assert len(readings) == len(sample_response.hourly.time)
    assert readings[0].timestamp == "2026-04-08T00:00"
    assert readings[-1].timestamp == "2026-04-08T02:00"


def test_reading_values_match_input_index(sample_response):
    """Each reading keeps measurements paired with its timestamp."""
    readings = to_readings(sample_response)

    for i, reading in enumerate(readings):
        assert reading.timestamp == sample_response.hourly.time[i]
        assert reading.temperature_c == sample_response.hourly.temperature_2m[i]
        assert reading.precipitation_mm == sample_response.hourly.precipitation[i]


def test_hourly_reading_dataclass_equality():
    """Dataclasses with identical field values compare equal."""
    first = HourlyReading(
        timestamp="2026-04-08T00:00",
        temperature_c=10.0,
        precipitation_mm=0.0,
    )
    second = HourlyReading(
        timestamp="2026-04-08T00:00",
        temperature_c=10.0,
        precipitation_mm=0.0,
    )

    assert first == second
