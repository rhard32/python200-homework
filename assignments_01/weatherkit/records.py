from dataclasses import dataclass
from .schemas import WeatherResponse

# Part 2 - Task 3: Inside the Boundary

@dataclass
class HourlyReading:
    """Represent one hourly weather observation.

    Temperature is measured in degrees Celsius.
    Precipitation is measured in millimeters.
    """
    timestamp: str
    temperature_c: float
    precipitation_mm: float


def to_readings(response: WeatherResponse) -> list[HourlyReading]:
    """Convert a validated weather response into hourly readings.

    Args:
    response: A validated WeatherResponse object.

    Returns:
    A list of HourlyReading objects.
    """

    readings = []
    for i in range(len(response.hourly.time)):
        reading = HourlyReading(
            timestamp=response.hourly.time[i],
            temperature_c=response.hourly.temperature_2m[i],
            precipitation_mm=response.hourly.precipitation[i]
        )
        readings.append(reading)
    return readings

# WeatherResponse uses Pydantic to validate untrusted data at the API boundary.
# HourlyReading uses a dataclass because it represents data already validated
# and converted into a simpler format for internal processing.
   