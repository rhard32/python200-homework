from pydantic import BaseModel, model_validator, Field
import json

# Part 2 - Task 2: The Boundary - Pydantic Schema Validation

class HourlyBlock(BaseModel):
    """Represent and validate parallel hourly weather data arrays."""

    time: list[str]
    temperature_2m: list[float]
    precipitation: list[float]
    
    
    @model_validator(mode="after")

    def validate_lengths(self):
        """Ensure the hourly time, temperature, and precipitation arrays have equal lengths."""
        if not (len(self.time) == len(self.temperature_2m) == len(self.precipitation)):
            raise ValueError("All lists must have the same length")
        return self

class WeatherResponse(BaseModel):
    """Represent and validate a weather API response."""
    latitude: float = Field( ge=-90, le=90)
    longitude: float = Field( ge=-180, le=180)
    timezone: str
    elevation: float
    hourly: HourlyBlock


if __name__ == "__main__":


    with open("assignments_01/weather_raw.json", "r") as file:
        weather_data = json.load(file)

    weather = WeatherResponse.model_validate(weather_data)

    print("Latitude:", weather.latitude)
    print("Timezone:", weather.timezone)
    print("Hourly readings:", len(weather.hourly.time))

