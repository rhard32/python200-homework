from dataclasses import dataclass, field, FrozenInstanceError
from pydantic import BaseModel, Field, ValidationError, model_validator
import pytest



#Classes Question 1

class Thermometer:
    def __init__(self, location: str, readings: list | None = None):
        self.location = location
        if readings is None:
            self.readings = []
        else:
            self.readings = readings

    def add(self, reading: float):
        self.readings.append(reading)

# average() needs to handle an empty list because otherwise a ZeroDivisionError will be raised.

    def average(self) -> float |None:
        
        if len(self.readings) == 0:
            return None
        else:
            return sum(self.readings) / len(self.readings)

    def hottest(self) -> float | None:
        if len(self.readings) == 0:
            return None
        else:
            return max(self.readings)

    #Classes Question 2
    #Without the __repr__ method, the output would give that the class was defined in the script being run,
    # the object's class and the object's memory address. The __repr__ method is helpful in debugging because it provides
    #a more informative and human readable representation of the object, which can help developers understand the state of the object.

    def __repr__(self) -> str:
        return f"Thermometer(location= '{self.location}', n_readings={len(self.readings)}, average= {self.average()})"

#Classes Question 3
class TemperatureAlert:
    def __init__(self, threshold: float= 30.0):
        self.threshold = threshold

    def breaches(self, thermometer: Thermometer) -> list:
        breached_readings = []
        for reading in thermometer.readings:
            if reading > self.threshold:
                breached_readings.append(reading)
        return breached_readings

#Storing the threshold as an attribute of TemperatureAlert allows each object to maintain its own
#threshold. This makes checking multiple thermometers more efficient because the same alert object and
#threshold can be reused without repeatedly passing the threshold as an argument.

breach_alert1 = TemperatureAlert()
breach_alert2 = TemperatureAlert(25.0)


city_thermometer = Thermometer("Detroit")
city_thermometer2 = Thermometer("New York")

city_thermometer.add(22.0)
city_thermometer.add(32.0)
city_thermometer.add(24.0)
city_thermometer.add(30.0)

city_thermometer2.add(20.0)
city_thermometer2.add(25.0)
city_thermometer2.add(30.0)
city_thermometer2.add(35.0)

print(breach_alert1.breaches(city_thermometer))
print(breach_alert2.breaches(city_thermometer))

print(f"Average temperature in {city_thermometer.location}: {city_thermometer.average()}°C")
print(f"Hottest temperature in {city_thermometer.location}: {city_thermometer.hottest()} °C")
print(city_thermometer)

print(f"Average temperature in {city_thermometer2.location}: {city_thermometer2.average()}°C")
print(f"Hottest temperature in {city_thermometer2.location}: {city_thermometer2.hottest()} °C")
print([city_thermometer, city_thermometer2])

# --- Dataclasses, Type Hints, and Docstrings ---
#Dataclasses Question 1

@dataclass(frozen=True)
class Station:
    """Represents a weather monitoring station and its location."""
    station_id: str
    name: str
    latitude: float
    longitude: float
    elevation: float
    

station1 = Station("ST001", "Station 1", 42.3314, -83.0458, 200.0)
station2 = Station("ST001", "Station 1", 42.3314, -83.0458, 200.0)
station3 = Station("ST002", "Station 2", 40.7128, -74.0060, 10.0)

print(station1 == station2)
# The original class would return False because it compares object identity
# rather than field values. The @dataclass decorator automatically generates
# an __eq__() method, allowing the values of the two objects to be compared.

#Dataclasses Question 2


try:
    station1.name = "Updated Station"
except FrozenInstanceError as e:
    print(e)

stations = {station1, station2, station3}
print(len(stations))

# Making Station frozen allows its instances to be hashable because
# all their fields are hashable. This allows Station objects to be
# stored in a set. Frozen also prevents fields from being reassigned,
# protecting data integrity. This is especially important for
# latitude, longitude, and elevation, where unintended changes
# could lead to inaccurate data.

# Dataclasses Question 3

@dataclass
class StationBatch:
    """Represents a collection of weather stations within a region."""
    region: str
    
    # Using stations: list[Station] = [] raises a ValueError because
    # a mutable list cannot be used directly as a dataclass default.
    # default_factory=list creates a new list for each StationBatch instance.
    stations: list[Station] = field(default_factory=list)
   

    def add(self, station: Station):
        """Adds a Station object to the stations list."""
        self.stations.append(station)

    def highest(self) -> Station | None:
        """Return the highest-elevation station, or None if the batch is empty."""
        if not self.stations:
            return None
        return max(self.stations, key=lambda station: station.elevation)
    
batch1 = StationBatch("Midwest")
batch1.add(station1)
batch1.add(station3)
batch2 = StationBatch("Northeast")

print(batch2.highest())
print(batch1.highest())
    
# --- Pydantic ---
# Question 1


class Reading(BaseModel):
    """Represents a weather reading with validated data."""
    station_id: str = Field(min_length=3)
    timestamp: str
    temperature_c: float = Field(ge=-90, le=60)
    humidity : float = Field(ge=0, le=100)
# Field constraints validate individual fields independently.
# A model_validator is needed to check a condition involving
# both temperature_c and humidity together.

    #Pydantic Question 4 

    @model_validator(mode="after")

    def validate_conditions(self):
        """Validate the relationship between temperature and humidity."""
        if self.temperature_c < -40 and self.humidity == 0.0:
            raise ValueError("Low temperature readings cannot have 0.0 humidity.")
        return self

reading1 = Reading(station_id="St001", timestamp="2026-10-10T10:00:00", temperature_c=25.5, humidity=65.0)

print(reading1)

try:
    reading2 = Reading(station_id="St002", timestamp="2026-10-10T11:00:00", temperature_c=25.5, humidity=110.0)

except ValidationError as e:
    print(e)

try:
    reading3 = Reading(station_id="St", temperature_c=25.5, humidity=50.0)

except ValidationError as e:
    print(e)

    
#Pydantic reported two validation errors for reading3: 
#one for the station_id being too short and another for the missing timestamp field.

#Pydantic Question 2


reading4 = Reading(station_id="St004", timestamp="2026-10-10T12:00:10", temperature_c="25.5", humidity=50)
print(reading4.temperature_c)
print(type(reading4.temperature_c))
print(type(reading4.humidity))

reading_int = Reading(
    station_id="St009",
    timestamp="2026-10-10T15:00:00",
    temperature_c=40,
    humidity=50.0
)
print(type(reading_int.temperature_c))

try:
    reading5 = Reading(station_id="St005", timestamp="2026-10-10T12:00:20", temperature_c="warm", humidity="very humid")

except ValidationError as e:
    print(e)
# Pydantic raises two validation errors for reading5:
# temperature_c="warm" cannot be converted to a float.
# humidity="very humid" cannot be converted to a float.
# Both errors demonstrate Pydantic's validation of numeric fields.

# Pydantic Question 3

try:
    reading6 = Reading(station_id="St", temperature_c="warm", humidity=50.0)
except ValidationError as e:
    for error in e.errors():
        print(error["loc"][0], error["msg"])
# e.errors() returns a list of dictionaries, each containing details
# about a validation error.
# My code iterates through the list and prints the field name
# (from the "loc" key) and the error message (from the "msg" key).
# Reporting multiple errors at once allows developers to correct all identified issues in one pass.

#Pydantic Question 4 testing section

try:
    reading7 = Reading(station_id="St007", timestamp="2026-10-10T13:00:30", temperature_c=-45.0, humidity=0.0)
except ValidationError as e:
    print(e)

reading8 = Reading(
    station_id="ST008",
    timestamp="2026-10-10T14:00:00",
    temperature_c=-45.0,
    humidity=25.0
)
print(reading8)

# --- pytest ---
# Question 1

def celsius_to_fahrenheit(celsius: float) -> float:
    """Convert Celsius to Fahrenheit."""
    return (celsius * 9/5) + 32

def test_celsius_to_fahrenheit():
    assert celsius_to_fahrenheit(0) == 32
    assert celsius_to_fahrenheit(100) == 212

# Use pytest.approx() to account for small floating-point rounding
# differences that can occur during calculations.s
    assert celsius_to_fahrenheit(37) == pytest.approx(98.6)

# pytest Question 2

def mean(values: list[float]) -> float:
    """Calculate the arithmetic mean of a list of numbers."""
    if not values:
        raise ValueError("Cannot compute mean of an empty list.")
    return sum(values) / len(values)


def test_mean_empty_raises():
    with pytest.raises(ValueError, match="Cannot compute mean of an empty list"):
        mean([])
# pytest.raises(ValueError) alone only checks that a ValueError is raised.
# Using match= also checks that the error message contains the expected text,
# helping catch an incorrect ValueError raised for a different reason.

# pytest Question 3

@pytest.mark.parametrize(
    "values, expected",
    [
        ([10], 10),
        ([10, 20, 30], 20),
        ([-10, -20, -30], -20),
        ([5, 10, 15, 20], 12.5),
    ]
)
def test_mean_values(values, expected):
    assert mean(values) == pytest.approx(expected)
# Parameterization is more efficient because it reduces duplicate code.
# It allows multiple test cases to use the same test logic, making
# the tests easier to maintain and expand.
# pytest Question 3 - Test execution results
# Command: python -m pytest warmup_01.py::test_mean_values -v
#
# warmup_01.py::test_mean_values[values0-10] PASSED
# warmup_01.py::test_mean_values[values1-20] PASSED
# warmup_01.py::test_mean_values[values2--20] PASSED
# warmup_01.py::test_mean_values[values3-12.5] PASSED
#
# ========================= 4 passed in 0.28s =========================

# pytest Question 4


# Deliberate failure output (changed +32 to +30):
# FAILED test_celsius_to_fahrenheit: assert 30.0 == 32
# FAILED parametrized[0-32]: Obtained 30.0, Expected 32
# FAILED parametrized[100-212]: Obtained 210.0, Expected 212
# FAILED parametrized[37-98.6]: Obtained 96.6, Expected 98.6
# FAILED parametrized[-40--40]: Obtained -42.0, Expected -40
# Test summary: 5 failed, 1 passed

# Pytest showed the actual values returned by the function and the
# expected values for each failed test. For example, when converting
# 100 degrees Celsius, the function returned 210.0 instead of 212.
# This is more useful than a generic "assertion failed" message
# because it identifies exactly which test cases failed and shows
# the difference between the actual and expected results, making
# it easier to locate and debug the problem.