from dataclasses import dataclass
from .records import HourlyReading

# Part 2 - Task 4: The Aggregation

@dataclass
class DailySummary:
    """Represent aggregated weather measurements for one calendar day."""
    date: str
    temp_max: float
    temp_min: float
    precipitation_sum: float
    hours_observed: int

   
    def temp_range(self) -> float:
        """Return the difference between maximum and minimum temperature."""
        return self.temp_max - self.temp_min

class DailyAggregator:
   """Aggregate hourly weather observations into daily summaries."""
   def __init__(self, min_hours: int = 24) -> None:
      """Initialize the minimum number of observations required per day."""
      self.min_hours = min_hours
      
   def summarize(self, readings: list[HourlyReading]) -> list[DailySummary]:
       """Return date-sorted daily summaries for sufficiently complete days.

        Args:
            readings: Hourly weather observations to aggregate.

        Returns:
            Daily summaries for dates meeting the minimum observation count.
        """
       grouped: dict[str, list[HourlyReading]] = {}
       for reading in readings:
            date = reading.timestamp[:10]
            if date not in grouped:
                grouped[date] = []
            grouped[date].append(reading)
       summaries: list[DailySummary] = []
       for date, daily_readings in grouped.items():
           if len(daily_readings) < self.min_hours:
              continue
           max_temp = max(reading.temperature_c for reading in daily_readings)
           min_temp = min(reading.temperature_c for reading in daily_readings)
           total_precipitation = sum(reading.precipitation_mm for reading in daily_readings)
           hours_observed = len(daily_readings)

           summary = DailySummary(
              date=date,
              temp_max=max_temp,
              temp_min=min_temp,
              precipitation_sum=total_precipitation,
              hours_observed=hours_observed
           )
           summaries.append(summary)
       return sorted(summaries, key=lambda summary: summary.date)

   def incomplete_days(self, readings: list[HourlyReading]) -> list[str]:
        """Return sorted dates with fewer than the required hourly observations.

        Args:
            readings: Hourly weather observations to examine.

        Returns:
            A sorted list of dates that do not meet min_hours.
        """
        hours_by_date: dict[str, int] = {}

        for reading in readings:
            date = reading.timestamp[:10]
            hours_by_date[date] = hours_by_date.get(date, 0) + 1

        incomplete = [
            date
            for date, count in hours_by_date.items()
            if count < self.min_hours
        ]

        return sorted(incomplete)
          


           
           



    