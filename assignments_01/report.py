
"""Generate a daily weather report from a validated API response."""

import json
from pathlib import Path

from weatherkit.schemas import WeatherResponse
from weatherkit.records import to_readings
from weatherkit.summarize import DailyAggregator


def main() -> None:
    """Load, validate, aggregate, and report daily weather observations."""

    # Locate the JSON file relative to this script, not the terminal directory.
    json_path = Path(__file__).parent / "weather_raw.json"

    # Step 1: Load the raw API response.
    with json_path.open("r", encoding="utf-8") as file:
        raw_data = json.load(file)

    # Step 2: Validate the external data at the package boundary.
    response = WeatherResponse.model_validate(raw_data)

    # Step 3: Convert columnar measurements into hourly records.
    readings = to_readings(response)

    # Step 4: Aggregate hourly records into daily summaries.
    aggregator = DailyAggregator()
    summaries = aggregator.summarize(readings)
    incomplete = aggregator.incomplete_days(readings)

    # Step 5: Print one readable row per complete day.
    print("Daily Weather Report")
    print("=" * 79)
    print(
        f"{'Date':<12}"
        f"{'High (C)':>12}"
        f"{'Low (C)':>12}"
        f"{'Precip (mm)':>15}"
        f"{'Range (C)':>14}"
    )
    print("-" * 79)

    for summary in summaries:
        print(
            f"{summary.date:<12}"
            f"{summary.temp_max:>12.1f}"
            f"{summary.temp_min:>12.1f}"
            f"{summary.precipitation_sum:>15.1f}"
            f"{summary.temp_range():>14.1f}"
        )

    # Step 6: Warn about any dates excluded by the completeness threshold.
    if incomplete:
        print(f"\nWARNING: Incomplete days dropped: {', '.join(incomplete)}")
    else:
        print("\nNo incomplete days were dropped.")


# Without this guard, importing report.py would automatically run main(),
# reading the JSON file and printing a report even if another module only
# wanted to reuse a helper function.
if __name__ == "__main__":
    main()

# Task 7: Reflection
#
# 1. Handling null temperatures
#
# Rejecting the entire response is appropriate when data completeness is
# critical, such as when a pipeline requires every hourly observation to
# produce reliable results. This prevents incomplete data from entering
# downstream calculations unnoticed.
#
# However, in a weather-monitoring pipeline, I might prefer to tolerate
# an occasional missing temperature rather than discard an entire week
# of otherwise useful observations.
#
# To allow missing temperatures, I would change the schema field to
# temperature_2m: list[float | None].
# I would also update the conversion and aggregation logic to handle
# missing values explicitly, rather than passing None into calculations.
#
# 2. The min_hours threshold
#
# If the pipeline runs at noon, the current day may have only 12 hourly
# observations. With min_hours=24, that day would be excluded even though
# the data is arriving normally. Reporting its maximum temperature as
# a complete daily maximum would also be misleading.
#
# The incomplete_days() method identifies excluded dates so the pipeline
# can report them instead of silently dropping them. This helps distinguish
# incomplete observations from complete daily summaries.
#
# 3. Benefits of the weatherkit package
#
# Keeping validation, record conversion, and aggregation in separate
# modules makes the code easier to reuse in Week 10.
#
# For example, a cloud pipeline can import WeatherResponse, to_readings,
# and DailyAggregator directly without importing or executing the report
# script. This also allows each component to be tested and maintained
# independently.
