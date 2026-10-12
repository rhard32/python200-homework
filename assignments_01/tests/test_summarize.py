
import pytest

from weatherkit.records import HourlyReading
from weatherkit.summarize import DailyAggregator


@pytest.fixture
def sample_readings() -> list[HourlyReading]:
    """Provide hourly readings spanning two calendar dates."""
    return [
        HourlyReading("2026-04-08T00:00", 10.0, 0.1),
        HourlyReading("2026-04-08T01:00", 16.0, 0.2),
        HourlyReading("2026-04-08T02:00", 7.0, 0.3),
        HourlyReading("2026-04-09T00:00", 12.0, 0.4),
        HourlyReading("2026-04-09T01:00", 18.0, 0.5),
    ]


def test_grouping_two_dates(sample_readings):
    """Readings spanning two dates produce two daily summaries."""
    aggregator = DailyAggregator(min_hours=2)

    summaries = aggregator.summarize(sample_readings)

    assert len(summaries) == 2
    assert [summary.date for summary in summaries] == [
        "2026-04-08",
        "2026-04-09",
    ]


def test_temperature_max_and_min(sample_readings):
    """Daily temperature extremes match the known input."""
    aggregator = DailyAggregator(min_hours=2)

    summaries = aggregator.summarize(sample_readings)

    assert summaries[0].temp_max == 16.0
    assert summaries[0].temp_min == 7.0
    assert summaries[0].temp_range() == 9.0

# Deliberate failure check: Changing max() to min() in
# DailyAggregator.summarize() caused this test to fail.
# The test detected an incorrect maximum temperature (7.0 instead of 16.0).
# The original max() calculation was restored afterward.


def test_precipitation_sum(sample_readings):
    """Daily precipitation is the sum of hourly measurements."""
    aggregator = DailyAggregator(min_hours=2)

    summaries = aggregator.summarize(sample_readings)

    assert summaries[0].precipitation_sum == pytest.approx(0.6)
    assert summaries[1].precipitation_sum == pytest.approx(0.9)


def test_incomplete_day_is_dropped(sample_readings):
    """A day below min_hours is excluded and reported as incomplete."""
    aggregator = DailyAggregator(min_hours=3)

    summaries = aggregator.summarize(sample_readings)
    incomplete = aggregator.incomplete_days(sample_readings)

    assert len(summaries) == 1
    assert summaries[0].date == "2026-04-08"
    assert incomplete == ["2026-04-09"]


def test_lowering_min_hours_keeps_day(sample_readings):
    """Lowering min_hours includes a previously incomplete day."""
    aggregator = DailyAggregator(min_hours=2)

    summaries = aggregator.summarize(sample_readings)

    assert len(summaries) == 2
    assert summaries[1].date == "2026-04-09"
    assert summaries[1].hours_observed == 2


@pytest.mark.parametrize(
    "min_hours, expected_count",
    [
        (1, 2),
        (2, 2),
        (3, 1),
        (4, 0),
    ],
)
def test_min_hours_parameter(sample_readings, min_hours, expected_count):
    """The number of reported days changes with the threshold."""
    aggregator = DailyAggregator(min_hours=min_hours)

    summaries = aggregator.summarize(sample_readings)

    assert len(summaries) == expected_count
