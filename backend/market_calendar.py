"""
Market calendar utility for computing US stock market trading days.

Provides functions to:
- Get list of trading days between two dates
- Map trading-day offsets to actual dates
- Cache trading days for performance
"""

import pandas as pd
from datetime import datetime, timedelta, date
from typing import List, Tuple


# Try to use exchange_calendars or pandas_market_calendars if available
try:
    from exchange_calendars import get_calendar
    USE_EXCHANGE_CALENDARS = True
except ImportError:
    USE_EXCHANGE_CALENDARS = False


def get_trading_days(start_date: date, end_date: date) -> List[date]:
    """
    Get list of US stock market trading days between start_date and end_date (inclusive).
    
    If exchange_calendars is available, use it. Otherwise, use a simple heuristic
    (exclude weekends; hardcoded major US holidays).
    
    Args:
        start_date: Start date (inclusive)
        end_date: End date (inclusive)
    
    Returns:
        Sorted list of trading dates
    """
    if USE_EXCHANGE_CALENDARS:
        try:
            calendar = get_calendar('XNYS')  # NYSE
            sessions = calendar.sessions
            mask = (sessions >= pd.Timestamp(start_date)) & (sessions <= pd.Timestamp(end_date))
            return [ts.date() for ts in sessions[mask]]
        except Exception as e:
            print(f"Warning: exchange_calendars failed: {e}. Falling back to simple heuristic.")
    
    # Simple fallback: exclude weekends and major US holidays
    trading_days = []
    current = start_date
    
    # Major US market holidays (2024-2027)
    holidays = {
        date(2024, 1, 1),   # New Year
        date(2024, 1, 15),  # MLK Jr.
        date(2024, 2, 19),  # Presidents Day
        date(2024, 3, 29),  # Good Friday
        date(2024, 5, 27),  # Memorial Day
        date(2024, 6, 19),  # Juneteenth
        date(2024, 7, 4),   # Independence Day
        date(2024, 9, 2),   # Labor Day
        date(2024, 11, 28), # Thanksgiving
        date(2024, 12, 25), # Christmas
        date(2025, 1, 1),   # New Year
        date(2025, 1, 20),  # MLK Jr.
        date(2025, 2, 17),  # Presidents Day
        date(2025, 4, 18),  # Good Friday
        date(2025, 5, 26),  # Memorial Day
        date(2025, 6, 19),  # Juneteenth
        date(2025, 7, 4),   # Independence Day
        date(2025, 9, 1),   # Labor Day
        date(2025, 11, 27), # Thanksgiving
        date(2025, 12, 25), # Christmas
        date(2026, 1, 1),   # New Year
        date(2026, 1, 19),  # MLK Jr.
        date(2026, 2, 16),  # Presidents Day
        date(2026, 4, 3),   # Good Friday
        date(2026, 5, 25),  # Memorial Day
        date(2026, 6, 19),  # Juneteenth
        date(2026, 7, 3),   # Independence Day (observed)
        date(2026, 9, 7),   # Labor Day
        date(2026, 11, 26), # Thanksgiving
        date(2026, 12, 25), # Christmas
    }
    
    while current <= end_date:
        # Skip weekends (5=Saturday, 6=Sunday)
        if current.weekday() < 5 and current not in holidays:
            trading_days.append(current)
        current += timedelta(days=1)
    
    return trading_days


def get_trading_day_offset_to_date(base_date: date, offset: int) -> date:
    """
    Map a trading-day offset (1-based) to an actual date.
    
    Args:
        base_date: Reference date (usually last trading day or run date)
        offset: Trading-day offset (1 = next trading day, 2 = day after, etc.)
    
    Returns:
        The target trading date
    
    Example:
        get_trading_day_offset_to_date(date(2026, 1, 1), 1) -> date(2026, 1, 2) (next trading day)
    """
    # Generate enough trading days (400 to handle 365 + buffer)
    end_date = base_date + timedelta(days=600)
    trading_days = get_trading_days(base_date, end_date)
    
    # Filter to days after base_date
    future_trading_days = [d for d in trading_days if d > base_date]
    
    if offset > len(future_trading_days):
        raise ValueError(f"Offset {offset} exceeds available trading days ({len(future_trading_days)})")
    
    return future_trading_days[offset - 1]


def get_next_n_trading_days(base_date: date, n: int) -> List[date]:
    """
    Get the next n trading days after base_date.
    
    Args:
        base_date: Reference date
        n: Number of trading days to return
    
    Returns:
        List of next n trading dates
    """
    end_date = base_date + timedelta(days=600)
    trading_days = get_trading_days(base_date, end_date)
    future_trading_days = [d for d in trading_days if d > base_date]
    return future_trading_days[:n]


def precompute_trading_day_list(start_date: date, days_ahead: int = 365) -> Tuple[List[date], dict]:
    """
    Precompute a list of trading days and a mapping of offset -> date for caching.
    
    Args:
        start_date: Date from which to start
        days_ahead: Number of trading days to compute (default 365)
    
    Returns:
        Tuple of (list of trading dates, dict mapping offset -> date)
    
    Example:
        dates, offset_map = precompute_trading_day_list(date(2026, 1, 2), 365)
        offset_map[1] -> date(2026, 1, 5)  (first trading day after start)
    """
    end_date = start_date + timedelta(days=600)
    trading_days = get_trading_days(start_date, end_date)
    future_trading_days = [d for d in trading_days if d >= start_date][:days_ahead]
    
    # Map offset (1-based) to date
    offset_map = {i + 1: d for i, d in enumerate(future_trading_days)}
    
    return future_trading_days, offset_map
