#!/usr/bin/env python3
"""Test script for auto-forecast agent.

Tests agent initialization, data loading, and basic forecast generation.
Run after installing dependencies: pip install -r requirements.txt
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.agents.auto_forecast import AutoForecastAgent


def test_agent_initialization():
    """Test that agent can be initialized."""
    print("Testing AutoForecastAgent initialization...")
    agent = AutoForecastAgent()
    assert agent is not None
    print("  ✓ Agent initialized")


def test_data_loading():
    """Test that data loads successfully."""
    print("Testing data loading...")
    agent = AutoForecastAgent()
    agent.load_data()
    assert agent.measures is not None
    assert len(agent.measures) > 0
    print(f"  ✓ Loaded {len(agent.measures)} measures")


def test_forecast_generation():
    """Test forecast generation for a single deal."""
    print("Testing forecast generation...")
    agent = AutoForecastAgent()
    agent.load_data()

    # Get first available deal
    deals = agent.measures["Deal Name"].unique().to_list()
    print(f"  Found {len(deals)} deals")

    if deals:
        deal = deals[0]
        print(f"  Forecasting for: {deal}")
        result = agent.forecast_deal(deal)
        if result:
            print(f"  ✓ Forecast: ${result.forecast_value:,.2f} "
                  f"(regime: {result.regime})")
        else:
            print("  ⚠ Forecast returned None (insufficient data?)")


def test_full_run():
    """Test full agent run (with alerts disabled)."""
    print("Testing full agent run...")
    agent = AutoForecastAgent()
    result = agent.run(run_all=True, send_alerts=False)
    print(f"  ✓ Run completed: {result}")


def main():
    """Run all tests."""
    print("=" * 60)
    print("AUTO-FORECAST AGENT TEST SUITE")
    print("=" * 60)

    tests = [
        test_agent_initialization,
        test_data_loading,
        test_forecast_generation,
        # test_full_run,  # Commented out to avoid long test (forecasts all deals)
    ]

    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            import traceback
            traceback.print_exc()

    print("=" * 60)
    print("Tests completed.")
    print("=" * 60)


if __name__ == "__main__":
    main()