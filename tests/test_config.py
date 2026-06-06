"""
Test that config.yaml loads correctly and all expected keys exist.
"""
import yaml
from pathlib import Path

def test_config_loads():
    config_path = Path("config/config.yaml")
    assert config_path.exists(), "config.yaml not found"

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Check all top level keys exist
    expected_keys = [
        "universe", "filters", "linreg", "smc",
        "volume", "scanner", "sentiment", "ml",
        "scheduler", "database", "logging", "fetcher"
    ]
    for key in expected_keys:
        assert key in config, f"Missing key: {key}"
        print(f"✅ {key}")

    # Check critical values
    assert config["filters"]["min_price"] == 10.0
    assert config["filters"]["min_avg_volume"] == 500000
    assert config["linreg"]["period"] == 200
    assert config["smc"]["min_pivot_candles"] == 5
    assert len(config["universe"]["indices"]) == 3
    assert len(config["universe"]["sectors"]) == 11

    print("\n✅ All config keys and values verified")

if __name__ == "__main__":
    test_config_loads()