"""
Test that logging initialises correctly and writes to file.
"""
import sys
import os
from pathlib import Path

# Add project root to path so imports work
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Override log dir to local for testing (not /app/logs)
import yaml
config_path = Path("config/config.yaml")
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# Temporarily point logs to local test folder
config["logging"]["log_dir"] = "logs"
with open(config_path, "w") as f:
    yaml.dump(config, f)

from utils.logging import get_logger

def test_logging():
    logger = get_logger("test_module", "test.log")

    logger.info("Test INFO message")
    logger.warning("Test WARNING message")
    logger.error("Test ERROR message")
    logger.debug("Test DEBUG message")

    # Verify log file was created
    log_file = Path("logs/test.log")
    assert log_file.exists(), "Log file was not created"

    # Verify content was written
    content = log_file.read_text()
    assert "Test INFO message" in content
    assert "Test WARNING message" in content
    assert "Test ERROR message" in content

    print("✅ Logger initialised successfully")
    print("✅ Log file created at logs/test.log")
    print("✅ All log levels written correctly")
    print(f"\nLog file content preview:\n{'-'*50}")
    print(content[:500])

if __name__ == "__main__":
    test_logging()