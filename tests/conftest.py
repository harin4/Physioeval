import pytest


# Configure pytest-asyncio to use auto mode for async tests
def pytest_configure(config):
    config.addinivalue_line(
        "markers", "asyncio: mark test as async"
    )
