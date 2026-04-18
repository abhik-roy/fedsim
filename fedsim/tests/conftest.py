import pytest
from plugins import clear_cache


@pytest.fixture(autouse=True, scope="session")
def reset_plugin_cache():
    clear_cache()
    yield
    clear_cache()
