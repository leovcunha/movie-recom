import pytest
import os
from dotenv import load_dotenv

@pytest.fixture(autouse=True)
def setup_test_env():
    """Load environment variables for tests"""
    load_dotenv()
    # Verify TMDB API key is available
    assert os.getenv("TMDB_API_KEY"), "TMDB_API_KEY environment variable is required" 