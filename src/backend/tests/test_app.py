from fastapi.testclient import TestClient
import sys
import os
import pytest

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_discover_movies():
    response = client.get("/api/movies/discover")
    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert isinstance(data["results"], list)

def test_get_movie_details():
    # Test with a known movie ID (The Matrix)
    response = client.get("/api/movies/603")
    assert response.status_code == 200
    data = response.json()
    assert data["title"] == "The Matrix"
    assert data["id"] == 603

def test_get_movies_by_genre():
    # Test with Action genre (ID: 28)
    response = client.get("/api/movies/genre/28")
    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert isinstance(data["results"], list)

def test_get_valid_movies():
    response = client.get("/valid-movies")
    assert response.status_code == 200
    data = response.json()
    assert "total_movies" in data
    assert "popular_movies" in data
    assert isinstance(data["popular_movies"], dict) 