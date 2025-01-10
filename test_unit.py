import pytest
from fastapi.testclient import TestClient
from test import app  # Import your FastAPI app

client = TestClient(app)

def test_valid_matrix():
    response = client.post(
        "/calculate",
        json={
            "matrix": [
                [1, 2, 3, 4, 5],
                [6, 7, 8, 9, 10],
                [11, 12, 13, 14, 15],
                [16, 17, 18, 19, 20],
                [21, 22, 23, 24, 25]
            ]
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "matrix_multiplication" in data
    assert "non_numpy_multiplication" in data
    assert "sigmoid_output" in data

def test_invalid_matrix():
    response = client.post(
        "/calculate",
        json={
            "matrix": [
                [1, 2, 3],  # Invalid: Not 5 columns
                [6, 7, 8, 9, 10],
                [11, 12, 13, 14, 15],
                [16, 17, 18, 19, 20],
                [21, 22, 23, 24, 25]
            ]
        },
    )
    assert response.status_code == 400
    assert response.json() == {"detail": "Matrix must be 5x5"}

def test_empty_matrix():
    response = client.post(
        "/calculate",
        json={"matrix": []},  # Empty matrix
    )
    assert response.status_code == 400
    assert response.json() == {"detail": "Matrix must be 5x5"}
