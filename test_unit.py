import pytest
from fastapi.testclient import TestClient
from test import app  # Import the FastAPI app from the file `test.py`

client = TestClient(app)

# Test a valid 5x5 matrix, expecting a successful response.
def test_valid_5x5_matrix():
    response = client.post(
        "/calculate",
        json={
            "matrix": [
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1]
            ]
        },
    )
    assert response.status_code == 200
    data = response.json()

    # Check that the response contains the expected keys.
    assert "matrix_multiplication" in data
    assert "non_numpy_multiplication" in data
    assert "sigmoid_output" in data

# Test an invalid matrix with incorrect dimensions (not 5x5).
def test_invalid_matrix_size():
    response = client.post(
        "/calculate",
        json={
            "matrix": [
                [1, 1, 1, 1],  # This row has only 4 elements instead of 5
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [1, 1, 1, 1]
            ]
        },
    )
    assert response.status_code == 400
    assert response.json() == {"detail": "Matrix must be 5x5"}

# Test an invalid matrix with a non-numeric value.
def test_non_numeric_in_matrix():
    response = client.post(
        "/calculate",
        json={
            "matrix": [
                [1, 1, 1, 1, 1],
                [1, "a", 1, 1, 1],  # Invalid, 'a' is not a number
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1]
            ]
        },
    )
    assert response.status_code == 422  # Should be a validation error
    assert "detail" in response.json()  # Check that the error details are included

# Test the matrix multiplication logic with a custom 5x5 matrix.
def test_matrix_multiplication_logic():
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

    # Ensure the response matrix has the correct dimensions.
    assert len(data["matrix_multiplication"]) == 5
    assert len(data["matrix_multiplication"][0]) == 5
    assert len(data["non_numpy_multiplication"]) == 5
    assert len(data["non_numpy_multiplication"][0]) == 5
    assert len(data["sigmoid_output"]) == 5
    assert len(data["sigmoid_output"][0]) == 5

# Optional test: Check the response when sending an empty matrix.
def test_empty_matrix():
    response = client.post(
        "/calculate",
        json={"matrix": []},  # Sending an empty matrix
    )
    assert response.status_code == 400  # This should fail validation
    assert "detail" in response.json()

