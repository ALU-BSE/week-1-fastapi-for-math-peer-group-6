from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import numpy as np

app = FastAPI()

# Sigmoid Function: σ(x) = 1 / (1 + e⁻ˣ)
# Where:
# x is the input value.
# e is Euler's number (approximately 2.71828).

# Sigmoid function
def sigmoid(x):
    """
    This sigmoid function returns an 
    output between 0 and 1.

    The formula is 1 / (1 + e^(-x))
    but e is equivalent to 2.71828.
    """
    return 1 / (1 + np.exp(-x))  # Optimized using numpy's np.exp

class MatrixInput(BaseModel):
    """
    This model validates the input going into the function,
    making sure they're a matrix of list[list] with floats.
    """
    matrix: list[list[float]]

# Initialize M and B
M = np.ones((5, 5))  # 5x5 matrix of ones
B = np.zeros((5, 5))  # 5x5 matrix of zeros

@app.post("/calculate")
def calculate(input_data: MatrixInput):
    """
    Perform the following operations:
    - matrix_multiplication: (M * X) + B using NumPy.
    - non_numpy_multiplication: (M * X) + B without NumPy.
    - sigmoid_output: Apply sigmoid to matrix_multiplication.
    """
    # Validate the matrix size (should be 5x5)
    if len(input_data.matrix) != 5 or any(len(row) != 5 for row in input_data.matrix):
        raise HTTPException(status_code=400, detail="Matrix must be 5x5")

    X = np.array(input_data.matrix)  # Convert input to NumPy array

    # Using NumPy for matrix multiplication
    matrix_multiplication = np.dot(M, X) + B

    # Without NumPy (manual calculation)
    non_numpy_multiplication = [[0 for _ in range(5)] for _ in range(5)]
    for i in range(5):
        for j in range(5):
            for k in range(5):
                non_numpy_multiplication[i][j] += M[i][k] * X[k][j]
            # Add bias
            non_numpy_multiplication[i][j] += B[i][j]

    # Apply sigmoid to the NumPy result
    sigmoid_output = sigmoid(matrix_multiplication)

    # Return the results
    return {
        "matrix_multiplication": matrix_multiplication.tolist(),
        "non_numpy_multiplication": non_numpy_multiplication,
        "sigmoid_output": sigmoid_output.tolist(),
    }

if __name__ == "__main__":
    uvicorn.run(app)

