[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/VesdRxnC)
# math_for_ml-fastapi_for_math
# FastAPI Math Application

**Overview**

This FastAPI application demonstrates how to perform matrix multiplication (MX + B) and apply a sigmoid function using both NumPy and a non-NumPy implementation. It accepts a JSON request containing a 5x5 matrix for calculation and returns the results.

**Formulas**

*   **Matrix Multiplication:** `Result = (M * X) + B`
    *   Where:
        *   `M` is a pre-defined matrix.
        *   `X` is the input matrix.
        *   `B` is a pre-defined matrix (bias).
*   **Sigmoid Function:** `σ(x) = 1 / (1 + e⁻ˣ)`
    *   Where:
        *   `x` is the input value.
        *   `e` is Euler's number (approximately 2.71828).

**API Endpoints**

*   **POST /calculate** (JSON request):
    *   Takes a JSON object with a `matrix` key containing a 5x5 list of lists representing your matrix.
    *   Returns a JSON object with the following keys:
        *   `matrix_multiplication`: The result of `(M * X) + B` using NumPy.
        *   `non_numpy_multiplication`: The result of `(M * X) + B` without NumPy.
        *   `sigmoid_output`: The result of applying the sigmoid function to the `matrix_multiplication` result using NumPy.

**Example Request (curl)**

```bash
curl -X POST http://localhost:8000/calculate -H "Content-Type: application/json" -d '{"matrix": [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15], [16, 17, 18, 19, 20], [21, 22, 23, 24, 25]]}'
