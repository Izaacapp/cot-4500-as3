import pytest
from main.assignment_3 import ODESolver


# Define the function f(t, y) = t - y^2
def f(t, y):
    return t - y**2


# Problem parameters for test case 1
t0 = 0
y0 = 1
t_end = 2
n = 10

# Expected results from the assignment description
EXPECTED_EULER_RESULT = 1.2446380979332121
EXPECTED_RK4_RESULT = 1.251316587879806


def test_euler_method():
    solver = ODESolver(f, t0, y0, t_end, n)
    euler_result = solver.euler_method()
    assert (
        abs(euler_result - EXPECTED_EULER_RESULT) < 1e-6
    )  # Allow small numerical errors


def test_runge_kutta_4():
    solver = ODESolver(f, t0, y0, t_end, n)
    rk4_result = solver.runge_kutta_4()
    assert abs(rk4_result - EXPECTED_RK4_RESULT) < 1e-6  # Allow small numerical errors


if __name__ == "__main__":
    pytest.main()
