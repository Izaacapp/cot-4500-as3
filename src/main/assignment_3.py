import numpy as np

def euler_method(f, t0, y0, t_end, n):
    h = (t_end - t0) / n  # Step size
    t_values = np.linspace(t0, t_end, n + 1)
    y_values = np.zeros(n + 1)
    y_values[0] = y0
    
    for i in range(n):
        y_values[i + 1] = y_values[i] + h * f(t_values[i], y_values[i])
    
    return y_values[-1]  # Returning the last value

def runge_kutta_4(f, t0, y0, t_end, n):
    h = (t_end - t0) / n  # Step size
    t_values = np.linspace(t0, t_end, n + 1)
    y_values = np.zeros(n + 1)
    y_values[0] = y0

    for i in range(n):
        t = t_values[i]
        y = y_values[i]

        k1 = h * f(t, y)
        k2 = h * f(t + h / 2, y + k1 / 2)
        k3 = h * f(t + h / 2, y + k2 / 2)
        k4 = h * f(t + h, y + k3)

        y_values[i + 1] = y + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return y_values[-1]  # Returning the last value

# Define the function f(t, y) = t - y^2
def f(t, y):
    return t - y**2

# Problem Parameters
t0 = 0
y0 = 1
t_end = 2
n = 10

# Compute results
euler_result = euler_method(f, t0, y0, t_end, n)
runge_kutta_result = runge_kutta_4(f, t0, y0, t_end, n)

print(f" Euler's Method Result: {euler_result}")
print(f" Runge-Kutta Method Result: {runge_kutta_result}")
