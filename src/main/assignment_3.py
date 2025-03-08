import numpy as np


class ODESolver:
    def __init__(self, f, t0, y0, t_end, n):
        self.f = f
        self.t0 = t0
        self.y0 = y0
        self.t_end = t_end
        self.n = n
        self.h = (t_end - t0) / n  # Step size
        self.t_values = np.linspace(t0, t_end, n + 1)

    def euler_method(self):
        y_values = np.zeros(self.n + 1)
        y_values[0] = self.y0
        for i in range(self.n):
            y_values[i + 1] = y_values[i] + self.h * self.f(
                self.t_values[i], y_values[i]
            )
        return y_values[-1]

    def runge_kutta_4(self):
        y_values = np.zeros(self.n + 1)
        y_values[0] = self.y0
        for i in range(self.n):
            t, y = self.t_values[i], y_values[i]
            k1 = self.h * self.f(t, y)
            k2 = self.h * self.f(t + self.h / 2, y + k1 / 2)
            k3 = self.h * self.f(t + self.h / 2, y + k2 / 2)
            k4 = self.h * self.f(t + self.h, y + k3)
            y_values[i + 1] = y + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        return y_values[-1]


# Define the function f(t, y) = t - y^2
def f(t, y):
    return t - y**2


# Problem Parameters
solver = ODESolver(f, t0=0, y0=1, t_end=2, n=10)

# Compute results
euler_result = solver.euler_method()
runge_kutta_result = solver.runge_kutta_4()

print(f"Euler's Method Result: {euler_result}")
print(f"Runge-Kutta Method Result: {runge_kutta_result}")
