import numpy as np
import matplotlib.pyplot as plt

def newton_interpolation(x, y, x_interp):
    def divided_differences(x, y):
        n = len(y)
        coef_matrix = np.zeros([n, n])
        coef_matrix[:, 0] = y
        for j in range(1, n):
            for i in range(n - j):
                coef_matrix[i][j] = (coef_matrix[i + 1][j - 1] - coef_matrix[i][j - 1]) / (x[i + j] - x[i])
        return coef_matrix[0, :]

    coefficients = divided_differences(x, y)
    n = len(coefficients)
    y_interp = coefficients[0]
    for i in range(1, n):
        term = coefficients[i]
        for j in range(i):
            term *= (x_interp - x[j])
        y_interp += term
    return y_interp

# Data input
x_data = np.array([5, 10, 15, 20, 25, 30, 35, 40])
y_data = np.array([40, 30, 25, 40, 18, 20, 22, 15])

# Plotting
x_vals = np.linspace(5, 40, 100)
y_vals_newton = [newton_interpolation(x_data, y_data, x) for x in x_vals]

plt.plot(x_data, y_data, 'o', label='Data points')
plt.plot(x_vals, y_vals_newton, '-', label='Newton interpolation')
plt.xlabel('Tegangan (kg/mm^2)')
plt.ylabel('Waktu patah (jam)')
plt.legend()
plt.title('Newton Interpolation')
plt.show()
