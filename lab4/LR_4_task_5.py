import matplotlib
from sklearn.preprocessing import PolynomialFeatures

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model

# 6 варіант
m = 100
X = np.linspace(-3, 3, m)
y = 2 * np.sin(X) + np.random.uniform(-0.6, 0.6, m)

# Приводимо X до форми двовим. масиву
X = X.reshape(-1, 1)

# Лінійна регресія
lin_model = linear_model.LinearRegression()
lin_model.fit(X, y)



# Поліноміальна
poly = PolynomialFeatures(degree=2, include_bias=False)
x_poly = poly.fit_transform(X)

poly_model = linear_model.LinearRegression()
poly_model.fit(x_poly, y)

# Вивід результатів
print('Linear ---')
print('Coef:', lin_model.coef_)
print('Intercept:', lin_model.intercept_)

print('\nPolynomial ---')
print('Coef:', poly_model.coef_)
print('Intercept:', poly_model.intercept_)

# Предикт для графіку
y_pred_lin = lin_model.predict(X)
y_pred_poly = poly_model.predict(x_poly)

# Малюємо графік
plt.figure(figsize=(9, 5))
plt.scatter(X, y, color='black', s=10, label="Початкові дані")
plt.plot(X, y_pred_lin, color='orange', label="Linear")
plt.plot(X, y_pred_poly, color='blue', label="Poly (deg=2)")

plt.xlabel("X")
plt.ylabel("Y")
plt.title("Порівняння регресій")
plt.legend()
plt.grid(alpha=0.5)
plt.show()