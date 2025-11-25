from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn import linear_model, preprocessing
from sklearn.pipeline import Pipeline


# будує криві навчання моделі для навчальних даних
def plot_learning_curves(model, X, y):
    # Розділення на навчальні та тестові дані
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    train_errors, val_errors = [], []
    for l in range(1, len(X_train)):
        model.fit(X_train[:l], y_train[:l])
        # Прогнози
        y_train_predict = model.predict(X_train[:l])
        y_val_predict = model.predict(X_val)
        # Перехоплення помилок
        train_errors.append(mean_squared_error(y_train_predict, y_train[:l]))
        val_errors.append(mean_squared_error(y_val_predict, y_val))

    # Відображення кривих навчання
    plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="Навчальний набір")
    plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="Тестовий набір")
    plt.xlabel("Training set size")
    plt.ylabel("RMSE")
    plt.legend()
    plt.show()


# 6 варіант
m = 100
X = np.linspace(-3, 3, m)
y = 2 * np.sin(X) + np.random.uniform(-0.6, 0.6, m)

# Приводимо X до форми двовим. масиву
X = X.reshape(-1, 1)

# Лінійна регресія
linear_regressor = linear_model.LinearRegression()

# Поліноміальні ознаки
polynomial_features = preprocessing.PolynomialFeatures(
    degree=2,
    include_bias=False,
)
X_poly_train = polynomial_features.fit_transform(X)

polynomial_regressor = Pipeline([
    ("poly_features", polynomial_features),
    ("lin_reg", linear_model.LinearRegression()),
])

# Навчання моделей на даних
linear_regressor.fit(X, y)
polynomial_regressor.fit(X_poly_train, y)

# Передбачення для обох моделей
y_linear = linear_regressor.predict(X)
y_polynomial = polynomial_regressor.predict(X_poly_train)

# Відображення кривих навчання для лінійної моделі та поліноміальної моделі
plot_learning_curves(linear_regressor, X, y)
plot_learning_curves(polynomial_regressor, X_poly_train, y)
