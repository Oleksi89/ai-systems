import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

diabetes = datasets.load_diabetes()
X = diabetes.data
y = diabetes.target

# Розділення на навчальні та тестові дані
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.5, random_state=0)

# Модель лінійної регресії і тренування
regr = linear_model.LinearRegression()
regr.fit(Xtrain, ytrain)

# прогноз по тестовій вибірці
ypred = regr.predict(Xtest)

# вивід коефіцієнтів регресії та показників
print("Коефіцієнти лінійної регресії: ", regr.coef_)
print("Зміщення (intercept) моделі: ", round(regr.intercept_, 2))
print("Показник R2: ", round(r2_score(ytest, ypred), 2))
print("Середня абсолютна помилка: ", round(mean_absolute_error(ytest, ypred), 2))
print("Середньоквадратична помилка: ", round(mean_squared_error(ytest, ypred), 2))

# графіки
fig, ax = plt.subplots()
ax.scatter(ytest, ypred, edgecolors=(0, 0, 0))
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
ax.set_xlabel('Виміряно')
ax.set_ylabel('Передбачено')
plt.show()
