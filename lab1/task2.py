import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

# Дані: вхідні вектори та очікуваний результат
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = [0, 1, 1, 0]

plt.figure(figsize=(8, 6))
x = np.linspace(-0.5, 1.5, 100)

# 1. Розділяючі прямі
# Рівняння OR: x1 + x2 = 0.5 => x2 = 0.5 - x1
plt.plot(x, 0.5 - x, 'g-', linewidth=2, label='OR boundary')

# Рівняння AND: x1 + x2 = 1.5 => x2 = 1.5 - x1
plt.plot(x, 1.5 - x, 'r-', linewidth=2, label='AND boundary')

# 2. Зафарбовування областей класів
# Клас 1 (XOR=1): між прямими
plt.fill_between(x, 0.5 - x, 1.5 - x, color='lightgreen', alpha=0.5, label='Class 1(XOR = 1)')
# Клас 0 (XOR=0): зовні
plt.fill_between(x, -1, 0.5 - x, color='salmon', alpha=0.2, label='Class 0(XOR = 0)')
plt.fill_between(x, 1.5 - x, 2.5, color='salmon', alpha=0.2)

# 3. Відображення точок
for i, point in enumerate(X):
    color = 'green' if y[i] == 1 else 'red'
    marker = 's' if y[i] == 1 else 'o' # квадрат для 1, коло для 0

    plt.scatter(point[0], point[1], c=color, marker=marker, s=150, edgecolors='k', zorder=5)
    plt.text(point[0] + 0.05, point[1], f"({point[0]},{point[1]})", fontsize=10)

# Оформлення
plt.xlim(-0.5, 1.5)
plt.ylim(-0.5, 1.5)
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('XOR Decision Boundaries')
plt.grid(True)
plt.legend(loc='upper right')
plt.show()