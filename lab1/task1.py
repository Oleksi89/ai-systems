import numpy as np

def step_function(x):
    """Функція активації: 1 якщо x > 0, інакше 0"""
    return 1 if x > 0 else 0

def or_neuron(x1, x2):
    """
    Реалізація функції OR.
    Розділяюча лінія: x1 + x2 = 0.5
    Ваги: w1=1, w2=1, поріг=-0.5
    """
    weights = np.array([1, 1])
    inputs = np.array([x1, x2])
    bias = -0.5

    # Суматор: w1*x1 + w2*x2 + bias
    total = np.dot(weights, inputs) + bias
    return step_function(total)

def and_neuron(x1, x2):
    """
    Реалізація функції AND.
    Розділяюча лінія: x1 + x2 = 1.5
    Ваги: w1=1, w2=1, поріг=-1.5
    """
    weights = np.array([1, 1])
    inputs = np.array([x1, x2])
    bias = -1.5

    total = np.dot(weights, inputs) + bias
    return step_function(total)

def xor_neuron(x1, x2):
    """
    Реалізація XOR через OR та AND.
    Вхідні дані: y1 (від OR) та y2 (від AND).
    Розділяюча лінія (y1, y2): y1 - y2 = 0.5
    Ваги: w_or=1, w_and=-1, поріг=-0.5
    """
    # Отримуємо сигнали від попереднього шару
    y1 = or_neuron(x1, x2)
    y2 = and_neuron(x1, x2)

    # Обчислюємо вихід XOR
    inputs = np.array([y1, y2])
    weights = np.array([-1, 1]) # Вага -1 для OR, 1 для AND
    bias = 0.5

    total = np.dot(weights, inputs) + bias
    return step_function(total)

# --- Тестування ---
print(f"{'x1':<5} {'x2':<5} | {'OR':<5} {'AND':<5} | {'XOR (Result)':<10}")
print("-" * 45)

test_cases = [(0, 0), (0, 1), (1, 0), (1, 1)]

for x1, x2 in test_cases:
    res_or = or_neuron(x1, x2)
    res_and = and_neuron(x1, x2)
    res_xor = xor_neuron(x1, x2)

    print(f"{x1:<5} {x2:<5} | {res_or:<5} {res_and:<5} | {res_xor:<10}")