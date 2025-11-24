import numpy as np
from sklearn import preprocessing
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Вхідний файл, який містить дані
input_file = 'income_data.txt'

# Читання даних
X = []
y = []
count_class1 = 0
count_class2 = 0
max_datapoints = 25000

with open(input_file, 'r') as f:
    for line in f.readlines():
        if count_class1 >= max_datapoints and count_class2 >= max_datapoints:
            break
        if '?' in line:
            continue

        data = line[:-1].split(', ')

        if data[-1] == '<=50K' and count_class1 < max_datapoints:
            X.append(data)
            count_class1 += 1
        if data[-1] == '>50K' and count_class2 < max_datapoints:
            X.append(data)
            count_class2 += 1

# Перетворення на масив numpy
X = np.array(X)

# Перетворення рядкових даних на числові
label_encoder = []
X_encoded = np.empty(X.shape)

for i, item in enumerate(X[0]):
    if item.isdigit():
        X_encoded[:, i] = X[:, i]
    else:
        label_encoder.append(preprocessing.LabelEncoder())
        X_encoded[:, i] = label_encoder[-1].fit_transform(X[:, i])

X = X_encoded[:, :-1].astype(int)
y = X_encoded[:, -1].astype(int)

# Розбиття даних на навчальний та тестовий набори
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

print("Training sigmoid kernel...")
# Створення SVМ-класифікатора з поліноміальним ядром, навчання та прогноз
classifier = OneVsOneClassifier(SVC(kernel='sigmoid', max_iter=5000))
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

# Обчислення F-міри для SVМ
f1 = cross_val_score(classifier, X, y, scoring='f1_weighted', cv=3)
print("Sigmoid kernel...")
print("F1 score: " + str(round(100 * f1.mean(), 2)) + "%")

# Обчислення показників якості класифікації на тестовому наборі
print("\n--- Metrics ---")
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average='weighted')
rec = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("Accuracy: " + str(round(100 * acc, 2)) + "%")
print("Precision: " + str(round(100 * prec, 2)) + "%")
print("Recall: " + str(round(100 * rec, 2)) + "%")

# Передбачення результату для тестової точки даних
input_data = ['37', 'Private', '215646', 'HS-grad', '9', 'Never-married', 'Handlers-cleaners', 'Not-in-family', 'White', 'Male', '0', '0', '40', 'United-States']

# Кодування тестової точки даних
input_data_encoded = [-1] * len(input_data)
count = 0

for i, item in enumerate(input_data):
    if item.isdigit():
        input_data_encoded[i] = int(input_data[i])
    else:
        # Використовуємо відповідний енкодер, reshape потрібен для transform
        input_data_encoded[i] = int(label_encoder[count].transform([input_data[i]])[0])
        count += 1

input_data_encoded = np.array(input_data_encoded)

# Використання класифікатора для кодованої точки даних та виведення результату
# Reshape data для передбачення одного зразка
predicted_class = classifier.predict(input_data_encoded.reshape(1, -1))

# Декодування результату
result_label = label_encoder[-1].inverse_transform(predicted_class)[0]
print("\nPredicted class: " + result_label)