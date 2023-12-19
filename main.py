# Импортируем необходимые библиотеки
import pandas as pd
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Ваш CSV-формат данных
data = pd.read_csv("./Titanic-Dataset.csv")
data = data.dropna()

# Преобразование данных в строку
data_string = data.to_csv(index=False)

# Преобразование данных в DataFrame
columns = ["PassengerId", "Survived", "Pclass", "Name", "Sex", "Age", "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked"]
df = pd.read_csv(StringIO(data_string), sep=',', header=None, names=columns, index_col="PassengerId", skiprows=1)

# Кодирование категориальных признаков
df = pd.get_dummies(df, columns=["Sex", "Embarked"], drop_first=True)

# Удаление ненужных столбцов (в этом примере)
df = df.drop(["Name", "Ticket", "Cabin"], axis=1)

# Разделение данных на признаки и целевую переменную
X = df.drop("Survived", axis=1)
y = df["Survived"]

# Разделение данных на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создание и обучение модели логистической регрессии
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Предсказание на тестовом наборе
y_pred = model.predict(X_test)

# Оценка производительности модели
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Вывод дополнительной информации о классификации
print(classification_report(y_test, y_pred))

# Вычисление процента выживаемости по полу
survival_rate_by_sex = df.groupby('Sex_male')['Survived'].mean()

# Замена значений True и False на Male и Female
survival_rate_by_sex.index = survival_rate_by_sex.index.map({True: 'Male', False: 'Female'})

# Вывод графика о проценте выживаемости по полу
survival_rate_by_sex.plot(kind='bar')
plt.title('Survival Rate by Gender')
plt.xlabel('Gender')
plt.ylabel('Survival Rate')
plt.show()
