# Импортируем необходимые библиотеки
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Загружаем данные
data = pd.read_csv('./spotify_songs.csv')

# Исключаем нечисловые столбцы перед построением корреляционной матрицы
numerical_data = data.select_dtypes(include=['float64', 'int64'])

# Построение тепловой карты корреляции
correlation_matrix = numerical_data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Корреляционная матрица")
plt.show()

# Выделяем числовые признаки и целевую переменную
numerical_features = data[['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms']]
target = data['track_popularity']

# Разделяем данные на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(numerical_features, target, test_size=0.2, random_state=42)

# Выбираем модель регрессии (в данном случае, линейную регрессию)
model = LinearRegression()

# Обучаем модель на обучающем наборе
model.fit(X_train, y_train)

# Предсказываем значения на тестовом наборе
y_pred = model.predict(X_test)

# Оцениваем производительность модели
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Выводим веса (коэффициенты) модели, которые показывают влияние каждого признака на популярность
coefficients = pd.DataFrame({'Feature': numerical_features.columns, 'Coefficient': model.coef_})
coefficients.sort_values(by='Coefficient', ascending=False, inplace=True)
print(coefficients)
