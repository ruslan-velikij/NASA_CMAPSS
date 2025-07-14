print("# Шаг 1: Подготовка данных")
# 1.A. Загрузка необходимых библиотек
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 1.B. Загрузка обучающих данных
col_names = ["id", "cycle", "setting1", "setting2", "setting3"] + [f"sensor{i}" for i in range(1, 22)]
train_df = pd.read_csv("datasets/train_FD001.txt", sep=r'\s+', header=None, names=col_names)
print("Размер обучающего набора:", train_df.shape)
print("Первые 5 строк:")
print(train_df.head())

# 1.C. Создание целевого столбца RUL для обучающего набора
max_cycle = train_df.groupby("id")["cycle"].max()
train_df = train_df.merge(max_cycle.to_frame(name="max_cycle"), on="id")
train_df["RUL"] = train_df["max_cycle"] - train_df["cycle"]
train_df.drop("max_cycle", axis=1, inplace=True)
print("\nПосле добавления RUL:")
print(f'{train_df[["id", "cycle", "RUL"]].head(10)}\n')


print("# Шаг 2: Разделение на обучающую и валидационную выборку 80/20%")
# 2.A. Разделение идентификаторов двигателей на train и validation
unique_ids = train_df["id"].unique()
np.random.seed(42)
val_ids = np.random.choice(unique_ids, size=20, replace=False)
val_ids = list(val_ids)
print("Выбраны двигатели для валидации:", val_ids[:5], "..., всего", len(val_ids))
train_ids = [i for i in unique_ids if i not in val_ids]
print("Двигателей в обучающей части:", len(train_ids))

# 2.B. Формирование обучающей и валидационной выборки
train_data = train_df[train_df["id"].isin(train_ids)].copy()
val_data   = train_df[train_df["id"].isin(val_ids)].copy()
print("Форма обучающих данных:", train_data.shape)
print(f'Форма валидационных данных: {val_data.shape}\n')


print("# Шаг 3: Выбор признаков")
# 3.A. Определение списка столбцов, которые будут исключены
cols_to_drop = ["id", "cycle", "setting3", "sensor1", "sensor5", "sensor10", "sensor16", "sensor18", "sensor19"]
X_train = train_data.drop(cols_to_drop + ["RUL"], axis=1)
y_train = train_data["RUL"]
X_val   = val_data.drop(cols_to_drop + ["RUL"], axis=1)
y_val   = val_data["RUL"]
print("Число признаков после удаления лишних столбцов:", X_train.shape[1])
print(f'Имена оставшихся признаков: {X_train.columns.tolist()}\n')


print("# Шаг 4: Обучение регрессионной модели\n")
# 4.A. Инициализация модели
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# 4.B. Обучение (тренировка) модели на обучающих данных
rf_model.fit(X_train, y_train)


print("# Шаг 5: Оценка качества модели на валидации")
# 5.A. Предсказание на валидационных данных
val_predictions = rf_model.predict(X_val)

# 5.B. Оценка качества с помощью метрик
mse_val = mean_squared_error(y_val, val_predictions)
rmse_val = np.sqrt(mean_squared_error(y_val, val_predictions))
mae_val = mean_absolute_error(y_val, val_predictions)
print(f"Среднеквадратичная ошибка (MSE) на валидации: {mse_val:.2f}")
print(f"Корень из MSE (RMSE) на валидации: {rmse_val:.2f} циклов")
print(f"Средняя абсолютная ошибка (MAE) на валидации: {mae_val:.2f} циклов\n")

# RMSE ~30–40 циклов означает, что в среднем модель ошибается на 30–40 циклов от фактического RUL.
# MAE ~25–30 циклов говорит, что в среднем модуль ошибки – несколько десятков циклов.
# Для пробной модели это ожидаемо.
# Random Forest, вероятно, не идеально предсказывает RUL, но важно было получить рабочий процесс.


print("# Шаг 6: Оценка качества модели на валидации")
# 6.A. Загрузка тестового набора и реальных RUL
test_df = pd.read_csv("datasets/test_FD001.txt", sep=r'\s+', header=None, names=col_names)
rul_df = pd.read_csv("datasets/RUL_FD001.txt", sep=r'\s+', header=None, names=["RUL"])
print("Тестовых последовательностей (двигателей):", len(test_df["id"].unique()))
print("Пример первых строк тестового набора:")
print(test_df.head(5)[["id", "cycle", "setting1", "sensor2", "sensor3", "sensor4"]])

# 6.B. Отбор последних записей каждого двигателя в тестовом наборе
test_sorted = test_df.sort_values(["id", "cycle"])
test_last = test_sorted.drop_duplicates(subset="id", keep="last")
print("\nФорма test_last:", test_last.shape)
print(test_last[["id", "cycle"]].head(10))

# 6.C. Подготовка признаков для прогноза (аналогично обучению)
X_test = test_last.drop(cols_to_drop, axis=1)
print("Признаки для тестовых данных:", X_test.columns.tolist())

# 6.D. Применение модели для предсказания RUL на тестовых данных
test_preds = rf_model.predict(X_test)

# 6.E. Извлечение реальных значений RUL из загруженного файла
y_test_true = rul_df["RUL"].values


print("# Шаг 7: Сравнение предсказаний с реальными RUL")
# 7.A. Вычисление метрик ошибки на тестовом наборе
rmse_test = mean_squared_error(y_test_true, test_preds, squared=False)
mae_test = mean_absolute_error(y_test_true, test_preds)
print(f"RMSE на тестовом наборе: {rmse_test:.2f} циклов")
print(f"MAE на тестовом наборе: {mae_test:.2f} циклов")

# 7.B. Примеры: сравнение для первых 20 двигателей
for i in range(1, 20):
    actual = y_test_true[i-1]
    predicted = test_preds[i-1]
    print(f"Двигатель {i}: реальный RUL = {actual}, предсказанный RUL ≈ {predicted:.1f}")