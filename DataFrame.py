import pandas as pd
import numpy as np

col_names = ["id", "cycle", "setting1", "setting2", "setting3"] + [f"sensor{i}" for i in range(1, 22)]
train_df = pd.read_csv("datasets/train_FD001.txt", sep=r'\s+', header=None, names=col_names)
print("Размер обучающего набора:", train_df.shape)
print("Первые 5 строк:")
print(train_df.head())

max_cycle = train_df.groupby("id")["cycle"].max()
train_df = train_df.merge(max_cycle.to_frame(name="max_cycle"), on="id")
train_df["RUL"] = train_df["max_cycle"] - train_df["cycle"]
train_df.drop("max_cycle", axis=1, inplace=True)
print("\nПосле добавления RUL:")
print(train_df[["id", "cycle", "RUL"]].head(10))