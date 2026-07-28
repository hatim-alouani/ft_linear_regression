import pandas as pd
import json

df = pd.read_csv("data.csv")

mileage = df["km"].values
price = df["price"].values

with open("theta.json", "r") as file:
    model = json.load(file)

theta0 = model["theta0"]
theta1 = model["theta1"]

predicted_price = theta0 + theta1 * mileage

m = len(price)

mse = sum((price - predicted_price) ** 2) / m

rmse = mse ** 0.5

print("========== Model Precision ==========\n")

print(f"Mean Squared Error (MSE)      : {mse:.2f}")

print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")