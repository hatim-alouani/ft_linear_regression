import pandas as pd
import matplotlib.pyplot as plt
import json

df = pd.read_csv("data.csv")

mileage = df["km"].values
price = df["price"].values

with open("theta.json", "r") as file:
    model = json.load(file)

theta0 = model["theta0"]
theta1 = model["theta1"]

predicted_price = theta0 + theta1 * mileage

plt.scatter(mileage, price, label="Dataset", color="blue")

plt.plot(mileage, predicted_price, label="Regression Line", color="red")

plt.xlabel("Mileage (km)")
plt.ylabel("Price")
plt.title("Linear Regression")
plt.legend()

plt.show()