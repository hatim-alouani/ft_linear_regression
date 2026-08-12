import pandas as pd
import json

df = pd.read_csv("data.csv")

mileage = df["km"].values
price = df["price"].values

m = len(price)

mileage_min = mileage.min()
mileage_max = mileage.max()

normalized_mileage = (mileage - mileage_min) / (mileage_max - mileage_min)

theta0 = 0
theta1 = 0

learning_rate = 0.01
iterations = 10000

# math formulat
# estimated_price = theta1 * mileage + theta0
# J = 1/2m * ​∑(estimated_price[i] − price[i])^2
# theta = theta - learning_rate * ∂J/∂theta
# ∂J/∂theta0 = 1/m * ∑(estimated_price[i] − price[i])
# ∂J/∂theta1 = 1/m * ∑mileage * (estimated_price[i] − price[i])
# tmp_theta1 = learning_rate * ∂J/∂theta1
# tmp_theta0 = learning_rate * ∂J/∂theta0
# theta1 = theta1 - tmp_theta1
# theta0 = theta0 - tmp_theta0

def gradient_descent(mileage, price, theta0, theta1, learning_rate, iterations):

    for _ in range(iterations):

        estimated_price = theta0 + theta1 * mileage

        error = estimated_price - price

        tmp_theta0 = learning_rate * (1 / m) * sum(error)

        tmp_theta1 = learning_rate * (1 / m) * sum(error * mileage)

        theta0 = theta0 - tmp_theta0
        theta1 = theta1 - tmp_theta1

    return theta0, theta1

theta0_normalized, theta1_normalized = gradient_descent(normalized_mileage, price, theta0, theta1, learning_rate, iterations)

theta1 = theta1_normalized / (mileage_max - mileage_min)

theta0 = theta0_normalized - (theta1_normalized * mileage_min / (mileage_max - mileage_min))

with open("theta.json", "w") as file:
    json.dump({ "theta0": theta0, "theta1": theta1 }, file)

print("Training completed.")
print(f"theta0: {theta0}")
print(f"theta1: {theta1}")