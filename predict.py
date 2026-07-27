import json

with open("theta.json", "r") as file:
    model = json.load(file)

theta0 = model["theta0"]
theta1 = model["theta1"]

mileage = float(input("Enter the mileage of the car: "))

estimated_price = theta0 + theta1 * mileage

print(f"Estimated price: {estimated_price:.2f}")