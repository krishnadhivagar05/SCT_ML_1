import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

data = pd.read_csv("data/train.csv")

features = ["GrLivArea", "BedroomAbvGr", "FullBath"]

X = data[features]
y = data["SalePrice"]

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_val)

print("Model Training Completed!")
print("Mean Absolute Error:", mean_absolute_error(y_val, predictions))
print("R2 Score:", r2_score(y_val, predictions))

plt.figure(figsize=(10, 6))

plt.scatter(
    y_val,
    predictions,
    alpha=0.5,
    label="Predicted Prices"
)

min_price = min(y_val.min(), predictions.min())
max_price = max(y_val.max(), predictions.max())

plt.plot(
    [min_price, max_price],
    [min_price, max_price],
    color="black",
    linestyle=":",
    linewidth=2,
    label="Perfect Prediction Line"
)

plt.xlabel("Actual House Prices")
plt.ylabel("Predicted House Prices")
plt.title("Actual vs Predicted House Prices")

plt.grid(True, linestyle="--", alpha=0.5)

plt.legend()
plt.tight_layout()

plt.savefig("house_price_predictions.png", dpi=150)
plt.close()

print("Graph saved successfully!")
print("File location: house_price_predictions.png")

os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/house_price_model.pkl")

print("Model saved successfully!")
