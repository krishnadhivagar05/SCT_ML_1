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

plt.figure(figsize=(8, 5))

plt.scatter(
    y_val,
    predictions,
    alpha=0.5,
    label="Predicted Prices"
)

plt.plot(
    [y_val.min(), y_val.max()],
    [y_val.min(), y_val.max()],
    color="black",
    linewidth=2,
    label="Perfect Prediction Line"
)

plt.xlabel("Actual House Prices")
plt.ylabel("Predicted House Prices")
plt.title("Actual vs Predicted House Prices")
plt.legend()
plt.tight_layout()

plt.savefig("house_price_predictions.png")
plt.close()

print("Graph saved successfully!")
print("File location: house_price_predictions.png")

os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/house_price_model.pkl")

print("Model saved successfully!")
