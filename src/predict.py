
import pandas as pd
import joblib

test_data = pd.read_csv("data/test.csv")

features = ["GrLivArea", "BedroomAbvGr", "FullBath"]
X_test = test_data[features]

model = joblib.load("model/house_price_model.pkl")

predictions = model.predict(X_test)

submission = pd.DataFrame({
    "Id": test_data["Id"],
    "SalePrice": predictions
})

submission.to_csv("submission.csv", index=False)

print("Predictions completed!")
print("Total predictions:", len(submission))
print("Saved to submission.csv")
print(submission.head())
