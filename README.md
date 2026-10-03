# House Price Prediction Using Linear Regression

## Project Overview

This project is part of Task 01 of the SkillCraft Technology Machine Learning internship. It uses a Linear Regression model to predict house prices based on selected housing features.

## Objective

To build a machine learning model that predicts house prices using the following three features:

- **GrLivArea:** Above-ground living area in square feet.
- **BedroomAbvGr:** Number of bedrooms above ground.
- **FullBath:** Number of full bathrooms.

## Technologies Used

- Python
- Pandas
- Scikit-learn
- Matplotlib
- Joblib

## Project Structure

```text
ML_1/
├── data/
│   ├── train.csv
│   └── test.csv
├── src/
│   ├── train_model.py
│   └── predict.py
├── model/
│   └── house_price_model.pkl
├── house_price_predictions.png
├── submission.csv
├── requirements.txt
└── README.md
```

## Methodology

1. Load the housing dataset using Pandas.
2. Select the three input features and the target variable, `SalePrice`.
3. Split the training data into training and validation sets.
4. Train a Linear Regression model.
5. Evaluate the model using Mean Absolute Error (MAE) and R² score.
6. Visualize actual versus predicted house prices.
7. Save the trained model and generate predictions for the test dataset.

## Model Evaluation

The model achieved the following results on the validation dataset:

- **Mean Absolute Error (MAE):** 35,788.06
- **R² Score:** 0.6341

The MAE represents the average absolute difference between actual and predicted prices. The R² score indicates how much variation in house prices is explained by the model on the validation data.

## Visualization

The file `house_price_predictions.png` contains a scatter plot comparing actual and predicted house prices. The black diagonal line represents perfect predictions.

## How to Run

Install the required libraries:

```bash
pip install -r requirements.txt
```

Train the model:

```bash
python src/train_model.py
```

Generate predictions:

```bash
python src/predict.py
```

The trained model is saved in the `model/` directory, the graph is saved as `house_price_predictions.png`, and test predictions are saved in `submission.csv`.

## Conclusion

This project demonstrates the basic machine learning workflow of data loading, feature selection, model training, evaluation, visualization, and prediction using Linear Regression.

## Internship

**Organization:** SkillCraft Technology  
**Task:** Task 01 – House Price Prediction Using Linear Regression