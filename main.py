import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from dataset import replace_missing_with_mean_numeric
from model import create_predict_price_model
#import tensorflow as tf
def main():
    
    train_data = pd.read_csv('house-prices-advanced-regression-techniques/train.csv')
    train_data = replace_missing_with_mean_numeric(train_data)

    #print(train_data)

    target_variable = 'SalePrice'

    X = train_data.drop(target_variable, axis=1)
    y = train_data[target_variable]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    numerical_cols = X.select_dtypes(include=['int64']).columns
    categorical_cols = X.select_dtypes(include=['object']).columns

    model= create_predict_price_model(X_train,y_train, numerical_cols, categorical_cols)
    print("Model created and trained successfully")
    # Evaluate the model
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    target_variable_range = train_data[target_variable].agg(['min', 'max'])

    print("\nTarget Variable Range:", target_variable_range)
    print(f'\nMean Squared Error on Validation Set: {mse}')
    print(f"\nRoot Mean Squared Error (RMSE): {rmse}")

    test_data = pd.read_csv('house-prices-advanced-regression-techniques/test.csv')
    test_data = replace_missing_with_mean_numeric(test_data)

    test_predictions = model.predict(test_data)
    result_set = pd.DataFrame({'Id': test_data['Id'], 'SalePrice': test_predictions})

    result_set.to_csv('output/predictions.csv', index=False)

    print("\nPrices predicted successfully and saved in output/predictions.csv")

if __name__ == "__main__":
    main()


