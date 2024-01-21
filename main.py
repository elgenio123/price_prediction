import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from dataset_house import treat_dataset
from model import create_predict_price_model
#import tensorflow as tf
def main():

    print("Predictiong prices")
        
    train_data = pd.read_csv('house-prices-advanced-regression-techniques/train.csv')
    train_data = treat_dataset(train_data)

    selected_features = ['MSSubClass', 'LotFrontage','LotArea','OverallQual','OverallCond','YearBuilt',  
                        'YearRemodAdd','MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF',
                        '1stFlrSF','2ndFlrSF','LowQualFinSF','GrLivArea','BsmtFullBath','BsmtHalfBath',  
                        'FullBath','HalfBath','KitchenAbvGr','KitchenAbvGr','TotRmsAbvGrd','Fireplaces', 
                        'GarageYrBlt','GarageCars','GarageArea','WoodDeckSF','OpenPorchSF','EnclosedPorch',  
                        '3SsnPorch','ScreenPorch','PoolArea','MiscVal','MoSold','YrSold',  
                        'PoolQC_encoded','Fence_encoded','PavedDrive_encoded',
                        ]
    target_variable = 'SalePrice'

    X = train_data[selected_features]
    y = train_data[target_variable]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model= create_predict_price_model(X_train,y_train)

    # Evaluate the model
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f'Mean Squared Error on Validation Set: {mse}')

    test_data = pd.read_csv('house-prices-advanced-regression-techniques/test.csv')
    test_data = treat_dataset(test_data)

    test_predictions = model.predict(test_data[selected_features])
    result_set = pd.DataFrame({'Id': test_data['Id'], 'SalePrice': test_predictions})

    result_set.to_csv('output/predictions.csv', index=False)

    print("Prices predicted successfully and saved in output/predictions.csv")

if __name__ == "__main__":
    main()


