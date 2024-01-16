import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from dataset import treat_dataset
from model import create_predict_price_model
#import tensorflow as tf
def main():
        
    df = pd.read_csv('house-prices-advanced-regression-techniques/train.csv')
    df = treat_dataset(df)
    for column, dtype in df.dtypes.items():
        if dtype != 'object':
            print(f"{column}: {dtype}")

    selected_features = ['MSSubClass', 'LotFrontage','LotArea','OverallQual','OverallCond','YearBuilt',  
                        'YearRemodAdd','MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF',
                        '1stFlrSF','2ndFlrSF','LowQualFinSF','GrLivArea','BsmtFullBath','BsmtHalfBath',  
                        'FullBath','HalfBath','KitchenAbvGr','KitchenAbvGr','TotRmsAbvGrd','Fireplaces', 
                        'GarageYrBlt','GarageCars','GarageArea','WoodDeckSF','OpenPorchSF','EnclosedPorch',  
                        '3SsnPorch','ScreenPorch','PoolArea','MiscVal','MoSold','YrSold','SalePrice',  
                        'PoolQC_encoded','Fence_encoded','PavedDrive_encoded',
                        ]
    target_variable = 'SalePrice'

    X = df[selected_features]
    y = df[target_variable]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model, scaler,X_test_scaled = create_predict_price_model(X_train,X_test,y_train)
    # Make predictions on the test set
    predictions = model.predict(X_test_scaled)

    # Evaluate the model
    mse = mean_squared_error(y_test, predictions)
    rmse = mse**0.5
    print(f"Root Mean Squared Error (RMSE): {rmse}")

    new_data = pd.DataFrame([[8,1,2,3,4,5,6,7,8,9,10,1,1,1,1,1,1,1,1,1,1,1,1, 2000, 2, 1000, 1200, 4,2, 6, 1990, 2005, 1, 800, 1, 200, 500,2010, 1, 8
                            ]], columns=selected_features)

    new_data_scaled = scaler.transform(new_data)
    predicted_price = model.predict(new_data_scaled)
    print(f"Predicted Sale Price for new data: {predicted_price[0]}")

if __name__ == "__main__":
    main()

    #print(df)

