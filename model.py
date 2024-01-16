from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

def create_predict_price_model(X_train, X_test,y_train):
    # Standardize the features (optional but often recommended)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Create a linear regression model
    model = LinearRegression()

    # Train the model
    model.fit(X_train_scaled, y_train)
    return model, scaler,X_test_scaled