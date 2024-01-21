from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

def create_predict_price_model(X_train,y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    return model