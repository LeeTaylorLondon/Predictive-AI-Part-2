from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score


def score_mse(model, X_test, y_test):
    """ Return the MSE score """
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")
    return mse

def score_r2(model, X_test, y_test):
    """ Return the R^2 score """
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    print(f"R^2 Score: {r2}")
    return r2

def score_model(model, X_test, y_test):
    return score_mse(model, X_test, y_test), score_r2(model, X_test, y_test)