# Author: Lee Taylor, ST Number: 190211479
from pre_processing             import *
from sklearn.linear_model       import LinearRegression
from sklearn.model_selection    import KFold
from sklearn.metrics            import r2_score
from sklearn.metrics            import mean_squared_error


# Init. train/test datasets
X_train, X_test, y_train_true, y_test_true = gen_data()

# Create model
np.random.seed(100)
m1 = LinearRegression()

# Fit model
m1.fit(X_train.values, y_train_true.values)

# Test model
y_test_pred = m1.predict(X_test.values)
print(f"-------------------------------")
print("MSE: {0:.3}".format(mean_squared_error(y_test_true, y_test_pred)))
print("R^2: {0:.2}".format(r2_score(y_test_true, y_test_pred)))
print(f"-------------------------------")

