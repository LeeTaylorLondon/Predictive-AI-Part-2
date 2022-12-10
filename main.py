# Author: Lee Taylor, ST Number: 190211479
import numpy  as np
import pandas as pd
from sklearn.linear_model       import LinearRegression
from sklearn.model_selection    import KFold
from sklearn.metrics            import r2_score
from sklearn.metrics            import mean_squared_error
from sklearn.model_selection    import train_test_split


# Configure pandas
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# Load houses dataset
df = pd.read_csv("houses.csv")
print(f"{df.shape}\n\n{df.columns}")

# Print correlation matrix
corr = df.corr()
print(corr)


# np.random.seed(42)
# X_train, X_test, y_train_true, y_test_true = train_test_split(X, y, test_size=0.33, random_state=42)


if __name__ == '__main__':
    pass