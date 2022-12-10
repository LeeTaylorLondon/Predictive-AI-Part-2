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
df = pd.read_csv("houses.csv", header=0)
print(f"{df.shape}\n\n{df.columns}")

''' A number of the features are such things as 'total_rooms' and 'total_bedrooms'.
This is for all houses in the area. As the size of the areas may be different it
is worth normalising these based on the number of properties in the area. 
[Drop column example ->] X_train_ = imputed_dataset.copy().drop([target, 'index'], axis=1) '''

# Normalize columns
total_rooms_norm    = df['total_rooms'] / df['households']
total_bedrooms_norm = df['total_bedrooms'] / df['households']

'''
        >>> df = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3', 'K4', 'K5'],
        ...                    'A': ['A0', 'A1', 'A2', 'A3', 'A4', 'A5']})
'''
# Modify dataset
norms = pd.DataFrame({'total_rooms_norm'    : total_rooms_norm,
                      'total_bedrooms_norm' : total_bedrooms_norm})
df = df.join(norms)
df = df.drop(columns=['total_rooms', 'total_bedrooms'])
print(f"{df.shape}\n\n{df.columns}")


# np.random.seed(42)
# X_train, X_test, y_train_true, y_test_true = train_test_split(X, y, test_size=0.33, random_state=42)


if __name__ == '__main__':
    pass