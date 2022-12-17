# Author: Lee Taylor, ST Number: 190211479
import numpy  as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import scale


# Configure pandas
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# Load houses dataset
df = pd.read_csv("../Data/houses.csv", header=0)
print(f"{df.shape}\n\n{df.columns}")

# Normalize columns
total_rooms_norm    = df['total_rooms'] / df['households']
total_bedrooms_norm = df['total_bedrooms'] / df['households']

# Modify dataset
norms = pd.DataFrame({'total_rooms_norm'    : total_rooms_norm,
                      'total_bedrooms_norm' : total_bedrooms_norm})
df = df.join(norms)
df = df.drop(columns=['total_rooms', 'total_bedrooms'])
print(f"{df.shape}\n\n{df.columns}")

def gen_data():
    """ Return Train-Test datasets """
    X = df.copy().drop(columns=['median_house_value'], axis=1)
    y = df['median_house_value']
    np.random.seed(42)
    # function returns: X_train, X_test, y_train, y_test
    X_train, X_test, y_train_true, y_test_true = train_test_split(X, y, test_size=0.33, random_state=42)
    return X_train, X_test, y_train_true, y_test_true


if __name__ == '__main__':
    pass