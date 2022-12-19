# Author: Lee Taylor, ST Number: 190211479
from    sklearn.ensemble    import  RandomForestRegressor
from    pre_processing      import  *
from    functions_ import *

# Define the datasets
X_train, X_test, y_train, y_test = gen_data(debug=False)

# Define the hyperparameters to search over
hyperparams = {"n_estimators": [10, 50, 100, 200, 300],
               "max_depth": [None, 10, 20, 30, 40, 50],
               "min_samples_split": [2, 5, 10, 15, 20],
               "min_samples_leaf": [1, 2, 4, 8]}

# Perform the grid search
best_params, best_score, results = grid_search(RandomForestRegressor, hyperparams,
                                               X_train, y_train, X_test, y_test,
                                               verbose=True)

# Write the results to a CSV file
csvwriter_(results, "3", "1")

# Print the results
print()
print("Best parameters: ", best_params)
print("Best score: ", best_score)