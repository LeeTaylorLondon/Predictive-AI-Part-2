# Author: Lee Taylor, ST Number: 190211479
from    sklearn.ensemble    import  RandomForestRegressor
from    functions_          import  *

# Define the datasets
X_train, X_test, y_train, y_test = gen_data(debug=False)

''' First group of hyperparams to test '''
# # Define the hyperparameters to search over
# hyperparams = {"n_estimators": [10, 50, 100, 200, 300],
#                "max_depth": [None, 10, 20, 30, 40, 50],
#                "min_samples_split": [2, 5, 10, 15, 20],
#                "min_samples_leaf": [1, 2, 4, 8]}
''' Second group of hyperparams to test '''
# Define the hyperparameters to search over
# hyperparams = {"n_estimators": [300],
#                "max_depth": [None],
#                "min_samples_split": [2],
#                "min_samples_leaf": [2],
#                "bootstrap": [True],
#                "oob_score": [True, False],
#                "warm_start": [True, False],
#                "min_impurity_decrease": [0, 0.1, 0.2, 0.3, 0.4, 0.5],
#                "ccp_alpha": [0, 0.1, 0.2, 0.3, 0.4, 0.5]
#                }
''' Third group of hyperparams to test '''
# Define the hyperparameters to search over
hyperparams = {"n_estimators": [300],
               "max_depth": [None],
               "min_samples_split": [2],
               "min_samples_leaf": [2],
               "bootstrap": [True],
               "oob_score": [True, False],
               "warm_start": [True, False],
               "min_impurity_decrease": [0],
               "ccp_alpha": [0]
               }

# Perform the grid search
best_params, best_score, results, _ = grid_search(RandomForestRegressor, hyperparams,
                                               X_train, y_train, X_test, y_test,
                                               verbose=True)

# Write the results to a CSV file
csvwriter_(results, "3", "3")

# Print the results
print()
print("Best parameters: ", best_params)
print("Best score: ", best_score)