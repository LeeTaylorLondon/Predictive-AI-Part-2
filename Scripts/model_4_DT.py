# Author: Lee Taylor, ST Number: 190211479
from    sklearn.tree        import DecisionTreeRegressor
from    functions_          import *

# Define the datasets
X_train, X_test, y_train, y_test = gen_data(debug=False)

''' First group of hyperparams to test '''
# Define the hyperparameters to search over
# hyperparams = {'max_depth':                 [None, 10, 20, 30, 40, 50],
#                 'min_samples_split':        [2, 5, 10, 15, 20],
#                 'min_samples_leaf':         [1, 2, 4, 8],
#                 'min_weight_fraction_leaf': [0.0, 0.1, 0.2, 0.3],
#                 'max_leaf_nodes':           [None, 10, 20, 30, 40, 50]}
''' Second group of hyperparams to test '''
# Define the hyperparameters to search over
hyperparams = {'max_depth':                 [20, 30, 40, 50],
                'min_samples_split':        [20],
                'min_samples_leaf':         [8, 16, 32, 64],
                'min_weight_fraction_leaf': [0.0],
                'max_leaf_nodes':           [None]}

# Perform the grid search
best_params, best_score, results, _ = grid_search(DecisionTreeRegressor, hyperparams,
                                               X_train, y_train, X_test, y_test,
                                               verbose=True)

# Write the results to a CSV file
csvwriter_(results, "4", "3")

# Print the results
print()
print("Best parameters: ", best_params)
print("Best score: ", best_score)