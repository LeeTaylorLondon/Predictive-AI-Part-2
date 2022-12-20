# Author: Lee Taylor, ST Number: 190211479
from Scripts.functions_   import grid_search, gen_data
from sklearn.linear_model import LinearRegression
from sklearn.ensemble     import RandomForestRegressor
from sklearn.tree         import DecisionTreeRegressor


# Define the datasets
X_train, X_test, y_train, y_test = gen_data(debug=False)


''' ------------------------------------- '''
''' ---- MODEL 1 : LINEAR REGRESSION ---- '''
# Define the hyperparameters to search over
hyperparams = {'fit_intercept': [True, False],
                'copy_X': [True, False],
                'n_jobs': [-1, 1, 2, 3, 4]}

# Test the model
print("Testing: Model 1 - LinearRegression\n")
best_params, best_score, r, bi = grid_search(LinearRegression, hyperparams,
                                               X_train, y_train, X_test, y_test,
                                               verbose=True)
# Print the results
print(f"Best Params: {best_params}, "
      f"Scores MSE & R^2: {r[bi].split(' ')[-2:]}")


''' ------------------------------------------ '''
''' ---- MODEL 2 : MULTI-LAYER-PERCEPTRON ---- '''
# # Define the hyperparameters to search over
# hyperparams = {'fit_intercept': [True, False],
#                 'copy_X': [True, False],
#                 'n_jobs': [-1, 1, 2, 3, 4]}
#
# # Test the model
# print("Testing: Model 2 - Multi-Layer-Perceptron")
# best_params, best_score, r, bi = grid_search(LinearRegression, hyperparams,
#                                                X_train, y_train, X_test, y_test,
#                                                verbose=True)
# # Print the results
# print(f"Best Params: {best_params}, "
#       f"Scores MSE & R^2: {r[bi].split(' ')[-2:]}")


''' --------------------------------- '''
''' ---- MODEL 3 : RANDOM FOREST ---- '''
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

# Test the model
print("\nTesting: Model 3 - RandomForestRegresson")
# Perform the grid search
best_params, best_score, r, bi = grid_search(RandomForestRegressor, hyperparams,
                                               X_train, y_train, X_test, y_test,
                                               verbose=True)

# Print the results
print(f"Best Params: {best_params}, "
      f"Scores MSE & R^2: {r[bi].split(' ')[-2:]}")


''' --------------------------------- '''
''' ---- MODEL 4 : DECISION TREE ---- '''
# Define the hyperparameters to search over
hyperparams = {'max_depth':                 [20, 30, 40, 50],
                'min_samples_split':        [20],
                'min_samples_leaf':         [8, 16, 32, 64],
                'min_weight_fraction_leaf': [0.0],
                'max_leaf_nodes':           [None]}

# Perform the grid search
best_params, best_score, r, bi = grid_search(DecisionTreeRegressor, hyperparams,
                                               X_train, y_train, X_test, y_test,
                                               verbose=True)

# Print the results
print(f"Best Params: {best_params}, "
      f"Scores MSE & R^2: {r[bi].split(' ')[-2:]}")
