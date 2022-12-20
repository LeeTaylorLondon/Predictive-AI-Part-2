# Author: Lee Taylor, ST Number: 190211479
from sklearn.linear_model import LinearRegression
from functions_ import *

# Define the datasets
X_train, X_test, y_train, y_test = gen_data(debug=False)

# Define the hyperparameters to search over
hyperparams = {'fit_intercept': [True, False],
                'copy_X': [True, False],
                'n_jobs': [-1, 1, 2, 3, 4]}

# Perform the grid search
best_params, best_score, results, _ = grid_search(LinearRegression, hyperparams,
                                               X_train, y_train, X_test, y_test,
                                               verbose=True)

# Write the results to a CSV file
csvwriter_(results, "1", "5")

# Print the results
print()
print("Best parameters: ", best_params)
print("Best score: ", best_score)

# results_ = []
# for line in results[1:]:
#     results_.append(','.join(line.split())+'\n')
#
# try:
#     fd = "../Output/m1/1.csv"
#     with open(fd, 'w+') as f:
#         f.writelines(results_)
# except FileNotFoundError as e:
#     print(f"Did not write to {fd}. FileNotFound!")
#
# # Print the results
# print()
# print("Best parameters: ", best_params)
# print("Best score: ", best_score)