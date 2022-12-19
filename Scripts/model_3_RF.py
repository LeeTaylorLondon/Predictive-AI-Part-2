# Author: Lee Taylor, ST Number: 190211479
import  numpy               as      np
from    sklearn.ensemble    import  RandomForestRegressor
from    pre_processing      import  *

# Define the datasets
X_train, X_test, y_train, y_test = gen_data()

# Define the grid search parameters
n_estimators = [10, 50, 100, 200, 300]
max_depth = [None, 10, 20, 30, 40, 50]
min_samples_split = [2, 5, 10, 15, 20]
min_samples_leaf = [1, 2, 4, 8]

# Initialize lists to store the results of each model
scores = []
params = []

# Iterate over all combinations of model parameters
for n in n_estimators:
    for d in max_depth:
        for s in min_samples_split:
            for l in min_samples_leaf:
                # Create the model with the current set of parameters
                model = RandomForestRegressor(n_estimators=n, max_depth=d, min_samples_split=s, min_samples_leaf=l)
                # Fit the model to the training data
                model.fit(X_train, y_train)
                # Evaluate the model on the test data
                score = model.score(X_test, y_test)
                # Store the results
                scores.append(score)
                params.append((n, d, s, l))

# Find the index of the model with the highest score
best_index = np.argmax(scores)

# Print the best parameters and best score
print("Best parameters: ", params[best_index])
print("Best score: ", scores[best_index])