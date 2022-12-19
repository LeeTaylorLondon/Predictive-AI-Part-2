# Author: Lee Taylor, ST Number: 190211479
import numpy as np
from sklearn.tree import DecisionTreeRegressor

# Define the datasets
X_train, X_test, y_train, y_test = gen_data()

# Define the grid search parameters
max_depth = [None, 10, 20, 30, 40, 50]
min_samples_split = [2, 5, 10, 15, 20]
min_samples_leaf = [1, 2, 4, 8]
min_weight_fraction_leaf = [0.0, 0.1, 0.2, 0.3]
max_leaf_nodes = [None, 10, 20, 30, 40, 50]

# Initialize lists to store the results of each model
scores = []
params = []

# Iterate over all combinations of model parameters
for d in max_depth:
    for s in min_samples_split:
        for l in min_samples_leaf:
            for w in min_weight_fraction_leaf:
                for n in max_leaf_nodes:
                    # Create the model with the current set of parameters
                    model = DecisionTreeRegressor(max_depth=d, min_samples_split=s, min_samples_leaf=l, min_weight_fraction_leaf=w, max_leaf_nodes=n)
                    # Fit the model to the training data
                    model.fit(X_train, y_train)
                    # Evaluate the model on the test data
                    score = model.score(X_test, y_test)
                    # Store the results
                    scores.append(score)
                    params.append((d, s, l, w, n))

# Find the index of the model with the highest score
best_index = np.argmax(scores)

# Print the best parameters and best score
print("Best parameters: ", params[best_index])
print("Best score: ", scores[best_index])