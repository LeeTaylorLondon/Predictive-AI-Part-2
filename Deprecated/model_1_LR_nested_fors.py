# Author: Lee Taylor, ST Number: 190211479
from pre_processing             import *
from sklearn.linear_model       import LinearRegression
from sklearn.model_selection    import KFold
from sklearn.metrics            import r2_score
from sklearn.metrics            import mean_squared_error


# Define the datasets
X_train, X_test, y_train, y_test = gen_data(debug=False)

# Create model
np.random.seed(100)

# Define the grid search parameters
fit_intercept = [True, False]
normalize = [True, False] # Deprecated - WARNED NOT TO USE
copy_X = [True, False]
n_jobs = [-1, 1, 2, 3, 4]

# Initialize lists to store the results of each model
scores = []
params = []
score_ = []

# Iterate over all combinations of model parameters
for f in fit_intercept:
    for c in copy_X:
        for j in n_jobs:
            # Create the model with the current set of parameters
            model = LinearRegression(fit_intercept=f, copy_X=c, n_jobs=j)
            # Fit the model to the training data
            model.fit(X_train, y_train)
            # Evaluate the model on the test data
            y_pred = model.predict(X_test)
            # Calculate the MSE score
            mse = mean_squared_error(y_test, y_pred)
            r2 = model.score(X_test, y_test)
            # Store the results
            scores.append(mse)
            score_.append([round(mse, 6), round(r2, 6)])
            params.append((f, c, j))

# Find the index of the model with the highest score
best_index = np.argmax(scores)

# Print the best parameters and best score
print("Best parameters: ", params[best_index])
print("Best score: ", scores[best_index])
print()

lines = []
# Print all results
for p, s in zip(params, score_):
    print(f"Fit_Intercept: {p[0]}, Copy_X: {p[1]}, "
          f"N_jobs: {p[2]}, "
          f"MSE: {s[0]}, R^2: {s[1]}")
    lines.append([p[0], p[1], p[2], s[0], s[1]])


