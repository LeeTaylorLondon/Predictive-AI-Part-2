# Author: Lee Taylor, ST Number: 190211479
try:
    from pre_processing  import *
except ModuleNotFoundError:
    from Scripts.pre_processing import *
from    sklearn.metrics import mean_squared_error
from    sklearn.metrics import r2_score
from    sklearn.metrics import accuracy_score
from    itertools       import product
import  numpy           as     np


def score_mse(model, X_test, y_test):
    """ Return the MSE score """
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")
    return mse

def score_r2(model, X_test, y_test):
    """ Return the R^2 score """
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    print(f"R^2 Score: {r2}")
    return r2

def score_model(model, X_test, y_test):
    return score_mse(model, X_test, y_test), score_r2(model, X_test, y_test)

def csvwriter_(results, model_x, test_iter):
    results_ = []
    for line in results:
        results_.append(','.join(line.split()) + '\n')
    try:
        fd = f"../Output/m{model_x}/{test_iter}.csv"
        with open(fd, 'w+') as f:
            f.writelines(results_)
    except FileNotFoundError as e:
        print(f"Did not write to {fd}. FileNotFound!")

def grid_search(model, hyperparams, X_train, y_train, X_test, y_test, verbose=True):
    """ Perform grid search on a given model with a given set of hyperparameters.
    Using a dictionary containing the hyperparameters to be searched over. The keys
    should be the names of the hyperparameters and the values should be lists
    of values to try.
    """
    # Initialize lists to store the results of each model
    scores = []
    params = []
    score_ = []

    # Get the names of the hyperparameters
    param_names = list(hyperparams.keys())

    # Create a list of lists of values to try for each hyperparameter
    param_values = list(hyperparams.values())

    # Use itertools.product to generate all combinations of hyperparameter values
    for values in product(*param_values):
        # Create a dictionary mapping hyperparameter names to values
        param_dict = dict(zip(param_names, values))
        # Create the model with the current set of parameters
        model_instance = model(**param_dict)
        # Fit the model to the training data
        model_instance.fit(X_train, y_train)
        # Evaluate the model on the test data
        y_pred = model_instance.predict(X_test)
        # Calculate the MSE score
        mse = mean_squared_error(y_test, y_pred)
        r2  = r2_score(y_test, y_pred)
        # Store the results
        score_.append([mse, r2])
        scores.append(mse)
        params.append(values)

    # Find the index of the model with the lowest MSE
    best_index = np.argmin(scores)

    results = []
    # Record the result of each model instance
    if verbose: print(f"| {' | '.join(hyperparams.keys())} | MSE | R^2 |")
    for p, s in zip(params, score_):
        p_, s_ = [str(v) for v in p], [str(round(v, 6)) for v in s]
        re_line = ' '
        re_line = re_line.join(p_) + ' '
        re_line = re_line + ' '.join(s_)
        if verbose: print(f"{re_line}")
        results.append(re_line)

    # Return the best parameters and the best score
    return params[best_index], scores[best_index], results