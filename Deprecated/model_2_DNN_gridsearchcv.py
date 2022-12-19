# Author: Lee Taylor, ST Number: 190211479
import  numpy                                   as      np
import  tensorflow                              as      tf
from    pre_processing                          import  *
from    sklearn.model_selection                 import  GridSearchCV
from    tensorflow.keras.wrappers.scikit_learn  import  KerasRegressor


# Define the datasets
X_train, X_test, y_train, y_test = gen_data()

# Define the model
def create_model(units=512, activation='relu', layers=2, optimizer='adam'):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(units, input_shape=(8,), activation=activation))
    for layer in range(layers):
        model.add(tf.keras.layers.Dense(units, activation=activation))
    model.add(tf.keras.layers.Dense(1))
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])
    return model

# Define the grid search parameters
layers      = [2]
units       = [256]
activations = ['relu']
optimizers  = ['adam']
epochs      = [25]
batch_size  = [256]
param_grid = dict(units=units, activation=activations, layers=layers,
                  optimizer=optimizers, epochs=epochs, batch_size=batch_size)

# Create the model
model = KerasRegressor(build_fn=create_model, epochs=epochs, batch_size=batch_size, verbose=0)

# Perform the grid search
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=2)
grid_result = grid.fit(X_train, y_train)

# Print the results of the grid search
print(f"Best: {grid_result.best_score_} using {grid_result.best_params_}")
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print(f"{mean} ({stdev}) with: {param}")


'''


'''
