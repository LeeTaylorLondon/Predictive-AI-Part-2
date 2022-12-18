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
epochs      = [10, 50]
param_grid = dict(units=units, activation=activations, layers=layers,
                  optimizer=optimizers, epochs=epochs)

# Create the model
model = KerasRegressor(build_fn=create_model, epochs=epochs, batch_size=16, verbose=0)

# Perform the grid search
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)
grid_result = grid.fit(X_train, y_train)

# Print the results of the grid search
print(f"Best: {grid_result.best_score_} using {grid_result.best_params_}")
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print(f"{mean} ({stdev}) with: {param}")


'''

Grid Search 1 results: 

Best: -0.219932 using {'activation': 'relu', 'layers': 2, 'units': 512}
-0.219932 (0.005488) with: {'activation': 'relu', 'layers': 2, 'units': 512}
-0.220033 (0.004069) with: {'activation': 'relu', 'layers': 2, 'units': 1024}
-0.225337 (0.010365) with: {'activation': 'relu', 'layers': 3, 'units': 512}
-0.241241 (0.027027) with: {'activation': 'relu', 'layers': 3, 'units': 1024}
-0.271150 (0.014250) with: {'activation': 'tanh', 'layers': 2, 'units': 512}
-0.267077 (0.015247) with: {'activation': 'tanh', 'layers': 2, 'units': 1024}
-0.274323 (0.009548) with: {'activation': 'tanh', 'layers': 3, 'units': 512}
-0.281756 (0.002757) with: {'activation': 'tanh', 'layers': 3, 'units': 1024}

Grid Search 2 results: 

Best: -0.2153350661198298 using {'activation': 'relu', 'layers': 2, 'units': 256}
-0.2257076303164164 (0.005610383694475058) with: {'activation': 'relu', 'layers': 2, 'units': 64}
-0.2153350661198298 (0.004047098886464389) with: {'activation': 'relu', 'layers': 2, 'units': 256}
-0.22635363539059958 (0.0055235757636850225) with: {'activation': 'relu', 'layers': 2, 'units': 512}
-0.22348946332931519 (0.0037692652185719083) with: {'activation': 'relu', 'layers': 3, 'units': 64}
-0.2185470163822174 (0.0032141666190944065) with: {'activation': 'relu', 'layers': 3, 'units': 256}
-0.2255500704050064 (0.004781145412296005) with: {'activation': 'relu', 'layers': 3, 'units': 512}
-0.22560536364714304 (0.004630681555513752) with: {'activation': 'relu', 'layers': 4, 'units': 64}
-0.2181053509314855 (0.006840586448255651) with: {'activation': 'relu', 'layers': 4, 'units': 256}
-0.22066317995389303 (0.007933670407982561) with: {'activation': 'relu', 'layers': 4, 'units': 512}
-0.22016904254754385 (0.005787989325121524) with: {'activation': 'relu', 'layers': 5, 'units': 64}
-0.22959243754545847 (0.006135902432048814) with: {'activation': 'relu', 'layers': 5, 'units': 256}
-0.2309460292259852 (0.010976929578112618) with: {'activation': 'relu', 'layers': 5, 'units': 512}
-0.2346449395020803 (0.003443314702353727) with: {'activation': 'tanh', 'layers': 2, 'units': 64}
-0.24718778332074484 (0.008642968678711664) with: {'activation': 'tanh', 'layers': 2, 'units': 256}
-0.2522334059079488 (0.004968353003027476) with: {'activation': 'tanh', 'layers': 2, 'units': 512}
-0.23588661352793375 (0.004143316236182298) with: {'activation': 'tanh', 'layers': 3, 'units': 64}
-0.24954255918661752 (0.00655520023442987) with: {'activation': 'tanh', 'layers': 3, 'units': 256}
-0.2743479708830516 (0.006677011607917516) with: {'activation': 'tanh', 'layers': 3, 'units': 512}
-0.2364360491434733 (0.008766370998344802) with: {'activation': 'tanh', 'layers': 4, 'units': 64}
-0.25851162274678546 (0.00943073712371849) with: {'activation': 'tanh', 'layers': 4, 'units': 256}
-0.2725006739298503 (0.015316740284002808) with: {'activation': 'tanh', 'layers': 4, 'units': 512}
-0.24615691602230072 (0.008237982697105703) with: {'activation': 'tanh', 'layers': 5, 'units': 64}
-0.2801317671934764 (0.016125180700005026) with: {'activation': 'tanh', 'layers': 5, 'units': 256}
-0.28620901703834534 (0.01854733789273122) with: {'activation': 'tanh', 'layers': 5, 'units': 512}

Grid Search 3 results: 

Best: -0.21311742067337036 using {'activation': 'relu', 'epochs': 50, 'layers': 2, 'optimizer': 'adam', 'units': 256}
-0.22185669839382172 (0.01106852511667019) with: {'activation': 'relu', 'epochs': 10, 'layers': 2, 'optimizer': 'adam', 'units': 256}
-0.21311742067337036 (0.010367324146148585) with: {'activation': 'relu', 'epochs': 50, 'layers': 2, 'optimizer': 'adam', 'units': 256}

'''
