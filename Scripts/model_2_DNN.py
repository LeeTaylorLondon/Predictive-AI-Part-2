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
units       = [512]     # , 128, 256, 512
activations = ['relu']  # , 'tanh'
layers      = [2]       # , 3, 4, 5
optimizers  = ['adam', 'rmsprop', 'adagrad', 'adamax']
epochs      = [10, 25, 50]
batch_size  = [64]      # 16, 32, 64, 128, 256
hyper_param = [layers, units, activations, optimizers, epochs, batch_size]
hyp_par_fit = [epochs, batch_size]

# Create all combinations of models - maybe epochs and BaSi cannot go here
models, m_desc, scores = [], [], []
for u in units:
    for a in activations:
        for l in layers:
            for o in optimizers:
                for e in epochs:
                    for b in batch_size:
                        models.append(create_model(u, a, l, o))
                        m_desc.append(f"Units: {u}, Act: {a}, Layers: {l}, Opt: {o}, Epochs: {e}, Batch_Size: {b}")

mptn = -1
while mptn != len(models) - 1:
    for i,e in enumerate(epochs):
        for j,b in enumerate(batch_size):
            mptn += 1
            model = models[mptn]
            model.fit(X_train, y_train, epochs=e, batch_size=b, verbose=0)
            score = model.evaluate(X_test, y_test, verbose=0)
            score = [round(score[0], 6), round(score[1], 6)]
            scores.append(score)

for desc, scor in zip(m_desc, scores):
    print(f"|Description|: {desc}, |Scores|: {scor}")

model = models[0]
print(model.metrics_names)


'''
GridSearchCV

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
   
---------------------------------------------------------------------------------------------
 
GridSearchCV: Grid Search 1 results: 

Best: -0.219932 using {'activation': 'relu', 'layers': 2, 'units': 512}
-0.219932 (0.005488) with: {'activation': 'relu', 'layers': 2, 'units': 512}
-0.220033 (0.004069) with: {'activation': 'relu', 'layers': 2, 'units': 1024}
-0.225337 (0.010365) with: {'activation': 'relu', 'layers': 3, 'units': 512}
-0.241241 (0.027027) with: {'activation': 'relu', 'layers': 3, 'units': 1024}
-0.271150 (0.014250) with: {'activation': 'tanh', 'layers': 2, 'units': 512}
-0.267077 (0.015247) with: {'activation': 'tanh', 'layers': 2, 'units': 1024}
-0.274323 (0.009548) with: {'activation': 'tanh', 'layers': 3, 'units': 512}
-0.281756 (0.002757) with: {'activation': 'tanh', 'layers': 3, 'units': 1024}

GridSearchCV: Grid Search 2 results: 

Best: -0.2153350661198298 using {'activation': 'relu', 'layers': 2, 'units': 256}
-0.225708 (0.005610383694475058) with: {'activation': 'relu', 'layers': 2, 'units': 64}
-0.215335 (0.004047098886464389) with: {'activation': 'relu', 'layers': 2, 'units': 256}
-0.226354 (0.0055235757636850225) with: {'activation': 'relu', 'layers': 2, 'units': 512}
-0.223490 (0.0037692652185719083) with: {'activation': 'relu', 'layers': 3, 'units': 64}
-0.218547 (0.0032141666190944065) with: {'activation': 'relu', 'layers': 3, 'units': 256}
-0.225550 (0.004781145412296005) with: {'activation': 'relu', 'layers': 3, 'units': 512}
-0.225605 (0.004630681555513752) with: {'activation': 'relu', 'layers': 4, 'units': 64}
-0.218105 (0.006840586448255651) with: {'activation': 'relu', 'layers': 4, 'units': 256}
-0.220663 (0.007933670407982561) with: {'activation': 'relu', 'layers': 4, 'units': 512}
-0.220169 (0.005787989325121524) with: {'activation': 'relu', 'layers': 5, 'units': 64}
-0.229592 (0.006135902432048814) with: {'activation': 'relu', 'layers': 5, 'units': 256}
-0.230946 (0.010976929578112618) with: {'activation': 'relu', 'layers': 5, 'units': 512}
-0.234645 (0.003443314702353727) with: {'activation': 'tanh', 'layers': 2, 'units': 64}
-0.247188 (0.008642968678711664) with: {'activation': 'tanh', 'layers': 2, 'units': 256}
-0.252233 (0.004968353003027476) with: {'activation': 'tanh', 'layers': 2, 'units': 512}
-0.235887 (0.004143316236182298) with: {'activation': 'tanh', 'layers': 3, 'units': 64}
-0.249543 (0.00655520023442987) with: {'activation': 'tanh', 'layers': 3, 'units': 256}
-0.274348 (0.006677011607917516) with: {'activation': 'tanh', 'layers': 3, 'units': 512}
-0.236436 (0.008766370998344802) with: {'activation': 'tanh', 'layers': 4, 'units': 64}
-0.258512 (0.00943073712371849) with: {'activation': 'tanh', 'layers': 4, 'units': 256}
-0.272500 (0.015316740284002808) with: {'activation': 'tanh', 'layers': 4, 'units': 512}
-0.246157 (0.008237982697105703) with: {'activation': 'tanh', 'layers': 5, 'units': 64}
-0.280132 (0.016125180700005026) with: {'activation': 'tanh', 'layers': 5, 'units': 256}
-0.286209 (0.01854733789273122) with: {'activation': 'tanh', 'layers': 5, 'units': 512}

GridSearchCV: Grid Search 3 results: 

Best: -0.21311742067337036 using {'activation': 'relu', 'epochs': 50, 'layers': 2, 'optimizer': 'adam', 'units': 256}
-0.22185669839382172 (0.01106852511667019) with: {'activation': 'relu', 'epochs': 10, 'layers': 2, 'optimizer': 'adam', 'units': 256}
-0.21311742067337036 (0.010367324146148585) with: {'activation': 'relu', 'epochs': 50, 'layers': 2, 'optimizer': 'adam', 'units': 256}

GridSearchCV: Grid Search 4 results:

Best: -0.212175 using {'activation': 'relu', 'batch_size': 256, 'epochs': 25, 'layers': 2, 'optimizer': 'adam', 'units': 256}
-0.221462 (0.001908) with: {'activation': 'relu', 'batch_size': 1, 'epochs': 25, 'layers': 2, 'optimizer': 'adam', 'units': 256}
-0.222489 (0.000914) with: {'activation': 'relu', 'batch_size': 2, 'epochs': 25, 'layers': 2, 'optimizer': 'adam', 'units': 256}
-0.233052 (0.015113) with: {'activation': 'relu', 'batch_size': 4, 'epochs': 25, 'layers': 2, 'optimizer': 'adam', 'units': 256}
-0.214064 (0.007966) with: {'activation': 'relu', 'batch_size': 8, 'epochs': 25, 'layers': 2, 'optimizer': 'adam', 'units': 256}
-0.220868 (0.000747) with: {'activation': 'relu', 'batch_size': 16, 'epochs': 25, 'layers': 2, 'optimizer': 'adam', 'units': 256}
-0.219638 (0.000152) with: {'activation': 'relu', 'batch_size': 32, 'epochs': 25, 'layers': 2, 'optimizer': 'adam', 'units': 256}
-0.215322 (0.004254) with: {'activation': 'relu', 'batch_size': 64, 'epochs': 25, 'layers': 2, 'optimizer': 'adam', 'units': 256}
-0.224837 (0.008066) with: {'activation': 'relu', 'batch_size': 128, 'epochs': 25, 'layers': 2, 'optimizer': 'adam', 'units': 256}
-0.212175 (0.006040) with: {'activation': 'relu', 'batch_size': 256, 'epochs': 25, 'layers': 2, 'optimizer': 'adam', 'units': 256} 
'''
