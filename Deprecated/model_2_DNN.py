# Author: Lee Taylor, ST Number: 190211479
from   pre_processing   import *
from   evaluation       import *
import tensorflow       as tf

# Define the datasets
X_train, X_test, y_train, y_test = gen_data()
data = [X_train, X_test, y_train, y_test]
for d in data: print(d.shape)

# Define the model
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(512, input_shape=(8,), activation='relu'))
model.add(tf.keras.layers.Dense(512, activation='relu'))
model.add(tf.keras.layers.Dense(512, activation='relu'))
model.add(tf.keras.layers.Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Fit the model to the training data
model.fit(X_train, y_train, epochs=10, batch_size=16)

# Evaluate the model on the test set
test_loss, test_mae = model.evaluate(X_test, y_test, verbose=2)

score_model(model, X_test, y_test)
