# Author: Lee Taylor, ST Number: 190211479
from pre_processing import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


def build_model(input_shape, verbose=False):
    model = Sequential()
    model.add(Dense(512, activation='relu', input_dim=input_shape))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(384, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(1))
    if verbose: model.summary()
    model.compile(loss="mse", metrics=['mse', 'accuracy']) # optimizer='adam'
    return model

def test_model(model, *data):
    if len(data) != 4: raise TypeError('Incorrect amount of data passed to test_model(...).')
    x_train, x_test, y_train, y_test = data
    train_pred = model.predict(x_train.values)
    train_rmse = np.sqrt(mean_squared_error(y_train.values, train_pred))
    test_pred = model.predict(x_test)
    test_rmse = np.sqrt(mean_squared_error(y_test.values, test_pred))
    print("Train RMSE: {:0.2f}".format(train_rmse))
    print("Test RMSE: {:0.2f}".format(test_rmse))


if __name__ == '__main__':
    X_train, X_test, y_train_true, y_test_true = gen_data()
    m3 = build_model(8, True)
    m3.fit(X_train.values, y_train_true.values, epochs=10, batch_size=4)
    test_model(m3, X_train, X_test, y_train_true, y_test_true)



