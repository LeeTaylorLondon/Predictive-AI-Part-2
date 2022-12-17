# Author: Lee Taylor, ST Number: 190211479
from pre_processing             import *
from sklearn                    import svm
from sklearn.datasets           import make_classification
from sklearn.model_selection    import train_test_split


X_train, X_test, y_train, y_test = gen_data()
clf = svm.SVC(kernel='precomputed')
# linear kernel computation
gram_train = np.dot(X_train, X_train.T)
clf.fit(gram_train, y_train)
SVC(kernel='precomputed')
# predict on training examples
gram_test = np.dot(X_test, X_train.T)
clf.predict(gram_test)