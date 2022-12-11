# Author: Lee Taylor, ST Number: 190211479
from    __future__ import print_function
from pre_processing import *
import  tpot
from    tpot import TPOTRegressor


X_train, X_test, y_train, y_test = gen_data()
tpot  = TPOTRegressor(generations=5, population_size=50, verbosity=2)
tpot.fit(X_train, y_train)
score = tpot.score(X_test, y_test)
print_function(score)
