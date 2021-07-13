import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn import linear_model, metrics
import timeit
import os
from yellowbrick.regressor import PredictionError
from dataProcessing import dataPreparation, findCorr

models_path = 'models'
dataSet_path = 'datasets'
data = pd.read_csv(os.path.join(dataSet_path, 'AppleStore.csv'))
processedData = dataPreparation(data, bonus=True, reduce_features=False)
findCorr(processedData)

X = processedData.iloc[:, :-1]
Y = processedData.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=1, test_size=0.2, shuffle=True)

model1 = linear_model.LinearRegression()
m1start = timeit.default_timer()
model1.fit(X_train, y_train)
m1end = timeit.default_timer()
prediction1 = model1.predict(X_test)
print('MAE Model #1:', metrics.mean_absolute_error(np.asarray(y_test), prediction1))
print('R2 Model #1:', metrics.r2_score(np.asarray(y_test), prediction1))
print('Time to train Model #1: ' + str(m1end - m1start))
visualizer = PredictionError(model1, alpha=0.15)
visualizer.fit(X_train, y_train)
visualizer.score(X_test, y_test)
visualizer.show()

model2 = SVR(kernel='rbf')
m2start = timeit.default_timer()
model2.fit(X_train, y_train)
m2end = timeit.default_timer()
prediction2 = model2.predict(X_test)
print('\n')
print('MAE Model #2:', metrics.mean_absolute_error(np.asarray(y_test), prediction2))
print('R2 Model #2:', metrics.r2_score(np.asarray(y_test), prediction2))
print('Time to train Model #2: ' + str(m2end - m2start))
visualizer = PredictionError(model2, alpha=0.15)
visualizer.fit(X_train, y_train)
visualizer.score(X_test, y_test)
visualizer.show()
