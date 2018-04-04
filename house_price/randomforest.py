import pandas as pd
from datetime import datetime
import numpy as np
from sklearn.preprocessing import RobustScaler

datadir='~/Desktop/my package/machine_learning-and-deep_learning/Data/houseprice/'
X_train = pd.read_csv(datadir+'X2.csv').drop('SalePrice',axis=1)
y_train = pd.read_csv(datadir+'X2.csv')['SalePrice']
X_test = pd.read_csv(datadir+'test_X2.csv')

scalor = RobustScaler()
X = scalor.fit_transform(X_train)
test_X = scalor.fit_transform(X_test)
Id = list(range(1461,1461+1459))

print(X.shape,test_X.shape)



#Random Forest
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
rf = RandomForestRegressor()


#random search CV
params = {'n_estimators':randint(low=50, high=200),
          'max_features':randint(low=10, high=100),
          'min_samples_split':[2,3,4,5,6,7]}


rf_RS = RandomizedSearchCV(rf, param_distributions=params,
                                n_iter=100, cv=5,
                                scoring='neg_mean_absolute_error',
                                random_state=42)#比grid search多了n_iter参数

####recording run time######
beg = datetime.now()
rf_RS.fit(X,y_train)
end = datetime.now()
print('rs run time:',end - beg)

#check the results
cvres = rf_RS.cv_results_
params = cvres["params"]
score =  cvres["mean_test_score"]

info = pd.DataFrame(list(params))
info['Score'] = np.log(-score)

print(info.sort_values(['Score']))

estimator = rf_RS.best_estimator_
predictions = estimator.predict(test_X)

SalePrice = predictions
data = pd.DataFrame({'Id':Id,
                     'SalePrice':SalePrice})
data.to_csv('~/Desktop/my package/machine_learning-and-deep_learning/submissions/rf_submission.csv',
            index=False)