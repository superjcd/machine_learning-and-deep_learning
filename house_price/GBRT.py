import pandas as pd
import datetime



'''
X_train = pd.read_csv(datadir+'X.csv').drop('SalePrice',axis=1)
y_train = pd.read_csv(datadir+'X.csv')['SalePrice']
X_test = pd.read_csv(datadir+'test_X.csv')

#print(X_train.shape,y_train.shape,X_test.shape)
#去掉ID项,
from sklearn.preprocessing import RobustScaler
scalor = RobustScaler()
X = scalor.fit_transform(X_train.drop(['Id'],axis=1))
test_X = scalor.fit_transform(X_test.drop(['Id'],axis=1))
Id = X_test['Id']
'''
import pandas as pd
import datetime
from sklearn.preprocessing import RobustScaler
datadir='~/Desktop/my package/machine_learning-and-deep_learning/Data/houseprice/'

X_train = pd.read_csv(datadir+'X2.csv').drop('SalePrice',axis=1)
y_train = pd.read_csv(datadir+'X2.csv')['SalePrice']
X_test = pd.read_csv(datadir+'test_X2.csv')

from sklearn.preprocessing import RobustScaler
scalor = RobustScaler()
X = scalor.fit_transform(X_train)
test_X = scalor.fit_transform(X_test)
Id = list(range(1461,1461+1459))

print(X.shape,test_X.shape)


#GBRT
from sklearn.ensemble import GradientBoostingRegressor
from scipy.stats import randint
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
from datetime import datetime
gbrt = GradientBoostingRegressor(random_state=42)


param_distribs = {
        'learning_rate':np.linspace(0.01,1,50),
        'n_estimators':randint(low=100, high=500),
        'max_features':randint(low=1, high=50)
    }

rsc_gbtr=RandomizedSearchCV(gbrt,param_distributions=param_distribs,
                                n_iter=100, cv=5, scoring='neg_mean_absolute_error', random_state=42)

#recording time
beg = datetime.now()
rsc_gbtr.fit(X,y_train)
end = datetime.now()
print('Run time:',end - beg)


rscv = rsc_gbtr.cv_results_
prams_data = pd.DataFrame(list(rscv['params']))
prams_data['Error']= np.log(-rscv['mean_test_score'])
print(prams_data.sort_values('Error'))


estimator = rsc_gbtr.best_estimator_
predictions = estimator.predict(test_X)

SalePrice = predictions
data = pd.DataFrame({'Id':Id,
                     'SalePrice':SalePrice})
data.to_csv('~/Desktop/my package/machine_learning-and-deep_learning/submissions/gbrt_submission.csv',
            index=False)


#基于random search 结果进行Gridsearch
'''
from sklearn.model_selection import GridSearchCV

params = {'learning_rate':np.linspace(0.03,0.051,5),
          'n_estimators':np.linspace(390,410,5),
          'max_features':[22,27,32,27,42]
    }

gbrt_gs = GridSearchCV(gbrt,param_grid=params,cv=5,
                                scoring='neg_mean_absolute_error')

####recording run time#####
beg = datetime.now()
gbrt_gs.fit(X,y_train)
end = datetime.now()
print('gs run time',end-beg)

gsres = gbrt_gs.cv_results_
best_score = np.min(np.log(-gsres['mean_test_score']))
print('updated best score,',best_score)

estimator = gbrt_gs.best_estimator_
predictions = estimator.predict(test_X)

SalePrice = predictions
data = pd.DataFrame({'Id':Id,
                     'SalePrice':SalePrice})
data.to_csv('~/Desktop/my package/machine_learning-and-deep_learning/submissions/rf_submission.csv',
            index=False)
'''


