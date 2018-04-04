import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
import numpy as np

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




#Base line croos validation error
lr = LinearRegression()
lr_val_score = cross_val_score(lr,X,y_train,
                               scoring='neg_mean_absolute_error',
                               cv=10,n_jobs=-1)
print('Baseline log CV score:',np.log(np.mean(-lr_val_score)))
