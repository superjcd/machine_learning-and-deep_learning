import pandas as pd
import datetime
import numpy as np

datadir='~/Desktop/my package/machine_learning-and-deep_learning/Data/houseprice/'
'''
X_train = pd.read_csv(datadir+'X.csv').drop('SalePrice',axis=1)
y_train = pd.read_csv(datadir+'X.csv')['SalePrice']
X_test = pd.read_csv(datadir+'test_X.csv')
Id = X_test['Id']

#print(X_train.shape,y_train.shape,X_test.shape)
#去掉ID项,standscale
from sklearn.preprocessing import RobustScaler
scalor = RobustScaler()
X = scalor.fit_transform(X_train.drop(['Id'],axis=1))
test_X = scalor.fit_transform(X_test.drop(['Id'],axis=1))
print(X.shape,test_X.shape)
'''
X_train = pd.read_csv(datadir+'X2.csv').drop('SalePrice',axis=1)
y_train = pd.read_csv(datadir+'X2.csv')['SalePrice']
X_test = pd.read_csv(datadir+'test_X2.csv')

from sklearn.preprocessing import RobustScaler
scalor = RobustScaler()
X = scalor.fit_transform(X_train)
test_X = scalor.fit_transform(X_test)
Id = list(range(1461,1461+1459))

print(X.shape,test_X.shape)

###deep learning###
from mxnet import ndarray as nd
from mxnet import autograd
from mxnet import gluon

#prepadata for train
X_train = nd.array(X)
y_train = nd.array(y_train)
y_train.reshape((X.shape[0], 1))

X_test = nd.array(test_X)

#define loss function
square_loss = gluon.loss.L2Loss()

def get_rmse_log(net, X_train, y_train):
    num_train = X_train.shape[0]
    clipped_preds = nd.clip(net(X_train), 1, float('inf'))
    return np.sqrt(2 * nd.sum(square_loss(
        nd.log(clipped_preds), nd.log(y_train))).asscalar() / num_train)#mean  square log error

##define model
def get_net():
    net = gluon.nn.Sequential()
    with net.name_scope():
        net.add(gluon.nn.Dense(1500,activation='relu'))
        net.add(gluon.nn.Dropout(0.3))
        net.add(gluon.nn.Dense(1))
    net.initialize()
    return net

####define train process
from mxnet import init
import matplotlib as mpl
mpl.rcParams['figure.dpi']= 120
import matplotlib.pyplot as plt

def train(net, X_train, y_train, X_test, y_test, epochs,
          verbose_epoch, learning_rate, weight_decay,batch_size = 100):
    train_loss = []
    if X_test is not None:
        test_loss = []

    dataset_train = gluon.data.ArrayDataset(X_train, y_train)
    data_iter_train = gluon.data.DataLoader(
        dataset_train, batch_size,shuffle=True)
    trainer = gluon.Trainer(net.collect_params(), 'adam',
                            {'learning_rate': learning_rate,
                             'wd': weight_decay})
    net.collect_params().initialize(init.Xavier(),force_reinit=True)#使用Xavier 进行初始化
    for epoch in range(epochs):
        for data, label in data_iter_train:
            with autograd.record():
                output = net(data)
                loss = square_loss(output, label)
            loss.backward()
            trainer.step(batch_size)

            cur_train_loss = get_rmse_log(net, X_train, y_train)
        if epoch > verbose_epoch:
            print("Epoch %d, train loss: %f" % (epoch, cur_train_loss))
        train_loss.append(cur_train_loss)
        if X_test is not None:
            cur_test_loss = get_rmse_log(net, X_test, y_test) #net 和上面相同
            test_loss.append(cur_test_loss)

    #plt.plot(train_loss)
    #plt.legend(['train'])
    if X_test is not None:
        pass
    #     plt.plot(test_loss)
    #     plt.legend(['train','test'])
    # plt.show()
    if X_test is not None:
        return cur_train_loss, cur_test_loss
    else:
        return cur_train_loss


#define cv
def k_fold_cross_valid(k, epochs, verbose_epoch, X_train, y_train,
                       learning_rate, weight_decay):
    assert k > 1
    fold_size = X_train.shape[0] // k
    train_loss_sum = 0.0
    test_loss_sum = 0.0
    for test_i in range(k):
        X_val_test = X_train[test_i * fold_size: (test_i + 1) * fold_size, :]
        y_val_test = y_train[test_i * fold_size: (test_i + 1) * fold_size]

        val_train_defined = False
        for i in range(k):
            if i != test_i:
                X_cur_fold = X_train[i * fold_size: (i + 1) * fold_size, :]
                y_cur_fold = y_train[i * fold_size: (i + 1) * fold_size]
                if not val_train_defined:
                    X_val_train = X_cur_fold
                    y_val_train = y_cur_fold
                    val_train_defined = True
                else:
                    X_val_train = nd.concat(X_val_train, X_cur_fold, dim=0)
                    y_val_train = nd.concat(y_val_train, y_cur_fold, dim=0)
        net = get_net()
        train_loss, test_loss = train(
            net, X_val_train, y_val_train, X_val_test, y_val_test,
            epochs, verbose_epoch, learning_rate, weight_decay)
        train_loss_sum += train_loss
        print("Test loss: %f" % test_loss)
        test_loss_sum += test_loss
    return train_loss_sum / k, test_loss_sum / k


k = 5
epochs = 100
verbose_epoch = 95
learning_rate = 0.06
weight_decay = 40

train_loss, test_loss = k_fold_cross_valid(k, epochs, verbose_epoch, X_train,
                                           y_train, learning_rate, weight_decay)
print("%d-fold validation: Avg train loss: %f, Avg test loss: %f" %
      (k, train_loss, test_loss))


def learn(epochs, verbose_epoch, X_train, y_train, learning_rate,
          weight_decay):
    net = get_net()
    train(net, X_train, y_train, None, None, epochs, verbose_epoch,
          learning_rate, weight_decay)
    preds = net(X_test).asnumpy()
    SalePrice = pd.Series(preds.reshape(1, -1)[0])
    submission = pd.DataFrame({'Id': Id,
                               'SalePrice': SalePrice})
    submission.to_csv('~/Desktop/my package/machine_learning-and-deep_learning/submissions/dp.csv', index=False)



learn(epochs,verbose_epoch,X_train,y_train,learning_rate,weight_decay)