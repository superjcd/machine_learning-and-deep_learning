from mxnet import ndarray as nd

def cross_entropy(yhat, y):
	return - nd.pick(nd.log(yhat), y)


def SGD(params, lr):
	for param in params:
		param[:] = param - lr*param.grad

