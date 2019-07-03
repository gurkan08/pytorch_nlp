from torch import optim

class Optimizer(object):

	def get_optimizer(idx_opt, nn_model, lr):
		if idx_opt==1:
			return optim.Adam(nn_model.parameters(), lr=lr)
		if idx_opt==2:
			return optim.SGD(nn_model.parameters(), lr=lr, momentum=0.9)

