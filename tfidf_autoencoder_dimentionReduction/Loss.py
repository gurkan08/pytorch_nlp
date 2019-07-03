from torch.nn import MSELoss

class Loss(object):

	def get_loss(idx_loss):
		if idx_loss==1:
			return MSELoss()

