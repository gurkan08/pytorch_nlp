from torch.nn import BCELoss

class Loss(object):
    
    def get_loss(loss_idx):
        if loss_idx==1:
        	return BCELoss()
       

