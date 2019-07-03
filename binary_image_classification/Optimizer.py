from torch import optim

class Optimizer(object):
    
    def get_optimizer(opt_idx,net,lr):
        if opt_idx==1:
            return optim.Adam(net.parameters(),lr=lr)
        if opt_idx==2:
            return optim.SGD(net.parameters(),lr=lr,momentum=0.9)
