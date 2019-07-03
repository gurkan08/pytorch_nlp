# -*- coding: utf-8 -*-
"""
Created on Tue May 14 14:16:29 2019

@author: gurkan.sahin
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Network(nn.Module):

    def __init__(self, 
                 use_cuda, 
                 drop_out,
                 vocab_size,
                 word_embedding_size,
                 n_class):
        
        super(Network, self).__init__()
        
        self.USE_CUDA = use_cuda
        self.DROP_OUT_RATE = drop_out
        
        #+1 for zero-padding word, padding_idx=0 diye parametre var ama zaten bu işi yapıyorum gibi !
        self.EMBEDDING = nn.Embedding(vocab_size + 1, word_embedding_size)
        self.EMBEDDING.weight.requires_grad = True #false yapınca hata veriyor (default:True zaten)
        
        self.CNN_1 = nn.Conv2d(1, 8, kernel_size=(3, 3))
        self.BN_1 = nn.BatchNorm2d(8)
        self.CNN_2 = nn.Conv2d(8, 8, 3)
        self.BN_2 = nn.BatchNorm2d(8)
        self.CNN_3 = nn.Conv2d(8, 8, 3)
        self.BN_3 = nn.BatchNorm2d(8)
        #
        self.DROP_OUT = nn.Dropout2d(p=self.DROP_OUT_RATE)
        
        self.FC_1 = nn.Linear(320, 100) #mecvut command line (-word_embedding_size, -max_sentence_len) göre 800 değeri değişiyor, güncelle ! 
        self.FC_2 = nn.Linear(100, n_class)
        
        


    def forward(self, doc): 
        #doc: longtensor tipinde olmalı
        out = self.EMBEDDING(doc) #(batch, max_sentence_len, word_embedding_size) float tensor
        out = out.view(out.size()[0], 1, out.size()[1], out.size()[2]) #(batch, 1, max_sentence_len, word_embedding_size) cnn input format
        
        out = F.relu(self.CNN_1(out))
        out = F.max_pool2d(out, 2)
        out = self.BN_1(out)
        
        out = F.relu(self.CNN_2(out))
        out = F.max_pool2d(out, 2)
        out = self.BN_2(out)
        
        out = F.relu(self.CNN_3(out))
        out = F.max_pool2d(out, 2)
        out = self.BN_3(out) #(5,8,2,35)
        
        out = out.view(out.size()[0], -1) #flatten
        
        #print(out, out.size())
        #exit()
                      
        if self.DROP_OUT_RATE:
            out = self.DROP_OUT(out)
                
        #print(out, out.size())
        #break 
            
        out = F.relu(self.FC_1(out))
        
        
        if self.DROP_OUT_RATE:
            out = self.DROP_OUT(out)
        
        out = F.softmax(self.FC_2(out), dim=1)
        return out
        
    
    
    
    
    
