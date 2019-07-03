# -*- coding: utf-8 -*-
"""
Created on Mon May 13 11:54:36 2019

@author: gurkan.sahin
"""


import numpy as np


class DatasetText(object):
    
    def __init__(self, corpus, label, word2idx, max_sentence_len):
        self.CORPUS = corpus #list of list of words
        self.LABEL = label
        self.WORD2IDX = word2idx #(train + valid) vocab words
        self.MAX_SENTENCE_LEN = max_sentence_len
    
    
    
    def __getitem__(self, index):
        """
        convert word to unique wordidx,
        pad/trunc,
        and return this np sentence vector
        """
        word2idx_sentence = []
        for word in self.CORPUS[index]:
            word2idx_sentence.append(self.WORD2IDX[word])
        
        #pad/trunc
        if len(word2idx_sentence) > self.MAX_SENTENCE_LEN:
            word2idx_sentence = word2idx_sentence[:self.MAX_SENTENCE_LEN]
        
        if len(word2idx_sentence) < self.MAX_SENTENCE_LEN:
            for idx in range(self.MAX_SENTENCE_LEN - len(word2idx_sentence)):
                word2idx_sentence.append(0) #zero index word pad
        
        return (np.array(word2idx_sentence), int(self.LABEL[index])) 
    
        


    def __len__(self):
        """
        bu metodu __getitem__() da kullanacağın array'e göre 
        modifiye etmeyi sakın unutma !
        """
        return len(self.CORPUS)
    
    
    




        

