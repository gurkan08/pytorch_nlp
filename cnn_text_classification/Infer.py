# -*- coding: utf-8 -*-
"""
Created on Tue May 21 13:57:26 2019

@author: gurkan.sahin
"""

from snowballstemmer import TurkishStemmer


class Infer(object):
    
    turkish_stemmer = TurkishStemmer()
    word2idx = Infer.load_wordIdx_txt("D:/Users/gurkan.sahin/Desktop/NLP/cnn_text_class/word2idx.txt")
    
    
    def __init__(self):
        pass
    
    
    def get_pred_class(doc):
        words = Infer.stemming(doc)
        print(words)
        
        """
        numeric_doc = [Infer.word2idx[word] for word in words]
        print(numeric_doc, len(numeric_doc))
        """
        
        
    def load_wordIdx_txt(dict_dir):
        import json
        with open(dict_dir, "r") as json_file:  
            return json.load(json_file)
        
        
        
        
    def stemming(doc):
        words = doc.split()
        for idx in range(len(words)):
            words[idx] = Infer.turkish_stemmer.stemWord(words[idx])
        return words
        
            
            
        
        

#test
Infer.get_pred_class("merhaba benim adım gürkan şahin")






