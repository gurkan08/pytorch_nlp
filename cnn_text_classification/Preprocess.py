# -*- coding: utf-8 -*-
"""
Created on Mon May 13 09:28:15 2019

@author: gurkan.sahin
""" 

from snowballstemmer import TurkishStemmer

class Preprocess(object):
    
    vocab = []
    word2idx = {}
    idx2word = {}
    turkish_stemmer = TurkishStemmer()
    
    
    def __init__(self):
        self.corpus = []
        self.label = []
        
        
        
    def read_corpus(self, corpus_dir, label_dir, first_k_char_stem=0):
        with open(corpus_dir, "r") as sentences:
            for __sentence in sentences:
                #stemmed_line = Preprocess.stemming(__sentence, first_k_char_stem) #first_k_char stemming
                stemmed_line = Preprocess.snowball_stemmer(__sentence)
                self.corpus.append(stemmed_line)
                [self.add_vocab(word) for word in stemmed_line]
        
        with open(label_dir, "r") as labels:
            for __label in labels:
                self.label.append(int(__label.strip()))
        
        


    def snowball_stemmer(sentence):
        words = sentence.split()
        for idx in range(len(words)):
            words[idx] = Preprocess.turkish_stemmer.stemWord(words[idx])
        return words
    



    def stemming(sentence, first_k_char_stem):
        words = sentence.split()
        if first_k_char_stem != 0:
            for idx in range(len(words)):
                words[idx] = words[idx][:first_k_char_stem]
                
        return words

    
    
    
    def add_vocab(self, word):
        if word not in Preprocess.vocab:
            Preprocess.vocab.append(word)
            """
            0 index for padding word
            """
            Preprocess.word2idx[word] = len(Preprocess.vocab)
            Preprocess.idx2word[len(Preprocess.vocab)] = word
    
    
    
    def get_vocab():
        return Preprocess.vocab
    
    def get_corpus(self):
        return self.corpus
    
    def get_label(self):
        return self.label
    
    def get_word2idx():
        return Preprocess.word2idx
    
    def get_idx2word():
        return Preprocess.idx2word
    
    
    
    
        
    
    
    
    
    