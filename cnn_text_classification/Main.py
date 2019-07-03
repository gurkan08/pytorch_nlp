# -*- coding: utf-8 -*-
"""
Created on Mon May 13 09:26:10 2019

@author: gurkan.sahin
"""


import torch
import os
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader
import argparse
import matplotlib.pyplot as plt
import time
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from torch.optim.lr_scheduler import ExponentialLR
import pandas as pd

from Optimizer import Optimizer
from Loss import Loss
from DatasetText import DatasetText
from Preprocess import Preprocess
from Network import Network
from Utility import Utility


class MainTextClass(object):
    
    def __init__(self, args):
        
        corpus_file_name = "corpus.txt"
        label_file_name = "label.txt"
        
        self.INPUT_DIR = args.input_dir
        self.TRAIN_DIR = os.path.join(self.INPUT_DIR,"train")
        self.VALID_DIR = os.path.join(self.INPUT_DIR,"valid")
        
        self.TRAIN_CORPUS_DIR = os.path.join(self.TRAIN_DIR, corpus_file_name)
        self.TRAIN_LABEL_DIR = os.path.join(self.TRAIN_DIR, label_file_name)
    
        self.VALID_CORPUS_DIR = os.path.join(self.VALID_DIR, corpus_file_name)
        self.VALID_LABEL_DIR = os.path.join(self.VALID_DIR, label_file_name)
        
        self.FIRST_K_CHAR_STEM = int(args.k_char_stem)
        self.MAX_SENTENCE_LEN = int(args.max_sentence_len)
        
        #
        self.MODEL_DIR = os.path.join(self.INPUT_DIR,"model")
        self.PLOT_DIR = os.path.join(self.INPUT_DIR,"plot")
        self.MISCLASS_DIR = os.path.join(self.INPUT_DIR,"misclass") 

        self.BATCH_SIZE = int(args.batch)
        self.DROPOUT = float(args.drop)
        self.LOSS = int(args.loss)
        self.OPTIMIZER = int(args.opt)
        self.LEARNING_RATE = float(args.lr)
        self.EPOCH = int(args.epoch)
        self.NETWORK = int(args.n)
        self.USE_CUDA = bool(args.cuda) and torch.cuda.is_available()
        self.SAVE_PERIOD = int(args.save)
        self.SAVE_MODEL = args.save_model

        self.VOCAB_SIZE = None
        self.WORD_EMBEDDING_SIZE = int(args.word_embedding_size)
        self.N_CLASS = int(args.n_class)
        
        EXP_FOLDER_NAME = str(self.NETWORK)+"_"+str(self.EPOCH)+"_"+str(self.BATCH_SIZE)+"_"+str(self.LEARNING_RATE)[2:]+"_"+str(self.DROPOUT)
        self.MODEL_DIR = os.path.join(self.MODEL_DIR,EXP_FOLDER_NAME)
        if not os.path.exists(self.MODEL_DIR):
            os.makedirs(self.MODEL_DIR)
        
        self.PLOT_DIR = os.path.join(self.PLOT_DIR,EXP_FOLDER_NAME)
        if not os.path.exists(self.PLOT_DIR):
            os.makedirs(self.PLOT_DIR)
            
            
        
    
    
    
    def get_network(self):
        if self.NETWORK==1:
            return Network(self.USE_CUDA, 
                           self.DROPOUT,
                           self.VOCAB_SIZE,
                           self.WORD_EMBEDDING_SIZE,
                           self.N_CLASS)
    
    
    
       
    
    def save_model(self, file_name):
        torch.save(self.NET.state_dict(), os.path.join(self.MODEL_DIR, file_name+".pth"))
    
    


    def save_plot(self, file_name, epoch, args):
        fig, ax = plt.subplots(1,2)
        ax[0].plot(np.array(range(1,epoch+1)),np.array(args[0]),"g",label="train")
        ax[0].plot(np.array(range(1,epoch+1)),np.array(args[1]),"r",label="valid")
        ax[0].set(xlabel="epoch",ylabel="loss")
        ax[0].legend(loc="upper right")
        
        ax[1].plot(np.array(range(1,epoch+1)),np.array(args[2]),"g",label="train")
        ax[1].plot(np.array(range(1,epoch+1)),np.array(args[3]),"r",label="valid")
        ax[1].set(xlabel="epoch",ylabel="acc")
        ax[1].legend(loc="lower right")
        
        fig.savefig(os.path.join(self.PLOT_DIR, file_name+".png"))
        plt.close(fig)
        
        
        
    """
    def save_misclassified(self, misclassified, filename, select="train"):
        save_dir = os.path.join(self.MISCLASS_DIR,select)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        dataframe = pd.DataFrame(misclassified, columns=["image","prob.","pred_label","gt_label"])
        pd_writer = pd.ExcelWriter(os.path.join(save_dir,filename) + ".xlsx", engine='xlsxwriter')
        dataframe.to_excel(pd_writer)
        pd_writer.save()
    """
        
    
        
    
    def __train(self, train_loader, valid_loader, optimizer, criterion, scheduler):
        plot_train_loss=[]
        plot_valid_loss=[]
        plot_train_acc=[]
        plot_valid_acc=[]
        for epoch in range(1,self.EPOCH+1):
            print("*"*20)
            print("epoch:{}".format(epoch))
        
            s_t = time.time()
            train_loss, cm, train_acc, f1 = self.train(train_loader, optimizer, criterion)
            t = Utility.elapsed_time(s_t, time.time())
            plot_train_loss.append(train_loss)
            plot_train_acc.append(train_acc)
            MainTextClass.information(train_loss, cm, train_acc, f1, t, mode="train")
            
            s_t = time.time()
            valid_loss, cm, valid_acc, f1 = self.valid(valid_loader, criterion)
            t = Utility.elapsed_time(s_t, time.time())
            plot_valid_loss.append(valid_loss)
            plot_valid_acc.append(valid_acc)
            MainTextClass.information(valid_loss, cm, valid_acc, f1, t, mode="valid")
            
            if not epoch % self.SAVE_PERIOD:
                file_name = str(self.NETWORK)+"_"+str(epoch)+"_"+str(self.BATCH_SIZE)+"_"+str(self.LEARNING_RATE)[2:]+"_"+str(self.DROPOUT)
                if self.SAVE_MODEL:
                    self.save_model(file_name)
                args = [plot_train_loss, 
                        plot_valid_loss,
                        plot_train_acc,
                        plot_valid_acc
                        ]
                self.save_plot(file_name, epoch, args)
                
            scheduler.step()
            #self.adjust_lr(optimizer,self.LEARNING_RATE,epoch)
    
    
    
    
    
    def information(loss, cm, acc, f1, t, mode="train"):
        print("mode:{}, time:{}".format(mode,t))
        print("cm: ", cm)
        print("loss: ", loss)
        print("acc: ", acc)
        print("f1_weighted: ", f1)
        
        
        
    
    
    def get_confusion_matrix(gt_labels, pred_labels):
        cm = confusion_matrix(gt_labels, pred_labels)
        acc = accuracy_score(gt_labels, pred_labels)
        f1 = f1_score(gt_labels, pred_labels, average='weighted')
        return cm, acc, f1
        
    
    
    
    def train(self, train_loader, optimizer, criterion):
        self.NET.train()
        train_loss=[]
        all_pred_labels=[]
        all_gt_labels=[]
        for _, (doc, label) in enumerate(train_loader):
            if self.USE_CUDA:
                doc = Variable(doc.cuda())
                label = Variable(label.cuda())
            else:
                doc = Variable(doc)
                label = Variable(label)
                
            out = self.NET.forward(doc.long())
            loss = criterion(out, label)
            train_loss.append(loss.data[0])
            _, pred_label = torch.max(out.data, 1)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            label = label.data.cpu().numpy().flatten()
            pred_label = pred_label.data.cpu().numpy().flatten()
            all_gt_labels.append(label)
            all_pred_labels.append(pred_label)
            
        cm, acc, f1 = MainTextClass.get_confusion_matrix(np.concatenate(all_gt_labels).ravel(), 
                                                         np.concatenate(all_pred_labels).ravel()
                                                         )
        return np.mean(train_loss), cm, acc, f1
        
    
    
    

    def valid(self, valid_loader, criterion):
        self.NET.eval()
        valid_loss=[]
        all_pred_labels=[]
        all_gt_labels=[]
        for _, (doc, label) in enumerate(valid_loader):
            if self.USE_CUDA:
                doc = Variable(doc.cuda())
                label = Variable(label.cuda())
            else:
                doc = Variable(doc)
                label = Variable(label)
            
            out = self.NET.forward(doc.long())
            loss = criterion(out, label)
            valid_loss.append(loss.data[0])
            _, pred_label = torch.max(out.data, 1)
            
            label = label.data.cpu().numpy().flatten()
            pred_label = pred_label.data.cpu().numpy().flatten()
            all_gt_labels.append(label)
            all_pred_labels.append(pred_label)
            
        cm, acc, f1 = MainTextClass.get_confusion_matrix(np.concatenate(all_gt_labels).ravel(), 
                                                         np.concatenate(all_pred_labels).ravel()
                                                         )
        return np.mean(valid_loss), cm, acc, f1




    
    def read_dataset(self):
        #train
        train_pre = Preprocess()
        train_pre.read_corpus(corpus_dir=self.TRAIN_CORPUS_DIR, 
                              label_dir=self.TRAIN_LABEL_DIR, 
                              first_k_char_stem=self.FIRST_K_CHAR_STEM)
        
        train_data = DatasetText(corpus=train_pre.get_corpus(), 
                                 label=train_pre.get_label(),
                                 word2idx=Preprocess.get_word2idx(), 
                                 max_sentence_len=self.MAX_SENTENCE_LEN)
        
        train_loader = DataLoader(dataset=train_data, 
                                  batch_size=self.BATCH_SIZE, 
                                  shuffle=True)
        
        #valid
        valid_pre = Preprocess()
        valid_pre.read_corpus(corpus_dir=self.VALID_CORPUS_DIR, 
                              label_dir=self.VALID_LABEL_DIR, 
                              first_k_char_stem=self.FIRST_K_CHAR_STEM)
        
        valid_data = DatasetText(corpus=valid_pre.get_corpus(), 
                                 label=valid_pre.get_label(),
                                 word2idx=Preprocess.get_word2idx(), 
                                 max_sentence_len=self.MAX_SENTENCE_LEN)
        
        valid_loader = DataLoader(dataset=valid_data, 
                                  batch_size=self.BATCH_SIZE, 
                                  shuffle=True)
        
        self.VOCAB_SIZE = len(Preprocess.get_vocab())
        MainTextClass.save_dict(Preprocess.get_word2idx())
        
        return train_loader, valid_loader
        
    


    def save_dict(dict2idx):
        import json
        with open("D:/Users/gurkan.sahin/Desktop/NLP/cnn_text_class/word2idx.txt", 'w') as file:
            file.write(json.dumps(dict2idx))
        

        


    def start(self):
        train_loader, valid_loader = self.read_dataset()
        
        #create network
        self.NET = self.get_network()
        print("#net params:", sum(p.numel() for p in self.NET.parameters()))
        if self.USE_CUDA:
            self.NET.cuda()
            
        optimizer = Optimizer.get_optimizer(self.OPTIMIZER, self.NET, self.LEARNING_RATE)
        criterion = Loss.get_loss(self.LOSS)
        scheduler = ExponentialLR(optimizer, gamma=0.95)
        self.__train(train_loader, valid_loader, optimizer, criterion, scheduler)
        
        
        

    """
    def adjust_lr(self,optimizer,lr,epoch):
        for params in optimizer.param_groups:
            params["lr"]=float(params["lr"])*0.1
            self.LEARNING_RATE=float(params["lr"])
        print("new_lr->",self.LEARNING_RATE)
    """
    
    
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Text Class. DL code, (Gurkan Sahin, 13/05/2019)")
    parser.add_argument('-input_dir', help='Input corpus directory', required=True)
    parser.add_argument('-epoch', help='Epoch size', required=True)
    parser.add_argument('-batch', help='Batch size', required=True)
    parser.add_argument('-lr', help='Learning rate (1e-4,1e-5)', required=True)
    parser.add_argument('-drop', help='Dropout rate', required=True)
    parser.add_argument('-k_char_stem', help='fisrt k char stemmimg', required=True)
    parser.add_argument('-max_sentence_len', help='max. sentence len', required=True)
    parser.add_argument('-word_embedding_size', help='word embedding size', required=True)
    parser.add_argument('-n_class', help='number of class', required=True) #bunu otmatik yap !

    parser.add_argument('-opt', help='Optimizer (1:adam, 2:SGD default=1)', default=1)
    parser.add_argument('-loss', help='Loss function (1:CELoss, default=1)', default=1)
    parser.add_argument('-n', help='Network (default=1)', default=1)
    parser.add_argument('-save', help='save model frequency (default:1 epoch)', default=1)
    parser.add_argument('-save_model', help='save model flag', default=True)
    parser.add_argument('-cuda', help='use cuda (True/False)', default=True)
    
    args = parser.parse_args()
    print("all args:", args)

    obj = MainTextClass(args)
    obj.start()







