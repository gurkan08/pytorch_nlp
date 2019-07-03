# -*- coding: utf-8 -*-
"""
Created on Wed May 15 11:45:46 2019

@author: gurkan.sahin
"""


"""
orijinal 1150 haber dataseti DL kodun alacağı formata dönüştüren kod !
"""


import os

dataset_dir = "D:/Users/gurkan.sahin/Desktop/NLP/1150haber/raw_texts"

new_dataset_dir = "D:/Users/gurkan.sahin/Desktop/1150haber"
train_dir = os.path.join(new_dataset_dir, "train")
valid_dir = os.path.join(new_dataset_dir, "valid")
if not os.path.exists(train_dir):
    os.makedirs(train_dir)
if not os.path.exists(valid_dir):
    os.makedirs(valid_dir)
    
"""
önemli: label indexlerini 0 dan başlat aksi halde hata alınıyor crossentropyloss()'da !
"""
label_dict = {"ekonomi":0,
              "magazin":1,
              "saglik":2,
              "siyasi":3,
              "spor":4}


train_corpus = open(os.path.join(train_dir, "corpus.txt"), "w")
train_label = open(os.path.join(train_dir, "label.txt"), "w")
valid_corpus = open(os.path.join(valid_dir, "corpus.txt"), "w")
valid_label = open(os.path.join(valid_dir, "label.txt"), "w")


for idx, (key, value) in enumerate(label_dict.items()):
    #print(key, value)
    directory = os.path.join(dataset_dir, key) 
    files = os.listdir(directory) 
    
    train_size = len(files)*0.8
    
    for idx, file in enumerate(files):
        new_file = "" #txt içindeki tüm satırların birleştirilmiş hali
        
        if idx < train_size:
            with open(os.path.join(directory, file), "r") as f:
                for _f in f:
                    if _f.strip() != "":
                        new_file += _f.strip()+" "
                        
            train_corpus.write(new_file.strip()+"\n")
            train_label.write(str(value)+"\n")
        
        if idx >= train_size:
            with open(os.path.join(directory, file), "r") as f:
                for _f in f:
                    if _f.strip() != "":
                        new_file += _f.strip()+" "
                        
            valid_corpus.write(new_file.strip()+"\n")
            valid_label.write(str(value)+"\n")
        
        
train_corpus.close()
train_label.close()
valid_corpus.close()
valid_label.close()
    
