import torch
from torch.utils.data import DataLoader 
import argparse
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt

from Loss import Loss 
from Optimizer import Optimizer
from Network_AE import Network_AE

torch.manual_seed(39)

class Main_AE(object):
	def __init__(self, args):
		self.INPUT_DIR = args.input_dir
		self.EPOCH = int(args.epoch)
		self.LEARNING_RATE = float(args.lr)
		self.BATCH_SIZE = int(args.batch)
		self.LOSS = int(args.loss)
		self.OPTIMIZER = int(args.optimizer)
		self.DROP_OUT = float(args.drop_out)

		self.FIRST_N_CHAR_STEMMING = int(args.first5stemming)
		self.MIN_NGRAM_SIZE = int(args.min_ngram)
		self.MAX_NGRAM_SIZE = int(args.max_ngram)
		self.MIN_DF = int(args.min_df)
		self.VOCAB = []
		self.USE_CUDA = bool(torch.cuda.is_available() and bool(args.use_cuda))




	def first5stemming(self, doc):
		doc = doc.split()
		for idx_word, word in enumerate(doc):
			if len(word) > self.FIRST_N_CHAR_STEMMING:
				doc[idx_word] = word[0:self.FIRST_N_CHAR_STEMMING]
		return doc




	def preprocess(self, doc):
		doc = self.first5stemming(doc.strip().strip("\n").lower())
		#[self.VOCAB.append(word) for word in doc if word not in self.VOCAB]

		#tf-idf vectorizer kullanabilmesi için list of list değil list şeklinde olsun 
		return_doc = ""
		for word in doc:
			return_doc += word+" "
		return return_doc.strip()



	def read_dataset(self):
		X = []
		r_cursor = open(os.path.join(self.INPUT_DIR, "X.csv"), "r") #encoding="utf8"
		X = [self.preprocess(doc) for doc in r_cursor if doc.strip() != ""]
		return X
		


	def data_loader(self, X_tfidf):
		loader = []
		[loader.append( (torch.from_numpy(np.array(doc_tfidf)).float(), torch.from_numpy(np.array(doc_tfidf)).float()) ) for doc_tfidf in X_tfidf]
		loader = DataLoader(dataset=loader, batch_size=self.BATCH_SIZE, shuffle=True)
		return loader



	def start(self):
		#read dataset
		X = self.read_dataset()
		print("# train sample:", len(X))

		#tf-idf vectorizer
		vectorizer = TfidfVectorizer(min_df=self.MIN_DF, ngram_range=(self.MIN_NGRAM_SIZE, self.MAX_NGRAM_SIZE))
		X_tfidf = vectorizer.fit_transform(X) #return document-feature tf-idf matrix
		self.VOCAB = vectorizer.get_feature_names()
		print("# VOCAB.:", len(self.VOCAB))
		X_tfidf = X_tfidf.toarray() 

		"""
		#0-1 ralığına normalize etmek gerekir mi !!!
		"""

		#dataLloader
		loader = self.data_loader(X_tfidf)

		#train
		self.train(loader)



	def plot(self, loss):
		fig, ax = plt.subplots()
		ax.plot(np.array(range(1,self.EPOCH+1)), np.array(loss), "g", label="train")
		ax.set(xlabel="epoch", ylabel="loss")
		ax.legend(loc="upper right")
		plt.figtext(0.2, 0.9, "last_loss:{}".format(loss[-1]))
		fig.savefig(os.path.join(self.INPUT_DIR, "loss.png"))
		plt.close(fig)



	def train(self, loader):
		net = Network_AE(vocab_size=len(self.VOCAB), drop_out=self.DROP_OUT)
		if self.USE_CUDA:
			net.cuda()
		
		optimizer = Optimizer.get_optimizer(self.OPTIMIZER, net, self.LEARNING_RATE)
		criterion = Loss.get_loss(self.LOSS)

		#train
		plot_train_loss = []

		net.train()
		for epoch in range(1, self.EPOCH + 1):
			train_loss = 0.0
			for batch_idx, (X, y) in enumerate(loader):
				y = torch.squeeze(y)
				if self.USE_CUDA:
					X = Variable(X.cuda())
					y = Variable(y.cuda())
				else:
					X = Variable(X)
					y = Variable(y)

				output = net(X)
				loss = criterion(output, y)
				train_loss += loss.data[0]

				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

			print("epoch:{}, train_loss:{}".format(epoch, train_loss))
			plot_train_loss.append(train_loss)

		#plot
		self.plot(plot_train_loss)




if __name__ == "__main__":

	parser = argparse.ArgumentParser(description="tf-idf dimention reduction autoencoder network (Gurkan Sahin, 26/10/2018)")
	parser.add_argument("-input_dir", help="input file", required=True)
	parser.add_argument("-epoch", help="epoch size", required=True)
	parser.add_argument("-lr", help="learning rate", required=True)
	parser.add_argument("-batch", help="batch size", required=True)
	parser.add_argument("-loss", help="loss function (1:MSE)", default=1)
	parser.add_argument("-optimizer", help="optimizer (1:ADAM, 2:SGD)", default=1)
	parser.add_argument("-use_cuda", help="use cuda (0:false, 1:true)", default=1)
	parser.add_argument("-drop_out", help="drop out rate (default:0.5)", default=0.5)
	parser.add_argument("-first5stemming", help="first5stemming", default=5)
	parser.add_argument("-min_ngram", help="min_ngram size for tfidfvectorizer (default:1)", default=1)
	parser.add_argument("-max_ngram", help="max_ngram size for tfidfvectorizer (default:1)", default=1)
	parser.add_argument("-min_df", help="min. document freq for tfidfvectorizer (default:1)", default=1)

	arguments = parser.parse_args()

	obj = Main_AE(arguments)
	obj.start()

	
	
	



