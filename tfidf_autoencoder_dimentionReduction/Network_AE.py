import torch.nn as nn
import torch.nn.functional as F


class Network_AE(nn.Module):
	def __init__(self, vocab_size, drop_out):
		super(Network_AE, self).__init__()

		#down
		self.down_fc1 = nn.Linear(vocab_size, 512)
		self.down_do_fc1 = nn.Dropout(p=drop_out)

		self.down_fc2 = nn.Linear(512, 256)
		self.down_do_fc2 = nn.Dropout(p=drop_out)

		self.down_fc3 = nn.Linear(256, 128)
		self.down_do_fc3 = nn.Dropout(p=drop_out)

		#up
		self.up_fc1 = nn.Linear(128, 256)
		self.up_do_fc1 = nn.Dropout(p=drop_out)

		self.up_fc2 = nn.Linear(256, 512)
		self.up_do_fc2 = nn.Dropout(p=drop_out)

		self.up_fc3 = nn.Linear(512, vocab_size)
		self.up_do_fc3 = nn.Dropout(p=drop_out)




	def forward(self, x):

		##TODO: add batch norm !!!
		##TODO: input 0-1 normalization, tf-idf zaten 0-1 aralıkta galiba
		##dropout kullanınca loss daha yüksek değerden başlıyor

		x = self.down_do_fc1(F.relu(self.down_fc1(x)))
		x = self.down_do_fc2(F.relu(self.down_fc2(x)))
		x = self.down_do_fc3(F.relu(self.down_fc3(x)))

		x = self.up_do_fc1(F.relu(self.up_fc1(x)))
		x = self.up_do_fc2(F.relu(self.up_fc2(x)))
		x = self.up_do_fc3(F.sigmoid(self.up_fc3(x)))
		
		"""
		#without dropout layer	
		x = F.relu(self.down_fc1(x))
		x = F.relu(self.down_fc2(x))
		x = F.relu(self.down_fc3(x))

		x = F.relu(self.up_fc1(x))
		x = F.relu(self.up_fc2(x))
		x = F.sigmoid(self.up_fc3(x))
		"""
		return x


