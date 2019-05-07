# coding:utf-8
import torch
import torch.nn as nn
import torch.nn.functional as f
class BiLSTM_ATT(nn.Module):
	def __init__(self,config):
		super(BiLSTM_ATT,self).__init__()
		self.hidden_size = config.hidden_size
		self.embedding_size =config.embedding_size
		self.embedding_dim = config.embedding_dim

		self.batch_size = config.batch_size
		self.classes_nums = config.classes_nums

		self.pos_size = config.pos_size
		self.pos_dim = config.pos_dim
		self.pre_train = config.pre_train
		self.word_embeds = nn.Embedding.from_pretrained(torch.FloatTensor(config.embedding_pre),freeze=False)

		# if self.pre_train:
		# 	self.word_embeds = nn.Embedding.from_pretrained(torch.FloatTensor(config.embedding_pre),freeze=False)
		# else:
		# 	self.word_embeds = nn.Embedding(self.embedding_size,self.embedding_dim)

		self.pos1_embeds = nn.Embedding(self.pos_size,self.pos_dim)
		self.pos2_embeds = nn.Embedding(self.pos_size,self.pos_dim)

		self.relation_embeds = nn.Embedding(self.classes_nums,self.hidden_size)

		self.lstm = nn.LSTM(input_size = self.embedding_dim+2*self.pos_dim,hidden_size=self.hidden_size//2,num_layers=1,bidirectional=True)
		self.hidden2rels = nn.Linear(self.hidden_size,self.classes_nums)

		self.dropout =nn.Dropout(config.dropout)

		#self.hidden = torch.randn(2, self.batch, self.hidden_size // 2)
		self.hidden = torch.randn(2, self.batch_size, self.hidden_size // 2)
		#self.hidden = (torch.randn(2,self.batch_size,self.hidden_size//2),
			#torch.randn(2,self.batch_size,self.hidden_size//2))
		self.att_weight = nn.Parameter(torch.randn(self.batch_size,1,self.hidden_size))
		self.relation_bias = nn.Parameter(torch.randn(self.batch_size,self.classes_nums,1))


	def attention(self,H):
		M = f.tanh(H)

		a = f.softmax(torch.bmm(self.att_weight,M),2)

		a = torch.transpose(a,1,2)

		return torch.bmm(H,a)

	def forward(self,sen,pos1,pos2):
		#[]
		self.hidden = (torch.randn(2,self.batch_size,self.hidden_size//2),
			torch.randn(2,self.batch_size,self.hidden_size//2))
		embeds = torch.cat((self.word_embeds(sen),self.pos1_embeds(pos1),self.pos2_embeds(pos2)),2)
		embeds = torch.transpose(embeds,0,1)
		lstm_out ,self.hidden = self.lstm(embeds,self.hidden)
		lstm_out = torch.transpose(lstm_out,0,1)
		lstm_out = torch.transpose(lstm_out,1,2)
		lstm_out = self.dropout(lstm_out)
		att_out = f.tanh(self.attention(lstm_out))

		relation = torch.tensor([i for i in range(self.classes_nums)],dtype = torch.long).repeat(self.batch_size,1)
		relation = self.relation_embeds(relation)

		res = torch.add(torch.bmm(relation,att_out),self.relation_bias)
		res = f.softmax(res,1)
		return res.view(self.batch_size,-1)




