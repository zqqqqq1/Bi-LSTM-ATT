# coding: utf-8
import pickle
import numpy as np
import codecs
import torch
import torch.nn as nn
import torch.functional as f
import torch.optim as optim
import torch.utils.data as D
from torch.autograd import Variable
from model.bi_lstm_att import BiLSTM_ATT
class Config(object):
	def __init__(self):
		self.hidden_size = 200
		self.embedding_dim = 100
		self.max_len = 50
		self.pos_size = 82
		self.pos_dim = 25
		self.pre_train = True
		self.embedding_pre = []
		self.batch_size = 128
		self.dropout = 0.5
		self.file_type = 'eng'
		self.train_file = './data/people_relation_train.pkl'
		self.test_file = './data/people_relation_test.pkl'

		# self.train_file = './data/engdata_train.pkl'
		# self.test_file = './data/engdata_test.pkl'

		self.vec_file = './data/vec.txt'
		self.load_train_data()
		self.load_test_data()
		self.load_vec_pretrain()
		self.embedding_size = len(self.word2id)+1
		self.classes_nums = len(self.relation2id)
		self.max_epoches = 15
		self.learning_rate = 0.01
		self.test_epoch = 1
		self.save_epoch = 20
		#self.init_model()
		


	def load_train_data(self):
		with open(self.train_file,'rb')as inp:
			print('loading_train_data...')
			# self.word2id,self.id2word,self.relation2id,self.train,self.labels,self.position1,self.position2=pickle.load(inp,encoding='bytes')

			self.word2id = pickle.load(inp)
			self.id2word = pickle.load(inp)
			self.relation2id = pickle.load(inp)
			self.train = pickle.load(inp)
			self.labels = pickle.load(inp)
			self.position1 = pickle.load(inp)
			self.position2 = pickle.load(inp)
		self.train = torch.LongTensor(self.train[:len(self.train)-len(self.train)%self.batch_size])
		self.position1 = torch.LongTensor(self.position1[:len(self.train)-len(self.train)%self.batch_size])
		self.position2 = torch.LongTensor(self.position2[:len(self.train)-len(self.train)%self.batch_size])
		self.labels = torch.LongTensor(self.labels[:len(self.train)-len(self.train)%self.batch_size])
		self.train_datasets = D.TensorDataset(self.train,self.position1,self.position2,self.labels)
		self.train_dataloader = D.DataLoader(self.train_datasets,self.batch_size,True,num_workers=1)
		print ("train len", len(self.train)) 
		print ("word2id len",len(self.word2id))
	def load_test_data(self):
		with open(self.test_file, 'rb') as inp:
			print('loading_test_data...')
			self.test = pickle.load(inp)
			self.labels_t = pickle.load(inp)
			self.position1_t = pickle.load(inp)
			self.position2_t = pickle.load(inp)
		self.test = torch.LongTensor(self.test[:len(self.test)-len(self.test)%self.batch_size])
		self.position1_t = torch.LongTensor(self.position1_t[:len(self.test)-len(self.test)%self.batch_size])
		self.position2_t = torch.LongTensor(self.position2_t[:len(self.test)-len(self.test)%self.batch_size])
		self.labels_t = torch.LongTensor(self.labels_t[:len(self.test)-len(self.test)%self.batch_size])
		self.test_datasets = D.TensorDataset(self.test,self.position1_t,self.position2_t,self.labels_t)
		self.test_dataloader = D.DataLoader(self.test_datasets,self.batch_size,True,num_workers=1)
		print ("test len", len(self.test))
	def load_vec_pretrain(self):
		print('use pre_train')
		self.word2vec = {}
		with codecs.open(self.vec_file,'r',encoding='utf-8')as fr:
			for line in fr:
				self.word2vec[line.split()[0]]=map(eval,line.split()[1:])
			unknow_pre = []
			unknow_pre.extend([1]*self.embedding_dim)
			self.embedding_pre.append(unknow_pre)

			for word in self.word2id:
				if word in self.word2vec:
				#if self.word2vec.has_key(word):
					self.embedding_pre.append(word2vec[word])
				else:
					self.embedding_pre.append(unknow_pre)

			self.embedding_pre = np.asarray(self.embedding_pre)
			print('embedding_shape :',self.embedding_pre.shape)

	# def set_model(self):
	# 	return BiLSTM_ATT(self)
	def init_model(self):
		self.model = BiLSTM_ATT(config = self)
		self.optimizer = self.set_optimizer()
		self.criterion = self.set_loss()

	def set_optimizer(self):
		return optim.Adam(self.model.parameters(),lr=self.learning_rate,weight_decay=1e-5)

	def set_loss(self):
		return nn.CrossEntropyLoss(size_average=True)

	def train_one_step(self,sen,pos1,pos2,label):
		acc = 0
		total = 0
		sen = Variable(sen)
		pos1 = Variable(pos1)
		pos2 = Variable(pos2)
		y = self.model(sen,pos1,pos2)  
		tags = Variable(label)
		self.optimizer.zero_grad()
		loss = self.criterion(y, tags)	  
		loss.backward()
		#loss.backward(retain_graph=True)
		self.optimizer.step()	
		   
		y = np.argmax(y.data.numpy(),axis=1)

		for y1,y2 in zip(y,label):
			if y1==y2:
				acc+=1
			total+=1
		return acc,total

	def test_one_step(self):
		acc_t=0
		total_t=0
		count_predict = [0]*self.classes_nums
		count_total = [0]*self.classes_nums
		count_right = [0]*self.classes_nums
		for sen,pos1,pos2,label in test_dataloader:
			sentence = Variable(sen)
			pos1 = Variable(pos1)
			pos2 = Variable(pos2)
			y = self.model(sentence,pos1,pos2)
			y = np.argmax(y.data.numpy(),axis=1)
			for y1,y2 in zip(y,label):
				count_predict[y1]+=1
				count_total[y2]+=1
				if y1==y2:
					count_right[y1]+=1

		
		precision = [0]*self.classes_nums
		recall = [0]*self.classes_nums
		for i in range(len(count_predict)):
			if count_predict[i]!=0 :
				precision[i] = float(count_right[i])/count_predict[i]
				
			if count_total[i]!=0:
				recall[i] = float(count_right[i])/count_total[i]
		

		precision = sum(precision)/len(relation2id)
		recall = sum(recall)/len(relation2id)
		f1 =  (2*precision*recall)/(precision+recall)	
		print ("准确率：",precision)
		print ("召回率：",recall)
		print ("f1：",f1)
		return precision,recall,f1
	def do_train(self):
		self.init_model()
		# model = BiLSTM_ATT(config=self)
		# #model = torch.load('model/model_epoch20.pkl')
		# optimizer = optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=1e-5)
		# criterion = nn.CrossEntropyLoss(size_average=True)
		print('start training...')
		best_p = 0.0
		best_r = 0.0
		best_f1 = 0.0
		for epoch in range(self.max_epoches):
			print('epoch:',epoch)
			acc = 0 
			total = 0
			index = 0
			for sen,pos1,pos2,tag in self.train_dataloader:
				print('cur_batch:',index)
				index+=1
				cur_acc ,cur_total = self.train_one_step(sen,pos1,pos2,tag)
				acc+=cur_acc
				total+=cur_total
				print('batch:',index,'train precision:',float(acc)/total*100,'%')
				

			print('epoch:',epoch,'train precision:',float(acc)/total*100,'%')
			p,r,f1 = self.test_one_step()
			if f1 > best_f1:
				best_f1 = f1
				model_name = "./checkpoint/model_best"+".pkl"
				torch.save(self.model, model_name)
				print (model_name,"has been saved")
		torch.save(model, "./model/model_01.pkl")
		print ("model has been saved")





