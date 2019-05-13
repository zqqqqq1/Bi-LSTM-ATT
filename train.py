# -*- coding: utf-8 -*-
from config.Config import Config
import argparse
#if __name__=='__main__':
import sys
import os
if __name__=='__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-l', type=str, default='eng', help='language type')
	parser.add_argument('-t', type=str, default='train', help='language type')
	parser.add_argument('-p', type=str, default=None, help='use pretrain_model or not')
	args = parser.parse_args()
	#os.environ["CUDA_VISIBLE_DEVICES"] = "2"
		#if len(sys.argv)>=1:
		#   type = sys.argv[1]
	config = Config(file_type = args.l)
	config.pretrain_model = args.p
	#config.file_type = args.l
		#config.file_type = type
	config.do_train()
