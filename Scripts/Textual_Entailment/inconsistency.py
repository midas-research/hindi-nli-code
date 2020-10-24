import os
import sys
import time
import argparse
import pdb

import numpy as np

import torch
from torch.autograd import Variable
import torch.nn as nn

from data import get_nli_hypoth, get_batch
# from get_embedding import get_emb
from nli_model import NLI_HYPOTHS_Net

def eval_coonsistency():
	correct, incorrect, inconsistent = 0., 0., 0.
	total = 0.0

	context = valid['context']
	hypoths = valid['hypoths']
	target = valid['nli_lbls']

	for i in range(0, len(hypoths), 2):

		context_batch = get_batch(context[i:i + 2], args.embedding_size)
		hypoths_batch = get_batch(hypoths[i:i + 2], args.embedding_size)

		context_batch = Variable(context_batch.cuda())
		hypoths_batch = Variable(hypoths_batch.cuda())
		tgt_batch = Variable(torch.LongTensor(target[i:i + 2])).cuda().long([0])
		

		output = nli_net(context_batch, hypoths_batch)
		pred = torch.argmax(output, dim=1).long()
		
		if(pred[0]==pred[1]):
			inconsistent += 1 
		elif(pred[0]==tgt_batch.data[0]):
			correct += 1
		else:
			incorrect += 1

		total += 1

	###### save model
	corr_precentage = (100 * correct / total).item()
	incorr_precentage = (100 * incorrect / total).item()
	cons_precentage = (100 * inconsistent / total).item()

	print('Correct %', corr_percentage)
	print('Incorrect %', incorr_percentage)
	print('Inconsistent %', cons_percentage)

if __name__ == '__main__':
	nli_net = NLI_HYPOTHS_Net().eval().cuda()
	nli_net = torch.load(os.path.join(args.outputdir, args.outputmodelname))
	eval_consistency()