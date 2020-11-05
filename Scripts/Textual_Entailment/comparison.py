import os
import sys
import time
import argparse
import pdb

import numpy as np

import torch
from torch.autograd import Variable
import torch.nn as nn

from model_data import get_nli_hypoth, get_batch
from nli_model import NLI_HYPOTHS_Net
from classification_model import Classifier_Net
from mutils import get_optimizer
import torch.nn.functional as F

def get_args():
  parser = argparse.ArgumentParser(description='Training NLI model for sent2vec embedding')

  # paths
  parser.add_argument("--outputdir", type=str, default='./saved_models/', help="Output directory")
  parser.add_argument("--outputmodelname", type=str, default='nli.pickle')
  parser.add_argument("--outputmodelname", type=str, default='clf.pickle')
  
  # data
  parser.add_argument("--train_data", type=str, default='./sentiment_data/Product_Review_Dataset/recasted_train_with_negation.tsv', help="data file containing the context-hypothesis pair along with the nli label.")
  parser.add_argument("--val_data", type=str, default='./sentiment_data/Product_Review_Dataset/recasted_dev_with_negation.tsv', help="data file containing the context-hypothesis pair along with the nli label.")
  parser.add_argument("--test_data", type=str, default='./sentiment_data/Product_Review_Dataset/recasted_test_with_negation.tsv', help="data file containing the context-hypothesis pair along with the nli label.")
  parser.add_argument("--max_train_sents", type=int, default=10000000, help="Maximum number of training examples")
  parser.add_argument("--max_val_sents", type=int, default=10000000, help="Maximum number of validation/dev examples")
  parser.add_argument("--max_test_sents", type=int, default=10000000, help="Maximum number of test examples")
  
  # training
  parser.add_argument("--n_epochs", type=int, default=10)
  parser.add_argument("--n_classes", type=int, default=2)
  parser.add_argument("--n_sentiment", type=int, default=4)
  parser.add_argument("--batch_size", type=int, default=128)
  parser.add_argument("--dpout_model", type=float, default=0., help="encoder dropout")
  parser.add_argument("--dpout_fc", type=float, default=0., help="classifier dropout")
  parser.add_argument("--optimizer", type=str, default="adam,lr=0.0001", help="adam or sgd,lr=0.1")
  parser.add_argument("--lrshrink", type=float, default=5., help="shrink factor for sgd")
  parser.add_argument("--decay", type=float, default=0.99, help="lr decay")
  parser.add_argument("--minlr", type=float, default=1e-5, help="minimum lr")
  parser.add_argument("--is_cr", type=bool, default=True, help="whether or not to include constranit regularization while training")
  
  parser.add_argument("--embedding_size", type=int, default=1024, help="sentence embedding size in the trained embedding model")
  parser.add_argument("--max_norm", type=int, default=5., help="maximum norm value")
  
  # gpu
  parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID")
  parser.add_argument("--seed", type=int, default=1234, help="seed")
  parser.add_argument("--load_saved", type=bool, default=True, help="seed")

  #misc
  parser.add_argument("--verbose", type=int, default=1, help="Verbose output")

  params, _ = parser.parse_known_args()
  args = params

  return params


def evaluate(epoch, valid, optimizer, args, nli_net, clf_net, eval_type='valid', final_eval=False):
  nli_net.eval()
  clf_net.eval()
  correct = 0.
  global val_acc_best, lr, stop_training, adam_stop

  if eval_type == 'valid':
    print('\n{0} : Epoch {1}'.format(eval_type, epoch))

  context = valid['context']
  hypoths = valid['hypoths']
  target = valid['nli_lbls']

  for i in range(0, len(hypoths), args.batch_size):
    
    context_batch = get_batch(context[i:i + args.batch_size], args.embedding_size)
    hypoths_batch = get_batch(hypoths[i:i + args.batch_size], args.embedding_size)
    
    tgt_batch = None
    if args.gpu_id > -1:
      context_batch = Variable(context_batch.cuda())
      hypoths_batch = Variable(hypoths_batch.cuda())
      tgt_batch = Variable(torch.LongTensor(target[i:i + args.batch_size])).cuda()
    else:
      context_batch = Variable(context_batch)
      hypoths_batch = Variable(hypoths_batch)
      tgt_batch = Variable(torch.LongTensor(target[i:i + args.batch_size]))

    output = nli_net(context_batch, hypoths_batch)
    pred = torch.argmax(output, dim=1)
    correct += pred.long().eq(tgt_batch.data.long()).float().cpu().sum()

    output_clf = clf_net(context_batch)
    pred_clf = torch.argmax(output, dim=1)
    correct_clf += pred_clf.long().eq(tgt_batch.data.long()).float().cpu().sum()

    a1 = torch.where(pred_clf.long()==tgt_batch.data.long(), pred_clf, tgt_batch)
    both_correct = torch.where(a1==pred, a1, pred).sum()

    a2 = torch.where(pred_clf.long()!=tgt_batch.data.long(), pred_clf, tgt_batch)
    both_incorrect = torch.where(a1==pred, a1, pred).sum()

    a3 = torch.where(pred.long()==tgt_batch.data.long(), pred_clf, tgt_batch)
    dir_correct_incorrect = torch.where(pred_clf!=a3, pred_clf, a3).sum()

    a4 = torch.where(pred_clf.long()==tgt_batch.data.long(), pred_clf, tgt_batch)
    dir_incorrect_correct = torch.where(pred!=a4, pred, a4).sum()

  ###### save model
  eval_acc_two_step = (100 * correct / len(context)).item()
  eval_acc_direct = (100 * correct / len(context)).item()

  print('Both Direct and Two-step classification are correct:', both_correct * 100. / len(tgt_batch))
  print('Both Direct and Two-step classification are incorrect:', both_incorrect * 100. / len(tgt_batch))
  print('Correct Direct Classification but incorrect Two-step Classification:', dir_correct_incorrect * 100. / len(tgt_batch))
  print('Incorrect Direct Classification but correct Two-step Classification:', dir_incorrect_correct * 100. / len(tgt_batch))
  
  return eval_acc_two_step, eval_acc_direct


def main(args):
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)
  
  if args.gpu_id > -1:
    torch.cuda.manual_seed(args.seed)

  train, val, test = get_nli_hypoth(args.train_data, args.val_data, args.test_data, args.max_train_sents, args.max_val_sents, args.max_test_sents)

  nli_net = NLI_HYPOTHS_Net()
  clf_net = Classifier_Net()

  if(args.load_saved):
    nli_net = torch.load(os.path.join(args.outputdir, args.outputmodelname))
    clf_net = torch.load(os.path.join(args.outputdir, args.outputclfmodelname))
  
  if args.gpu_id > -1:
    nli_net.cuda()
    loss_mse.cuda()

  two_step_acc, direct_acc = evaluate("NO", test, optimizer , args , nli_net , clf_net , "test")

if __name__ == '__main__':
  args = get_args()
  main(args)
