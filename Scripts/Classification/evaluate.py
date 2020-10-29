import os
import sys
import time
import argparse
import pdb

import numpy as np

import torch
from torch.autograd import Variable
import torch.nn as nn

from classification_data import get_clf, get_batch, get_embeddings_xlmr

from classification_model import Classifier_Net
from mutils import get_optimizer 

device = "cuda:1"

def get_args():
  parser = argparse.ArgumentParser(description='Training NLI model for roberta embedding')

  # paths
  parser.add_argument("--outputdir", type=str, default='saved_models/classification_models/', help="Output directory")
  parser.add_argument("--outputmodelname", type=str, default='classifier_Roberta_PR1.pickle')
  parser.add_argument('--cuda', type=bool, default=True, help="run the following code on a GPU")

  # data
  parser.add_argument("--train_data", type=str, default='./Product_Review_Dataset/PR_train_data.tsv', help="train data file containing the text-label.")
  parser.add_argument("--val_data", type=str, default='./Product_Review_Dataset/PR_dev_data.tsv', help="val data file containing the text-label.")
  parser.add_argument("--test_data", type=str, default='./Product_Review_Dataset/PR_test_data.tsv', help="test data file containing the text-label.")
  parser.add_argument("--max_train_sents", type=int, default=10000000, help="Maximum number of training examples")
  parser.add_argument("--max_val_sents", type=int, default=10000000, help="Maximum number of validation/dev examples")
  parser.add_argument("--max_test_sents", type=int, default=10000000, help="Maximum number of test examples")
  
  # training
  parser.add_argument("--n_epochs", type=int, default=10)
  parser.add_argument("--n_classes", type=int, default=4)
  parser.add_argument("--batch_size", type=int, default=128)
  parser.add_argument("--dpout_model", type=float, default=0., help="encoder dropout")
  parser.add_argument("--dpout_fc", type=float, default=0., help="classifier dropout")
  parser.add_argument("--nonlinear_fc", type=float, default=0, help="use nonlinearity in fc")
  parser.add_argument("--optimizer", type=str, default="adam,lr=0.05", help="adam or sgd,lr=0.1")
  parser.add_argument("--lrshrink", type=float, default=5, help="shrink factor for sgd")
  parser.add_argument("--decay", type=float, default=0.99, help="lr decay")
  parser.add_argument("--minlr", type=float, default=1e-5, help="minimum lr")
  parser.add_argument("--embedding_size", type=int, default=4096, help="sentence embedding size in the sent2vec trained embedding model")
  parser.add_argument("--load_saved", type=bool, default=False, help="load checkpoints from the saved model")
  parser.add_argument("--seed", type=int, default=1234, help="seed")

  #misc
  parser.add_argument("--verbose", type=int, default=1, help="Verbose output")

  params, _ = parser.parse_known_args()
  args = params

  return params

def evaluate(epoch, valid, optimizer, args, net, eval_type='valid', final_eval=False):
  net.eval()
  correct = 0.
  global  val_acc_best, lr, stop_training, adam_stop

  if eval_type == 'valid':
    print('\n{0} : Epoch {1}'.format(eval_type, epoch))

  text = valid['text']
  target = valid['lbls']
  print(len(target))
  counter = 1

  eval_acc = (100 * correct / len(text)).item()
  if final_eval:
    print('finalgrep : accuracy {0} : {1}'.format(eval_type, eval_acc))
  else:
    print('togrep : results : epoch {0} ; mean accuracy {1} :\
              {2}'.format(epoch, eval_type, eval_acc))

  return eval_acc

def main(args):

  np.random.seed(args.seed)
  torch.manual_seed(args.seed)
  
  if args.cuda:
    torch.cuda.manual_seed(args.seed)

  train, val, test = get_clf(args.train_data, args.val_data, args.test_data, args.max_train_sents, args.max_val_sents, args.max_test_sents)
  
  net = Classifier_Net()
  if(args.load_saved):
    print('Loaded from saved model ..... ')
    net = torch.load(os.path.join(args.outputdir, args.outputmodelname))

  if args.cuda:
    net.cuda()
        
  print("The Tests Accuracy is ",evaluate("NO", test, optimizer , args , net , "test"))


if __name__ == '__main__':
  args = get_args()
  main(args)
