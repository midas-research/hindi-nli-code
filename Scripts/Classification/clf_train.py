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


def trainepoch(epoch, train, optimizer, args, net, loss_fn):
  print('\nTRAINING : Epoch ' + str(epoch))
  net.train()
  all_costs = []
  logs = []
  last_time = time.time()
  correct = 0.
  
  # shuffle the data
  permutation = np.random.permutation(len(train['text']))

  text , target = [], [] 
  for i in permutation:
    text.append(train['text'][i])
    target.append(train['lbls'][i])
    

  optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * args.decay if epoch>1\
      and 'sgd' in args.optimizer else optimizer.param_groups[0]['lr']
  print('Learning rate : {0}'.format(optimizer.param_groups[0]['lr']))

  trained_sents = 0

  start_time = time.time()
  counter = 1
  for stidx in range(0, len(text), args.batch_size):
    # prepare batch
    if stidx % (100 * args.batch_size) == 0:
        print(stidx)
    # text_batch = get_batch(text[stidx:stidx + args.batch_size])
    
    text_batch = get_embeddings_xlmr(text[stidx:stidx + args.batch_size])
    tgt_batch = torch.from_numpy(np.asarray(target[stidx:stidx + args.batch_size]))
    
    if args.cuda:
      text_batch = Variable(text_batch).cuda()
      tgt_batch = Variable(tgt_batch).cuda()
    else:
      text_batch = Variable(text_batch)
      tgt_batch = Variable(tgt_batch)
    
    k = text_batch.size(1)  # actual batch size

    # model forward
    output = net(text_batch)
    pred = torch.argmax(output, dim=1)

    #pred = output.data.max(1)[1]
    correct += (pred == tgt_batch).cpu().sum()
    assert len(pred) == len(text[stidx:stidx + args.batch_size])
    
    # loss
    loss = loss_fn(output, tgt_batch)
    all_costs.append(loss.item())
    
    # backward
    optimizer.zero_grad()
    loss.backward()

    # gradient clipping (off by default)
    shrink_factor = 1
    total_norm = 0
    
    
    for p in net.parameters():
      if p.requires_grad:
        p.grad.data.div_(k)  # divide by the actual batch size
        total_norm += p.grad.data.norm() ** 2
    total_norm = np.sqrt(total_norm.cpu())

    current_lr = optimizer.param_groups[0]['lr'] # current lr (no external "lr", for adam)
    optimizer.param_groups[0]['lr'] = current_lr * shrink_factor # just for update

    # optimizer step
    optimizer.step()
    optimizer.param_groups[0]['lr'] = current_lr

    if args.verbose:
      trained_sents = counter*args.batch_size
      counter += 1
      print ("epoch: %d -- correct %d / %d trained " % (epoch, correct, trained_sents), "loss: ", loss.item())
      
  train_acc = (100 * correct/len(text)).item()
  print('results : epoch {0} ; mean accuracy train : {1}, loss : {2}'
          .format(epoch, train_acc, round(np.mean(all_costs), 2)))
  return train_acc, net


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

  for i in range(0, len(text), args.batch_size):
    # prepare batch
    text_batch = get_embeddings_xlmr(text[i:i + args.batch_size]) 
    # text_batch = get_batch(text[i:i + args.batch_size]) 
    tgt_batch = torch.from_numpy(np.asarray(target[i:i + args.batch_size])) 
    
    if args.cuda:
      text_batch = Variable(text_batch).cuda()
      tgt_batch = Variable(tgt_batch).cuda()
    else:
      text_batch = Variable(text_batch)
      tgt_batch = Variable(tgt_batch)
    # model forward
    output = net(text_batch)
    pred = torch.argmax(output, dim=1)
    # pred = output.data.max(1)[1]
    correct += pred.long().eq(tgt_batch.data.long()).cpu().sum()

    # if args.verbose:
    #   validated_sents = counter*args.batch_size
    #   counter += 1
    #   print ("epoch: %d -- correct %d / %d validated " % (epoch, correct, validated_sents))

  # save model
  eval_acc = (100 * correct / len(text)).item()
  if final_eval:
    print('finalgrep : accuracy {0} : {1}'.format(eval_type, eval_acc))
  else:
    print('togrep : results : epoch {0} ; mean accuracy {1} :\
              {2}'.format(epoch, eval_type, eval_acc))

  if eval_type == 'valid' and epoch <= args.n_epochs:
    if eval_acc > val_acc_best:
      print('saving model at epoch {0}'.format(epoch))
      if not os.path.exists(args.outputdir):
        os.makedirs(args.outputdir)
      torch.save(net, os.path.join(args.outputdir, args.outputmodelname))
      val_acc_best = eval_acc
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
  
  # loss
  # weight = torch.FloatTensor(args.n_classes).fill_(1)
  loss_fn = nn.CrossEntropyLoss() #weight=weight)
  loss_fn.size_average = False

  # optimizer
  optim_fn, optim_params = get_optimizer(args.optimizer)
  optimizer = optim_fn(net.parameters(), **optim_params)
  
  if args.cuda:
    net.cuda()
    loss_fn.cuda()
    
  global val_acc_best, lr, stop_training, adam_stop
  val_acc_best = -1e10
  adam_stop = False
  stop_training = False
  lr = optim_params['lr'] if 'sgd' in args.optimizer else 0.005

  epoch = 1

  while not stop_training and epoch <= args.n_epochs:
    train_acc, net = trainepoch(epoch, train, optimizer, args, net, loss_fn)
    eval_acc = evaluate(epoch, val, optimizer, args, net, 'valid')
    epoch += 1

  # net = torch.load(os.path.join(args.outputdir, args.outputmodelname))
    
  print("The Tests Accuracy is ",evaluate("NO", test, optimizer , args , net , "test"))


if __name__ == '__main__':
  args = get_args()
  main(args)
