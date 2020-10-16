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
from mutils import get_optimizer
import torch.nn.functional as F

def get_args():
  parser = argparse.ArgumentParser(description='Training NLI model for sent2vec embedding')

  # paths
  parser.add_argument("--outputdir", type=str, default='./saved_models/', help="Output directory")
  parser.add_argument("--outputmodelname", type=str, default='constraint.pickle')
  
  # data
  parser.add_argument("--train_data", type=str, default='./sentiment_data/Product_Review_Dataset/recasted_train_with_negation.tsv', help="data file containing the context-hypothesis pair along with the nli label.")
  parser.add_argument("--val_data", type=str, default='./sentiment_data/Product_Review_Dataset/recasted_dev_with_negation.tsv', help="data file containing the context-hypothesis pair along with the nli label.")
  parser.add_argument("--test_data", type=str, default='./sentiment_data/Product_Review_Dataset/recasted_test_with_negation.tsv', help="data file containing the context-hypothesis pair along with the nli label.")
  parser.add_argument("--max_train_sents", type=int, default=10000000, help="Maximum number of training examples")
  parser.add_argument("--max_val_sents", type=int, default=10000000, help="Maximum number of validation/dev examples")
  parser.add_argument("--max_test_sents", type=int, default=10000000, help="Maximum number of test examples")
  
  # training
  parser.add_argument("--n_epochs", type=int, default=15)
  parser.add_argument("--n_classes_nli", type=int, default=2)
  parser.add_argument("--n_classes_clf", type=int, default=4)
  parser.add_argument("--batch_size", type=int, default=128)
  parser.add_argument("--dpout_model", type=float, default=0., help="encoder dropout")
  parser.add_argument("--dpout_fc", type=float, default=0., help="classifier dropout")
  parser.add_argument("--optimizer", type=str, default="adam,lr=0.001", help="adam or sgd,lr=0.1")
  parser.add_argument("--lrshrink", type=float, default=5., help="shrink factor for sgd")
  parser.add_argument("--decay", type=float, default=0.9, help="lr decay")
  parser.add_argument("--minlr", type=float, default=1e-5, help="minimum lr")
  
  parser.add_argument("--embedding_size", type=int, default=1024, help="sentence embedding size in the trained embedding model")
  parser.add_argument("--max_norm", type=int, default=5., help="maximum norm value")
  parser.add_argument("--reg_lambda", type=int, default=2., help="lambda for supervised loss")
  
  # gpu
  parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID")
  parser.add_argument("--seed", type=int, default=1234, help="seed")
  parser.add_argument("--load_saved", type=bool, default=True, help="seed")

  #misc
  parser.add_argument("--verbose", type=int, default=1, help="Verbose output")

  params, _ = parser.parse_known_args()
  args = params

  return params

def get_ordered_batch(context, hypothesis, target):
  c_wo_n, h_wo_n, t_wo_n = [], [], []
  c_w_n, h_w_n, t_w_n = [], [], []
  
  for i in range(len(context)):
    if(i%2==0):
      c_wo_n.append(context[i])
      h_wo_n.append(hypothesis[i])
      t_wo_n.append(target[i])
    else:
      c_w_n.append(context[i])
      h_w_n.append(hypothesis[i])
      t_w_n.append(target[i])

  return c_wo_n, h_wo_n, t_wo_n, c_w_n, h_w_n, t_w_n

def trainepoch(epoch, train, optimizer, args, nli_net, loss_fn, loss_mse):
  nli_net.train()
  print('\nTRAINING : Epoch ' + str(epoch))
  
  nli_net.train()
  all_costs = []
  logs = []
  
  last_time = time.time()
  correct = 0.
  
  # shuffle the data
  permutation = np.random.permutation(len(train['hypoths']))
  context, hypoths, target = train['context'], train['hypoths'], train['nli_lbls']

  optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * args.decay if epoch>1\
      and 'sgd' in args.optimizer else optimizer.param_groups[0]['lr']
  print('Learning rate : {0}'.format(optimizer.param_groups[0]['lr']))

  trained_sents = 0
  counter = 1

  start_time = time.time()
  for stidx in range(0, len(hypoths), args.batch_size):
    # prepare batch
    c_wo_n, h_wo_n, t_wo_n, c_w_n, h_w_n, t_w_n = get_ordered_batch(context[stidx:stidx + args.batch_size], hypoths[stidx:stidx + args.batch_size], target[stidx:stidx + args.batch_size])
    
    c_wo_n, c_w_n = get_batch(c_wo_n, args.embedding_size), get_batch(c_w_n, args.embedding_size)
    h_wo_n, h_w_n = get_batch(h_wo_n, args.embedding_size), get_batch(h_w_n, args.embedding_size)
    
    tgt_batch = None

    if args.gpu_id > -1:
      c_wo_n, c_w_n = Variable(c_wo_n.cuda()), Variable(c_w_n.cuda())
      h_wo_n, h_w_n = Variable(h_wo_n.cuda()), Variable(h_w_n.cuda())
      t_wo_n, t_w_n = Variable(torch.LongTensor(t_wo_n).cuda()), Variable(torch.LongTensor(t_w_n).cuda())
    else:
      c_wo_n, c_w_n = Variable(c_wo_n), Variable(c_w_n)
      h_wo_n, h_w_n = Variable(h_wo_n), Variable(h_w_n)
      t_wo_n, t_w_n = Variable(torch.LongTensor(t_wo_n)), Variable(torch.LongTensor(t_w_n))

    k = 2 * t_wo_n.shape[0]

    # model forward
    optimizer.zero_grad()

    op_wo_vector = nli_net(c_wo_n, h_wo_n)
    op_w_vector = nli_net(c_w_n, h_w_n)

    p_wo_vector = torch.sigmoid(op_wo_vector)
    p_w_vector = torch.sigmoid(op_w_vector)

    pred = torch.argmax(op_wo_vector, dim=1)
    
    correct += pred.long().eq(t_wo_n.data.long()).float().cpu().sum()
    assert len(pred) == len(h_wo_n)

    supervised_loss = loss_fn(op_wo_vector, t_wo_n)
    constraint_loss = loss_mse(p_wo_vector, torch.ones_like(p_wo_vector)-p_w_vector)

    loss = args.reg_lambda * supervised_loss + constraint_loss

    all_costs.append(loss.item())
    
    loss.backward()

    shrink_factor = 0.9
    total_norm = 0

    for p in nli_net.parameters():
      if p.requires_grad:
        p.grad.data.div_(k)
        total_norm += p.grad.data.norm() ** 2
    total_norm = np.sqrt(total_norm.cpu().data)

    if total_norm > args.max_norm:
        shrink_factor = args.max_norm / total_norm

    if('sgd' in args.optimizer):
      current_lr = optimizer.param_groups[0]['lr']
      optimizer.param_groups[0]['lr'] = current_lr * shrink_factor

    optimizer.step()
    
    if('sgd' in args.optimizer):
      optimizer.param_groups[0]['lr'] = current_lr

    if args.verbose:
      trained_sents += len(t_wo_n)
      counter += 1
      print("supervised_loss: ", supervised_loss.item())
      print("constraint_loss: ", constraint_loss.item())
      print("Total_loss: ", loss.item())
      print ("epoch: %d -- correct %d / %d trained  ----- accuracy %d " % (epoch, correct, trained_sents, ((correct * 100 ) / trained_sents).item()))
      
  train_acc = (200 * correct / len(hypoths)).item()
  print('results : epoch {0} ; mean accuracy train : {1}, loss : {2}'
          .format(epoch, train_acc, round(np.mean(all_costs), 2)))
  
  return train_acc, nli_net

def evaluate(epoch, valid, optimizer, args, nli_net, eval_type='valid', final_eval=False):
  nli_net.eval()
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

  ###### save model
  eval_acc = (100 * correct / len(hypoths)).item()
  
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
      torch.save(nli_net, os.path.join(args.outputdir, args.outputmodelname))
      val_acc_best = eval_acc
    else:
      if 'sgd' in args.optimizer:
        optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] / args.lrshrink
        if optimizer.param_groups[0]['lr'] < args.minlr:
          stop_training = True
      if 'adam' in args.optimizer:
        stop_training = adam_stop
        adam_stop = True
  return eval_acc

def main(args):

  np.random.seed(args.seed)
  torch.manual_seed(args.seed)
  
  if args.gpu_id > -1:
    torch.cuda.manual_seed(args.seed)

  train, val, test = get_nli_hypoth(args.train_data, args.val_data, args.test_data, args.max_train_sents, args.max_val_sents, args.max_test_sents)

  nli_net = NLI_HYPOTHS_Net()
  if(args.load_saved):
    nli_net = torch.load('./saved_models/nli_models/model_nli_BHAAV_2.pickle')
  
  # loss
  loss_fn = nn.CrossEntropyLoss()
  loss_mse = nn.MSELoss()
  
  # optimizer
  optim_fn, optim_params = get_optimizer(args.optimizer)
  optimizer = optim_fn(nli_net.parameters(), **optim_params)

  if args.gpu_id > -1:
    nli_net.cuda()
    loss_fn.cuda()
    loss_mse.cuda()

  global val_acc_best, lr, stop_training, adam_stop
  val_acc_best = -1e10
  adam_stop = False
  stop_training = False
  lr = optim_params['lr'] if 'sgd' in args.optimizer else None

  epoch = 1

  while not stop_training and epoch <= args.n_epochs:
    train_acc, nli_net = trainepoch(epoch, train, optimizer, args, nli_net, loss_fn, loss_mse)
    eval_acc = evaluate(epoch, val, optimizer, args, nli_net, 'valid')
    epoch += 1

  nli_net = torch.load(os.path.join(args.outputdir, args.outputmodelname))

  print("The Tests Accuracy is ",evaluate("NO", test, optimizer , args , nli_net , "test"))

if __name__ == '__main__':
  args = get_args()
  main(args)
