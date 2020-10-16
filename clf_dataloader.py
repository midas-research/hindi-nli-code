import numpy as np
import torch
import pdb
import csv
from torch.nn.utils.rnn import pad_sequence

##### FOR XLMR Model

from fairseq.models.roberta import XLMRModel

if torch.cuda.is_available():
    map_location=lambda storage, loc: storage.cuda()
else:
    map_location='cpu'

# model = torch.load('./xlmr.large/XLMR_BBC.pickle', map_location=map_location)
# model.eval()
xlmr = XLMRModel.from_pretrained('xlmr.large', checkpoint_file='model.pt')
xlmr.eval()

def get_emb(sentence):
  hi_tokens = xlmr.encode(sentence)
  f = xlmr.extract_features(hi_tokens)
  return f.squeeze(0)

def extract_from_file(data_file, max_sents):
  data = {'lbls': [], 'text': []}
  label_to_int = {'positive': 0, 'negative': 1, 'neutral': 2, 'conflict': 3}

  t = []
  l = []
  with open(data_file, "r") as f:
    data1 = list(csv.reader(f, delimiter='\t'))
    for i, row in enumerate(data1):
      if(i==0):
        continue
      if(len(row)==0):
        continue
      if(len(row)==2):
        t.append(row[0])
        l.append((int)(row[1]))
      elif(len(row)==3):
        t.append(row[1])
        #l.append(row[2])
        l.append(label_to_int[row[2]])
      else:
        t.append(row[0])
        l.append(int(row[1]))
      
  assert len(l) == len(t), "%s: labels and source files are not same length"
  added_sents = 0
  for i in range(len(l)):
    lbl = l[i]
    txt = t[i].strip()
    
    if int(lbl) > 4 or int(lbl) < 0:
      print ("bad label: %s" % (lbl))
      continue

    if added_sents >= max_sents:
      continue

    label = int(lbl)

    data['lbls'].append(label)
    data['text'].append(txt)
    added_sents += 1

  return data

def get_clf(train_data_file, val_data_file, test_data_file, max_train_sents, max_val_sents, max_test_sents):
  labels = {}
  text = {}

  train_data = extract_from_file(train_data_file, max_train_sents)
  train_length = len(train_data['lbls'])

  val_data = extract_from_file(val_data_file, max_val_sents)
  val_length = len(val_data['lbls'])

  test_data = extract_from_file(test_data_file, max_test_sents)
  test_length = len(test_data['lbls'])

  data_train = {'lbls': [], 'text': []}
  data_train['lbls'] = train_data['lbls']
  data_train['text'] = train_data['text']
  
  data_val = {'lbls': [], 'text': []}
  data_val['lbls'] = val_data['lbls']
  data_val['text'] = val_data['text']
  
  data_test = {'lbls': [], 'text': []}
  data_test['lbls'] = test_data['lbls']
  data_test['text'] = test_data['text']
  
  return data_train, data_val, data_test

# def get_batch(batch, word_emb_dim=None):
#   # embed = np.zeros((len(batch), word_emb_dim))
#   embed = []
  
#   defined_length = 64
#   for i in range(len(batch)):
#     x = get_emb(batch[i])
#     if(x.shape[0]<defined_length):
#       pad = defined_length - x.shape[0]
#       padded = torch.zeros(pad, x.shape[1])
#       x = torch.cat([x, padded], dim=0)
#     else:
#       x = x[:defined_length, :]
#     embed.append(x)
#   embed = pad_sequence(embed, batch_first=True, padding_value=0.)
#   return embed.float()

def get_batch(batch):
  # embed = np.zeros((len(batch), word_emb_dim))
  embed = []
  
  defined_length = 96

  for i in range(len(batch)):
    x = get_emb(batch[i])
    
    if(x.shape[0]<defined_length):
      pad = defined_length - x.shape[0]
      padded = torch.zeros(pad, x.shape[1])
      x = torch.cat([x, padded], dim=0)
    else:
      x = x[:defined_length, :]
    
    avg_pool = torch.mean(x, axis=0)
    embed.append(avg_pool) # shape of x: 64 x 768 (for base model); 768 after average pooling
  embed = pad_sequence(embed, batch_first=True, padding_value=0.)
  
  return torch.FloatTensor(embed)

def get_embeddings_xlmr(sent_list):
    embed_list = torch.Tensor([])
    for i,sent in enumerate(sent_list):
        tokens = xlmr.encode(sent)
        if len(tokens) >= 512 :
            tokens = tokens[:512]
        last_layer_features = xlmr.extract_features(tokens).to("cpu")
        avg_pool = torch.sum(last_layer_features,axis=1)/last_layer_features.size(1)
        embed_list = torch.cat((embed_list,avg_pool),dim=0)
    return embed_list
    
def get_embeddings_valid_xlmr(sent_list):
    # xlmr.to("cuda:2")
    embed_list = torch.Tensor([])
    for i,sent in enumerate(sent_list):
        tokens = xlmr.encode(sent)
        if len(tokens) >= 512 :
            tokens = tokens[:512]
        last_layer_features = xlmr.extract_features(tokens).to("cpu")
        avg_pool = torch.sum(last_layer_features,axis=1)/last_layer_features.size(1)
        embed_list = torch.cat((embed_list,avg_pool),dim=0)
    # xlmr.to("cuda:3")
    return embed_list
