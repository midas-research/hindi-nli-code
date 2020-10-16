import numpy as np
import torch
import pdb
import csv
from torch.nn.utils.rnn import pad_sequence

##### embedding
# xlmr = model.load('pre-trained') # load sentence representation model here 

def get_emb(sentence):
  hi_tokens = xlmr.encode(sentence)
  f = xlmr.extract_features(hi_tokens)
  return f.squeeze(0)

def extract_from_file(data_file, max_sents):
  nli_labels_to_int = {'entailed': 0, 'not-entailed': 1}
  sentiment_labels_to_int = {'positive': 0, 'negative': 1, 'neutral': 2, 'conflict': 3}

  data = {'context': [], 'hypoths': [], 'senti_lbls': [], 'nli_lbls': []}

  c = []
  h = []
  l = []
  s = []

  dataset = ''
  with open(data_file, "r") as f:
    data1 = list(csv.reader(f, delimiter='\t'))
    for i, row in enumerate(data1):
      if(len(row)==0 or i==0):
        continue
      c.append(row[1])
      h.append(row[2])
      s.append(row[3])
      l.append(row[4])

  assert len(l) == len(s), "%s: labels and source files are not same length"
  
  added_sents = 0
  
  for i in range(len(l)):
    nli_lbl = l[i].strip()
    senti_lbl = s[i].strip()
    ctxt = c[i].strip()
    hypoth = h[i].strip()

    if nli_lbl not in nli_labels_to_int:
      print(nli_lbl)
      print ("bad nli label: %s" % (nli_lbl))
      continue

    if(dataset=='review'):
      if senti_lbl not in sentiment_labels_to_int:
        print(senti_lbl)
        print ("bad sentiment label: %s" % (senti_lbl))
        continue

    if added_sents >= max_sents:
      continue

    nli_label = nli_labels_to_int[nli_lbl]

    if(dataset=='review'):
      senti_label = sentiment_labels_to_int[senti_lbl]
    else:
      senti_label = (int)(senti_lbl)

    added_sents += 1
    data['context'].append(ctxt)
    data['nli_lbls'].append(nli_label)
    data['hypoths'].append(hypoth)  
    data['senti_lbls'].append(senti_label)

  return data

def get_nli_hypoth(train_data_file, val_data_file, test_data_file, max_train_sents, max_val_sents, max_test_sents):
  labels = {}
  hypoths = {}

  train_data = extract_from_file(train_data_file, max_train_sents)
  train_length = len(train_data['nli_lbls'])

  val_data = extract_from_file(val_data_file, max_val_sents)
  val_length = len(val_data['nli_lbls'])

  test_data = extract_from_file(test_data_file, max_test_sents)
  test_length = len(test_data['nli_lbls'])

  data_train = {'nli_lbls': [], 'senti_lbls': [], 'context': [], 'hypoths': []}
  
  data_train['nli_lbls'] = train_data['nli_lbls']
  data_train['senti_lbls'] = train_data['senti_lbls']
  data_train['context'] = train_data['context']
  data_train['hypoths'] = train_data['hypoths']

  data_val = {'nli_lbls': [], 'senti_lbls': [], 'context': [], 'hypoths': []}
  data_val['nli_lbls'] = val_data['nli_lbls']
  data_val['senti_lbls'] = val_data['senti_lbls']
  data_val['context'] = val_data['context']
  data_val['hypoths'] = val_data['hypoths']

  data_test = {'nli_lbls': [], 'senti_lbls': [], 'context': [], 'hypoths': []}
  data_test['nli_lbls'] = test_data['nli_lbls']
  data_test['senti_lbls'] = test_data['senti_lbls']
  data_test['context'] = test_data['context']
  data_test['hypoths'] = test_data['hypoths']

  return data_train, data_val, data_test

def get_batch(batch, word_emb_dim):
  # embed = np.zeros((len(batch), word_emb_dim))
  embed = []
  
  defined_length = 32

  for i in range(len(batch)):
    x = get_emb(batch[i])
    
    if(x.shape[0]<defined_length):
      pad = defined_length - x.shape[0]
      padded = torch.zeros(pad, x.shape[1])
      x = torch.cat([x, padded], dim=0)
    else:
      x = x[:defined_length, :]
    
    avg_pool = torch.mean(x, axis=0)
    embed.append(avg_pool)
  embed = pad_sequence(embed, batch_first=True, padding_value=0.)
  
  return torch.FloatTensor(embed)