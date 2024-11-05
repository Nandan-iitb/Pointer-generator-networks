import pandas as pd
import numpy as np
import os
import pickle
from collections import Counter
from typing import Callable, Optional
from copy import deepcopy
import math
from code_2 import Parameters, simple_tokenizer, Vocab, load_vocab, Dataset, MyDataset, tensorize, pad_sequence, preprocess, sort_batch_by_len, EncoderRNN, get_next_batch, get_coverage_vector,DecoderRNN, predict, get_predictions 
# Import pytoch libraries and modules 
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import optim
from torch.nn.utils import clip_grad_norm_
#from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import pkbar
from rouge import Rouge
#Import libraries for text procesing
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')  
params = Parameters()
all_in_one_dataset = Dataset(params.all_in_one_path, simple_tokenizer, params.max_src_len, params.max_tgt_len, max_rows=1e5,
                        truncate_src=True, truncate_tgt=True)
# convert the embeddings to a tensor

vocab = all_in_one_dataset.build_vocab(params.vocab_min_frequency, embed_file=params.embed_file)
embedding_weights = torch.from_numpy(vocab.embeddings)
preds = predict(all_in_one_dataset, vocab,params, embedding_weights)
df = pd.read_csv(r"/workspace/myproject/test_w_splits-test_w_splits.csv")
preds = preds[:7226]
df.to_csv("/workspace/myproject/lstm_based_ptr_gtr_64_128/samples.py")
#print(predict([x_test[0]], vocab, params))
#y_pred = get_predictions(x_test, vocab, params)

print("Jai Shree Ram")