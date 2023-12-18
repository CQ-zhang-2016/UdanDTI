import os
import random
import numpy as np
import torch
import dgl
import logging
import torch.nn.functional as F
import torch.nn as nn

def set_seed(seed=1000):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def graph_collate_func(x):
    d, p, y = zip(*x)
    d = dgl.batch(d)
    p = torch.stack(p,0)
    
    return d, p, torch.tensor(y)


def mkdir(path):
    path = path.strip()
    path = path.rstrip("\\")
    is_exists = os.path.exists(path)
    if not is_exists:
        os.makedirs(path)


def generate_protein(iind, seq, LLM_type):
    LLM_path = './datasets/protein_features/'+LLM_type+'/'
    if LLM_type == 'esm':
        max_length = 1200
        protein = torch.load(LLM_path+str(iind)+'.pt')[1:-1]
    else:
        max_length = 1202
        protein = torch.load(LLM_path+str(iind)+'.pt')

    assert (len(protein)==len(seq)) or (len(protein)==max_length)
    encoding = torch.zeros((max_length, protein.shape[-1]))
    encoding[:len(protein)] = protein

    return encoding

def binary_cross_entropy(pred_output, labels):
    loss_fct = torch.nn.BCELoss()
    m = torch.nn.Sigmoid()
    
    n = torch.squeeze(m(pred_output))
    loss = loss_fct(n, labels)
    
    return n, loss
