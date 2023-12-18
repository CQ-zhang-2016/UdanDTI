import torch.nn as nn
import torch.nn.functional as F
import torch
from dgllife.model.gnn import GCN
from einops import repeat
from attention import aggregation



class Model(nn.Module):
    def __init__(self, emb_dim, LLM_type, task):
        super(Model, self).__init__()
        drug_in_feats = 75
        drug_embedding = emb_dim
        drug_hidden_feats = [emb_dim, emb_dim, emb_dim]
        if LLM_type == 'esm':
            protein_hid_dim = 2560
        else: 
            protein_hid_dim = 1024
        protein_emb_dim = emb_dim

        self.drug_extractor = MolecularGCN(in_feats=drug_in_feats, dim_embedding=drug_embedding,
                                           hidden_feats=drug_hidden_feats)
        self.protein_extractor = ProteinCNN(protein_hid_dim, protein_emb_dim)
        self.aggregation = aggregation(depth=1)

        self.cls_token = nn.Parameter(torch.randn(1, 1, 128))
        self.task = task
        if self.task != 'cluster':
            self.decoder = Decoder(128, 1024, 256, 1)
        

    def forward(self, bg_d, v_p, mode="train"):
        v_d = self.drug_extractor(bg_d)
        v_p = self.protein_extractor(v_p)
        
        bs = v_p.size(0)

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = bs)
        cls_tokens = self.aggregation(v_p, v_d, cls_tokens)
        if self.task != 'cluster':
            score = self.decoder(cls_tokens)
            return score
        else: return cls_tokens


class MolecularGCN(nn.Module):
    def __init__(self, in_feats, dim_embedding=128, hidden_feats=None, activation=None):
        super(MolecularGCN, self).__init__()
        self.init_transform = nn.Linear(in_feats, dim_embedding, bias=False)
        with torch.no_grad():
            self.init_transform.weight[-1].fill_(0)
        self.gnn = GCN(in_feats=dim_embedding, hidden_feats=hidden_feats, activation=activation)
        self.output_feats = hidden_feats[-1]

    def forward(self, batch_graph):
        node_feats = batch_graph.ndata['h']
        node_feats = self.init_transform(node_feats)
        node_feats = self.gnn(batch_graph, node_feats)
        batch_size = batch_graph.batch_size
        node_feats = node_feats.view(batch_size, -1, self.output_feats)
        return node_feats


class ProteinCNN(nn.Module):
    def __init__(self, hid_dim, emb_dim):
        super(ProteinCNN, self).__init__()        
        self.L1 = nn.Linear(hid_dim, emb_dim)
        self.bn1 = nn.BatchNorm1d(emb_dim)


    def forward(self, v):
        v = self.L1(v)
        v = v.transpose(2, 1)
        v = self.bn1(F.relu(v))
        return v.transpose(2, 1)

class Decoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, binary=1):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.bn3 = nn.BatchNorm1d(out_dim)
        self.fc4 = nn.Linear(out_dim, binary)

    def forward(self, x):
        x = self.bn1(F.relu(self.fc1(x)))
        x = self.bn2(F.relu(self.fc2(x)))
        x = self.bn3(F.relu(self.fc3(x)))
        
        x = self.fc4(x)
        return x