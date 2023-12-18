

from udandti import Model, Decoder
from time import time
from utils import set_seed, graph_collate_func, mkdir
from dataloader import DTIDataset, MultiDataLoader
from torch.utils.data import DataLoader
from trainer_in import Trainer_in
from trainer_cross import Trainer_cross
import torch
import argparse
import warnings, os
import pandas as pd
import datetime

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
torch.set_num_threads(16)


parser = argparse.ArgumentParser(description="UdanDTI for DTI prediction")
parser.add_argument('--data', default='bindingdb', type=str, metavar='Dataset', help="dataset", choices=['bindingdb', 'biosnap', 'human'])
parser.add_argument('--split', default='random', type=str, metavar='TASK', help="split task", choices=['random', 'cold', 'cluster', 'unseen_drug', 'unseen_protein'])
parser.add_argument('--res_dir', default='./result/', type=str, metavar='Output', help="save dir")
parser.add_argument('--model', default=None, type=str, metavar='Model', help="use model checkpoints")

parser.add_argument('--lr', default=0.0001, type=float, metavar='lr', help="learning rate")
parser.add_argument('--bs', default=96, type=int, metavar='bs', help="batch size")
parser.add_argument('--epoch', default=100, type=int, metavar='Epoch', help="max epoch")
parser.add_argument('--seed', default=42, type=int, metavar='Seed', help="random seed")

parser.add_argument('--dim', default=128, type=int, metavar='Dim', help="embedding dim")
parser.add_argument('--LLM', default='esm', type=str, metavar='LLM', help="protein large language model", choices=['esm', 'bertbfd'])
args = parser.parse_args()

def in_domain_process(args, ):
    dataFolder = f'./datasets/{args.data}/{args.split}/'

    train_path = os.path.join(dataFolder, 'train.csv')
    val_path = os.path.join(dataFolder, "val.csv")
    test_path = os.path.join(dataFolder, "test.csv")
    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    df_test = pd.read_csv(test_path)

    train_dataset = DTIDataset(df_train.index.values, df_train, args.LLM)
    val_dataset = DTIDataset(df_val.index.values, df_val, args.LLM)
    test_dataset = DTIDataset(df_test.index.values, df_test, args.LLM)
    

    params = {'batch_size': args.bs, 'shuffle': True, 'num_workers': 0,
              'drop_last': True, 'collate_fn': graph_collate_func}


    train_generator = DataLoader(train_dataset, **params)
    params['shuffle'] = False
    params['drop_last'] = False
    
    val_generator = DataLoader(val_dataset, **params)
    test_generator = DataLoader(test_dataset, **params)
    
    model = Model(args.dim, args.LLM, args.split).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    torch.backends.cudnn.benchmark = True

    trainer = Trainer_in(model, opt, device, train_generator, val_generator, test_generator, args)
    
    return trainer

def cross_domain_process(args, ):
    dataFolder = f'./datasets/{args.data}/{args.split}/'

    source_train_path = os.path.join(dataFolder, 'source_train.csv')
    train_path = os.path.join(dataFolder, 'target_train.csv')
    val_path = os.path.join(dataFolder, "target_test.csv")
    test_path = os.path.join(dataFolder, "target_test.csv")
    df_source_train = pd.read_csv(source_train_path)
    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    df_test = pd.read_csv(test_path)

    source_train_dataset = DTIDataset(df_source_train.index.values, df_source_train, args.LLM)
    train_dataset = DTIDataset(df_train.index.values, df_train, args.LLM)
    val_dataset = DTIDataset(df_val.index.values, df_val, args.LLM)
    test_dataset = DTIDataset(df_test.index.values, df_test, args.LLM)
    

    params = {'batch_size': args.bs, 'shuffle': True, 'num_workers': 0,
              'drop_last': True, 'collate_fn': graph_collate_func}


    source_train_generator = DataLoader(source_train_dataset, **params)
    train_generator = DataLoader(train_dataset, **params)
    n_batches = max(len(source_train_generator), len(train_generator))
    multi_generator = MultiDataLoader(dataloaders=[source_train_generator, train_generator], n_batches=n_batches)
    params['shuffle'] = False
    params['drop_last'] = False
    
    val_generator = DataLoader(val_dataset, **params)
    test_generator = DataLoader(test_dataset, **params)
    
    Generator = Model(args.dim, args.LLM, args.split).to(device)
    Classifier1 = Decoder(128, 1024, 256, 1).to(device)
    Classifier2 = Decoder(128, 1024, 256, 1).to(device)
    
    opt_G = torch.optim.Adam(Generator.parameters(), lr=args.lr)
    opt_C1 = torch.optim.Adam(Classifier1.parameters(), lr=args.lr)
    opt_C2 = torch.optim.Adam(Classifier2.parameters(), lr=args.lr)

    torch.backends.cudnn.benchmark = True

    trainer = Trainer_cross([Generator, Classifier1, Classifier2], [opt_G, opt_C1, opt_C2], device, multi_generator, val_generator, test_generator, args)

    return trainer


def main():
    torch.cuda.empty_cache()
    warnings.filterwarnings("ignore", message="invalid value encountered in divide")
    set_seed(args.seed)
    mkdir(args.res_dir)
    
    print(f"Hyperparameters:")
    for arg in vars(args):
        print(format(arg, '<20'), format(str(getattr(args, arg)), '<'))  
    print(f"Running on: {device}", end="\n\n")

    if args.split != 'cluster':
        trainer = in_domain_process(args)
    else: 
        trainer = cross_domain_process(args)
    
    result = trainer.train()

    return result


if __name__ == '__main__':
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    s = time()
    result = main()
    e = time()
    print(f"Total running time: {round(e - s, 2)}s")
