# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 19:09:45 2021

@author: Ranak Roy Chowdhury
"""
import utils, argparse, warnings, sys, shutil, torch, os, numpy as np, math
warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='LC')
parser.add_argument('--batch', type=int, default=128)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--nlayers', type=int, default=2)
parser.add_argument('--emb_size', type=int, default=64)
parser.add_argument('--nhead', type=int, default=8)
parser.add_argument('--task_rate', type=float, default=0.5)
parser.add_argument('--masking_ratio', type=float, default=0.15)
parser.add_argument('--lamb', type=float, default=0.8)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--ratio_highest_attention', type=float, default=0.5)
parser.add_argument('--avg', type=str, default='macro')
parser.add_argument('--dropout', type=float, default=0.01)
parser.add_argument('--nhid', type=int, default=128)
parser.add_argument('--nhid_task', type=int, default=128)
parser.add_argument('--nhid_tar', type=int, default=128)
parser.add_argument('--task_type', type=str, default='classification', help='[classification, regression]')
args = parser.parse_args()



def main():    
    prop = utils.get_prop(args)
    path = './../transformer/'
    
    print('Data loading start...') #  dataset, data_path, task_type
    X_train, y_train, X_test, y_test = utils.data_loader(args.dataset, path, prop['task_type'])  # CHANGE INPUT
    print('Data loading complete...')

    print('Data preprocessing start...')
    X_train_task, y_train_task, X_test, y_test = utils.preprocess(prop, X_train, y_train, X_test, y_test)
    print(X_train_task.shape, y_train_task.shape, X_test.shape, y_test.shape)
    print('Data preprocessing complete...')

    prop['nclasses'] = torch.max(y_train_task).item() + 1 if prop['task_type'] == 'classification' else None
    prop['dataset'], prop['seq_len'], prop['input_size'] = prop['dataset'], X_train_task.shape[1], X_train_task.shape[2]
    prop['device'] = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    
    print('Initializing model...')
    model, optimizer, criterion_tar, criterion_task, best_model, best_optimizer = utils.initialize_training(prop)
    print('Model intialized...')

    print('Training start...')
    utils.training(model, optimizer, criterion_tar, criterion_task, best_model, best_optimizer, X_train_task, y_train_task, X_test, y_test, prop)
    print('Training complete...')



if __name__ == "__main__":
    main()


'''RAN ON 10 EPOCHS
Data loading start...
Data loading complete...
Data preprocessing start...
torch.Size([512, 100, 2]) torch.Size([448]) torch.Size([128, 100, 2]) torch.Size([112])
Data preprocessing complete...
Initializing model...
Model intialized...
Training start...
Epoch: 1, TAR Loss: 38.300318479537964 , TASK Loss: 1.3012901118823461
Epoch: 2, TAR Loss: 9.31547874212265 , TASK Loss: 0.7666930300848824
Epoch: 3, TAR Loss: 10.423442900180817 , TASK Loss: 0.7214092271668571
Epoch: 4, TAR Loss: 10.836728930473328 , TASK Loss: 0.7055205873080662
Epoch: 5, TAR Loss: 10.618939757347107 , TASK Loss: 0.6788468190601894
Epoch: 6, TAR Loss: 10.077557921409607 , TASK Loss: 0.6665080274854388
Epoch: 7, TAR Loss: 8.614743411540985 , TASK Loss: 0.6227702839033944
Epoch: 8, TAR Loss: 6.918114483356476 , TASK Loss: 0.58811183486666
Epoch: 9, TAR Loss: 6.637998044490814 , TASK Loss: 0.5583974123001099
Epoch: 10, TAR Loss: 6.621802508831024 , TASK Loss: 0.5264310623918261
Dataset: LC, Acc: 0.5178571428571429
Training complete...
'''