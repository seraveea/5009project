import numpy as np
import pandas as pd
import torch
import collections
import copy
from torch.utils.data import DataLoader
from sklearn.preprocessing import OneHotEncoder
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from utils import *
import datetime
from model import MLP
# loss func use loss_fn = nn.BCELoss()

device = 'cuda:5' if torch.cuda.is_available() else 'cpu'


def average_params(params_list):
    assert isinstance(params_list, (tuple, list, collections.deque))
    n = len(params_list)
    if n == 1:
        return params_list[0]
    new_params = collections.OrderedDict()
    keys = None
    for i, params in enumerate(params_list):
        if keys is None:
            keys = params.keys()
        for k, v in params.items():
            if k not in keys:
                raise ValueError('the %d-th model has different params' % i)
            if k not in new_params:
                new_params[k] = v / n
            else:
                new_params[k] += v / n
    return new_params


def data_creator():
    train = pd.read_csv('./data/train.csv')
    train_x = train[[str(x) for x in range(187)]].values
    train_y = np.array(train[['187']].values, dtype=int)
    # ohe = OneHotEncoder().fit(train_y)
    # train_y = ohe.transform(train_y)
    train_y = np.eye(np.max(train_y) + 1)[train_y]
    # convert into PyTorch tensors
    X = torch.tensor(train_x, dtype=torch.float32)
    y = torch.tensor(train_y, dtype=torch.float32)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True)
    train_loader = DataLoader(list(zip(X_train, y_train)), shuffle=True, batch_size=128)
    test_loader = DataLoader(list(zip(X_test, y_test)), shuffle=True, batch_size=128)
    # for x,y in train_loader:
    #     print(y.squeeze().tolist())
    #     break
    return train_loader, test_loader


def train_epoch(data_loader, model, optimizer, loss_fn):
    global global_step
    for x, y in tqdm(data_loader):
        pre = model(x)
        loss = loss_fn(pre, y.squeeze())
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), 3.)
        optimizer.step()

    model.train()


def test_epoch(data_loader, model, loss_fn):

    model.eval()

    losses = []
    preds = []
    labels = []

    for x, y in tqdm(data_loader):
        with torch.no_grad():
            pre = model(x)
            loss = loss_fn(pre, y.squeeze())
            preds.extend(torch.argmax(pre, dim=1).tolist())
            labels.extend(torch.argmax(y.squeeze(), dim=1).tolist())

        losses.append(loss.item())
    # evaluate
    acc = accuracy(labels, preds)
    maf1 = macro_f1(labels, preds)
    mif1 = micro_f1(labels, preds)
    return np.mean(losses), acc, maf1, mif1


def prediction(data_loader, model):
    model.eval()
    preds = []
    for x in data_loader:
        pred = model(x)
        preds.extend(pred)
    return preds


def main():
    seed = np.random.randint(1000000)
    np.random.seed(seed)
    torch.manual_seed(seed)
    print('create loaders...')
    train_loader, test_loader = data_creator()
    all_precision = []
    all_recall = []
    all_ic = []
    all_rank_ic = []
    print('create model...')
    model = MLP()
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-6)
    best_score = -np.inf
    best_epoch = 0
    stop_round = 0
    params_list = collections.deque(maxlen=5)
    loss_fn = nn.CrossEntropyLoss()
    # save best parameters
    for epoch in range(100):
        print('Running Epoch:', epoch)
        print('training...')
        train_epoch(train_loader, model, optimizer, loss_fn)
        params_ckpt = copy.deepcopy(model.state_dict())
        params_list.append(params_ckpt)
        avg_params = average_params(params_list)
        # when evaluating, use the avg_params in the current time to evaluate
        model.load_state_dict(avg_params)

        print('evaluating...')
        # compute the loss, score, pre, recall, ic, rank_ic on train, valid and test data
        train_loss, train_acc, train_maf1, train_mif1 = test_epoch(train_loader, model, loss_fn)
        val_loss, val_acc, val_maf1, val_mif1 = test_epoch(test_loader, model, loss_fn)
        test_loss, test_acc, test_maf1, test_mif1 = test_epoch(test_loader, model, loss_fn)
        print('train_loss %.6f, valid_loss %.6f, test_loss %.6f' % (train_loss, val_loss, test_loss))
        print('Train acc: ', train_acc, ' Train macro F1: ', train_maf1, ' Train micro F1: ', train_mif1)
        print('Valid acc: ', val_acc, ' Valid macro F1: ', val_maf1, ' Valid micro F1: ', val_mif1)
        print('Test acc: ', train_acc, ' Test macro F1: ', test_maf1, ' Test micro F1: ', test_mif1)


        # load back the current parameters
        model.load_state_dict(params_ckpt)

        if val_acc > best_score:
            # the model performance is increasing
            best_score = val_acc
            stop_round = 0
            best_epoch = epoch
            best_param = copy.deepcopy(avg_params)
        else:
            # the model performance is not increasing
            stop_round += 1
            if stop_round >= 30:
                print('early stop')
                break

        print('best score:', best_score, '@', best_epoch)
        model.load_state_dict(best_param)
        torch.save(best_param, './output/model.bin')

    print('finished.')


if __name__ == '__main__':
    main()

