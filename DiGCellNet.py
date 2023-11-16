import torch 
import numpy as np
import random
import pandas as pd
import torch.nn as nn 
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix 
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from torch import Tensor 
import torch.optim as optim
import utils_ml
import copy
import models
from collections import Counter
from AutomaticWeightedLoss import AutomaticWeightedLoss

import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import warnings
warnings.filterwarnings('ignore')

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
setup_seed(3407)

parser = utils_ml.get_parser()
args = parser.parse_args()

def get_GCN(nfeat):
    mymodel = models.GCN(nfeat=nfeat,
            embed1=1024,#args.embed1,
            dropout=args.dropout)
    return mymodel

def get_MLP(nfeat, nclass):
    mymodel = models.MLP_1(nfeat = nfeat,
            nclass = nclass)
    return mymodel

def get_MTL1(nfeat):
    mymodel = models.MTL_1(nfeat = nfeat, 
            nhid=1024, 
            dropout=args.dropout)
    return mymodel

def get_nod2MLP(nfeat, nclass):
    mymodel = models.MLP_2(nfeat=nfeat,
            nhid=1024,
            nclass=nclass,
            dropout = args.dropout)
    return mymodel


def get_GCN_MLP(nfeat, nclass):
    mymodel = models.GCN_MLP_1(nfeat = nfeat,
            embed1 = 1024,
            nclass = nclass,
            dropout = args.dropout)
    return mymodel

#KNN model
def classify_knn(train_feature, train_yc, test_feature, test_yc, n_neighbors):
    knn = KNeighborsClassifier(n_neighbors)
    knn.fit(train_feature, train_yc)
    prediction_yc = knn.predict_proba(test_feature)
    prediction_yc = torch.FloatTensor(prediction_yc)
    loss_test = F.cross_entropy(prediction_yc, test_yc)
    acc_test = utils_ml.accuracy(prediction_yc, test_yc)
    auc_test, fpr_test, tpr_test = utils_ml.roc_auc(prediction_yc, test_yc)
    f1_test = utils_ml.f1(prediction_yc, test_yc)
    aupr_test  = utils_ml.aupr(prediction_yc, test_yc)

    print('KNN Test...\t'
        'Loss: %.4f\t'
        'acc: %.4f\t'
        'auc: %.4f\t'
        'aupr: %.4f\t'
        'f1: %.4f\t' % (
            loss_test, acc_test, auc_test, aupr_test, f1_test))

    prob_test = prediction_yc
    prob_test = prob_test.numpy()
    prob_test = prob_test - np.max(prob_test,axis=1).reshape((-1,1))
    prob_test = np.exp(prob_test)
    prob_test = prob_test/np.sum(prob_test,axis=1).reshape((-1,1))

    yc_pred = np.argmax(prob_test, axis=1)
    yc_true = test_yc
    print('yc_pred untrain: ', yc_pred[:20])
    print('yc_true untrain: ', yc_true[:20])
    y = np.array(yc_true.cpu())
    cmcv = confusion_matrix(y, yc_pred)
    print('confu_matrix:\n ', cmcv)
    print('wrong/right: ',cmcv[0][1]+cmcv[1][0], cmcv[0][0]+cmcv[1][1])

    return loss_test, acc_test, auc_test, aupr_test, f1_test, cmcv


def classifier_SVM(train_feature, train_yc, test_feature, test_yc):
    clf = svm.SVC(C=100,kernel='rbf',gamma=0.01,class_weight='balanced',probability=True)
    clf.fit(train_feature, train_yc)
    prediction_yc = clf.predict_proba(test_feature)
    prediction_yc = torch.FloatTensor(prediction_yc)
    loss_test = F.cross_entropy(prediction_yc, test_yc)
    acc_test = utils_ml.accuracy(prediction_yc, test_yc)
    auc_test, fpr_test, tpr_test = utils_ml.roc_auc(prediction_yc, test_yc)
    f1_test = utils_ml.f1(prediction_yc, test_yc)
    aupr_test  = utils_ml.aupr(prediction_yc, test_yc)

    print('SVM Test...\t'
        'Loss: %.4f\t'
        'acc: %.4f\t'
        'auc: %.4f\t'
        'aupr: %.4f\t'
        'f1: %.4f\t' % (
            loss_test, acc_test, auc_test, aupr_test, f1_test))

    prob_test = prediction_yc
    prob_test = prob_test.numpy()
    prob_test = prob_test - np.max(prob_test,axis=1).reshape((-1,1))
    prob_test = np.exp(prob_test)
    prob_test = prob_test/np.sum(prob_test,axis=1).reshape((-1,1))

    yc_pred = np.argmax(prob_test, axis=1)
    yc_true = test_yc
    print('yc_pred untrain: ', yc_pred[:20])
    print('yc_true untrain: ', yc_true[:20])
    y = np.array(yc_true.cpu())
    cmcv = confusion_matrix(y, yc_pred)
    print('confu_matrix:\n ', cmcv)
    print('wrong/right: ',cmcv[0][1]+cmcv[1][0], cmcv[0][0]+cmcv[1][1])

    return loss_test, acc_test, auc_test, aupr_test, f1_test, cmcv


def train_GCNMLP(model_gcn_mlp, idx_train, idx_val, idx_test, features, label_c, adj):
    optimizer = optim.Adam(model_gcn_mlp.parameters(),
                    lr=args.lr, weight_decay=args.weight_decay)

    max_f1 = 0
    patience = 0

    for epoch in range(args.epochs):
        model_gcn_mlp.train()
        out = model_gcn_mlp(features, adj)

        # Multiply the loss of each sample by the penalty weight weight
        loss = F.cross_entropy(out[idx_train], label_c[idx_train])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print('Train...\t'
            'Epoch: [%d/%d]\t'
            'loss: %.4f\t'% (
                epoch + 1, args.epochs, loss))

        #val
        model_gcn_mlp.eval()
        with torch.no_grad():
            out_val = model_gcn_mlp(features, adj)

        loss_v = F.cross_entropy(out_val[idx_val], label_c[idx_val])

        auc_val, fpr_val, tpr_val = utils_ml.roc_auc(out_val[idx_val], label_c[idx_val])
        f1_val = utils_ml.f1(out_val[idx_val], label_c[idx_val])

        if f1_val > max_f1:
            max_f1 = f1_val
            patience = 0
            best_param_gcn_mlp = copy.deepcopy(model_gcn_mlp.state_dict())
            print('Valida...\t'
                'Loss val: %.4f\t'
                'auc val: %.4f\t'
                'f1 val: %.4f\t' % (
                    #epoch + 1, args.epochs, 
                    loss_v, auc_val, f1_val))
                    
        else:
            patience += 1

        if patience>20:
            model_gcn_mlp.load_state_dict(best_param_gcn_mlp)
            model_gcn_mlp.eval()
            with torch.no_grad():
                out_test = model_gcn_mlp(features, adj)

            loss_test = F.cross_entropy(out_test[idx_test], label_c[idx_test])
            acc_test = utils_ml.accuracy(out_test[idx_test], label_c[idx_test])
            auc_test, fpr_test, tpr_test = utils_ml.roc_auc(out_test[idx_test], label_c[idx_test])
            f1_test = utils_ml.f1(out_test[idx_test], label_c[idx_test])
            aupr_test  = utils_ml.aupr(out_test[idx_test], label_c[idx_test])
            print('Base Test...\t'
                'Loss: %.4f\t'
                'acc: %.4f\t'
                'auc: %.4f\t'
                'aupr: %.4f\t'
                'f1: %.4f\t' % (
                    loss_test, acc_test, auc_test, aupr_test, f1_test))

            prob_test = out_test[idx_test]
            prob_test = prob_test.detach().cpu().numpy()
            prob_test = prob_test - np.max(prob_test,axis=1).reshape((-1,1))
            prob_test = np.exp(prob_test)
            prob_test = prob_test/np.sum(prob_test,axis=1).reshape((-1,1))

            yc_pred = np.argmax(prob_test, axis=1)
            yc_true = label_c[idx_test]
            print('yc_pred untrain: ', yc_pred[:20])
            print('yc_true untrain: ', yc_true[:20])
            y = np.array(yc_true.cpu())
            cmcv = confusion_matrix(y, yc_pred)
            print('confu_matrix:\n ', cmcv)
            print('wrong/right: ',cmcv[0][1]+cmcv[1][0], cmcv[0][0]+cmcv[1][1])

            return loss_test, acc_test, auc_test, aupr_test, f1_test, cmcv


def train_GCNMTL(model_gcn, model_mlp_d, model_mlp_c, awl, features, adj, idx_train, idx_val, idx_test, label_d, label_c):
    optimizer = optim.Adam([
                {'params': model_gcn.parameters(), 'lr':args.lr, 'weight_decay':args.weight_decay},
                {'params': model_mlp_d.parameters(), 'lr':args.lr_d, 'weight_decay':args.weight_decay},
                {'params': model_mlp_c.parameters(), 'lr':args.lr, 'weight_decay':args.weight_decay},
                {'params': awl.parameters(), 'lr':args.lr, 'weight_decay':args.weight_decay}
            ])

    max_f1 = 0
    patience = 0

    for epoch in range(args.epochs):
        model_gcn.train()
        model_mlp_d.train()
        model_mlp_c.train()
        out_gcn = model_gcn(features, adj)
        out_d = model_mlp_d(out_gcn)
        out_c = model_mlp_c(out_gcn)

        # Multiply the loss of each sample by the penalty weight weight
        loss_d = F.cross_entropy(out_d[idx_train], label_d[idx_train])
        loss_c = F.cross_entropy(out_c[idx_train], label_c[idx_train])
        loss = awl(loss_d, loss_c)
        # loss = 0.8*loss_d + 0.2*loss_c

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print('Train...\t'
            'Epoch: [%d/%d]\t'
            'loss: %.4f\t'
            'lossd: %.4f\t'
            'lossc: %.4f\t'% (
                epoch + 1, args.epochs, loss, loss_d, loss_c))

        #val
        model_gcn.eval()
        model_mlp_d.eval()
        model_mlp_c.eval()

        with torch.no_grad():
            out_val_gcn = model_gcn(features, adj)
            out_val_d = model_mlp_d(out_val_gcn)
            out_val_c = model_mlp_c(out_val_gcn)

        loss_vd = F.cross_entropy(out_val_d[idx_val], label_d[idx_val])
        loss_vc = F.cross_entropy(out_val_c[idx_val], label_c[idx_val])
        loss_v = awl(loss_vd, loss_vc)
        # loss_v = 0.8*loss_vd + 0.2*loss_vc

        auc_val_d, fpr_val_d, tpr_val_d = utils_ml.roc_auc(out_val_d[idx_val], label_d[idx_val])
        auc_val_c, fpr_val_c, tpr_val_c = utils_ml.roc_auc(out_val_c[idx_val], label_c[idx_val])
        f1_val_d = utils_ml.f1(out_val_d[idx_val], label_d[idx_val])
        f1_val_c = utils_ml.f1(out_val_c[idx_val], label_c[idx_val])

        if f1_val_c > max_f1:
            patience = 0
            max_f1 = f1_val_c
            best_param_gcn = copy.deepcopy(model_gcn.state_dict())
            best_param_mlpd = copy.deepcopy(model_mlp_d.state_dict())
            best_param_mlpc = copy.deepcopy(model_mlp_c.state_dict())
            best_param_awl = copy.deepcopy(awl.state_dict())

            print('Valida...\t'
                'Loss val: %.4f\t'
                'Lossvd: %.4f\t'
                'Lossvc: %.4f\t'
                'aucd val: %.4f\t'
                'f1d val: %.4f\t'
                'aucc val: %.4f\t'
                'f1c val: %.4f\t' % (
                    loss_v, loss_vd, loss_vc, auc_val_d, f1_val_d, auc_val_c, f1_val_c))
                    
        else:
            patience += 1

        if patience>20:
            print('***early stop***')
            model_gcn.load_state_dict(best_param_gcn)
            model_mlp_d.load_state_dict(best_param_mlpd)
            model_mlp_c.load_state_dict(best_param_mlpc)
            awl.load_state_dict(best_param_awl)

            model_gcn.eval()
            model_mlp_d.eval()
            model_mlp_c.eval()

            with torch.no_grad():
                out_test_gcn = model_gcn(features, adj)
                out_test_d = model_mlp_d(out_test_gcn)
                out_test_c = model_mlp_c(out_test_gcn)

            lossc_test = F.cross_entropy(out_test_c[idx_test], label_c[idx_test])
            lossd_test = F.cross_entropy(out_test_d[idx_test], label_d[idx_test])
            loss_test = awl(lossd_test, lossc_test)
            # loss_test = 0.8*lossd_test+0.2*lossc_test

            acc_test = utils_ml.accuracy(out_test_c[idx_test], label_c[idx_test])
            auc_test, fpr_test, tpr_test = utils_ml.roc_auc(out_test_c[idx_test], label_c[idx_test])
            f1_test = utils_ml.f1(out_test_c[idx_test], label_c[idx_test])
            aupr_test  = utils_ml.aupr(out_test_c[idx_test], label_c[idx_test])
            print('Base Test...\t'
                'Loss: %.4f\t'
                'acc: %.4f\t'
                'auc: %.4f\t'
                'aupr: %.4f\t'
                'f1: %.4f\t' % (
                    loss_test, acc_test, auc_test, aupr_test, f1_test))

            prob_test = out_test_c[idx_test]
            prob_test = prob_test.detach().cpu().numpy()
            prob_test = prob_test - np.max(prob_test,axis=1).reshape((-1,1))
            prob_test = np.exp(prob_test)
            prob_test = prob_test/np.sum(prob_test,axis=1).reshape((-1,1))

            yc_pred = np.argmax(prob_test, axis=1)
            yc_true = label_c[idx_test]
            print('yc_pred test: ', yc_pred[:20])
            print('yc_true test: ', yc_true[:20])
            y = np.array(yc_true.cpu())
            cmcv = confusion_matrix(y, yc_pred)
            print('confu_matrix:\n ', cmcv)
            print('wrong/right: ',cmcv[0][1]+cmcv[1][0], cmcv[0][0]+cmcv[1][1])

            return loss_test, acc_test, auc_test, aupr_test, f1_test, cmcv, out_test_c


def train_nod2MLP(mymodel, train_x, train_y, val_x, val_y, test_x, test_y):
    optimizer = optim.Adam(mymodel.parameters(),
                        lr=args.lr, weight_decay=args.weight_decay)

    max_f1 = 0
    patience = 0

    for epoch in range(args.epochs):
        mymodel.train()
        output = mymodel(train_x)        
        loss = F.cross_entropy(output, train_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print('nonod2ve_Train...\t'
            'Epoch: [%d/%d]\t'
            'loss: %.4f\t'% (
                epoch + 1, args.epochs, loss))

        # eval
        mymodel.eval()
        with torch.no_grad():
            output_v = mymodel(val_x)
        loss_val = F.cross_entropy(output_v, val_y)
        auc_val, fpr_val, tpr_val = utils_ml.roc_auc(output_v, val_y)
        f1_val = utils_ml.f1(output_v, val_y)

        if f1_val > max_f1:
            max_f1 = f1_val
            patience = 0
            best_param_mymodel = copy.deepcopy(mymodel.state_dict())
            print('Valida...\t'
                'Loss val: %.4f\t'
                'auc val: %.4f\t'
                'f1 val: %.4f\t' % (
                    loss_val, auc_val, f1_val))
                    
        else:
            patience += 1

        if patience>20:
            mymodel.load_state_dict(best_param_mymodel)
            mymodel.eval()
            with torch.no_grad():
                out_test = mymodel(test_x)

            loss_test = F.cross_entropy(out_test, test_y)
            acc_test = utils_ml.accuracy(out_test, test_y)
            auc_test, fpr_test, tpr_test = utils_ml.roc_auc(out_test, test_y)
            f1_test = utils_ml.f1(out_test, test_y)
            aupr_test  = utils_ml.aupr(out_test, test_y)
            print('Base Test...\t'
                'Loss: %.4f\t'
                'acc: %.4f\t'
                'auc: %.4f\t'
                'aupr: %.4f\t'
                'f1: %.4f\t' % (
                    loss_test, acc_test, auc_test, aupr_test, f1_test))

            prob_test = out_test.detach().cpu().numpy()
            prob_test = prob_test - np.max(prob_test,axis=1).reshape((-1,1))
            prob_test = np.exp(prob_test)
            prob_test = prob_test/np.sum(prob_test,axis=1).reshape((-1,1))

            yc_pred = np.argmax(prob_test, axis=1)
            yc_true = test_y
            print('yc_pred untrain: ', yc_pred[:20])
            print('yc_true untrain: ', yc_true[:20])
            y = np.array(yc_true.cpu())
            cmcv = confusion_matrix(y, yc_pred)
            print('confu_matrix:\n ', cmcv)
            print('wrong/right: ',cmcv[0][1]+cmcv[1][0], cmcv[0][0]+cmcv[1][1])

            return loss_test, acc_test, auc_test, aupr_test, f1_test, cmcv



def train_nod2MTL(model_mtl, model_mlp_d, model_mlp_c, awl, x_train, yc_train, yd_train, x_val, yc_val, yd_val, x_test, yc_test, yd_test):
    optimizer = optim.Adam([
                {'params': model_mtl.parameters(), 'lr':args.lr, 'weight_decay':args.weight_decay},
                {'params': model_mlp_d.parameters(), 'lr':args.lr_d, 'weight_decay':args.weight_decay},
                {'params': model_mlp_c.parameters(), 'lr':args.lr, 'weight_decay':args.weight_decay},
                {'params': awl.parameters(), 'lr':args.lr, 'weight_decay':args.weight_decay}
            ])

    max_f1 = 0
    patience = 0

    for epoch in range(args.epochs):
        model_mtl.train()
        model_mlp_d.train()
        model_mlp_c.train()
        out_mtl = model_mtl(x_train)
        out_d = model_mlp_d(out_mtl)
        out_c = model_mlp_c(out_mtl)

        loss_d = F.cross_entropy(out_d, yd_train)
        loss_c = F.cross_entropy(out_c, yc_train)
        loss = awl(loss_d, loss_c)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print('nonod2ve_Train...\t'
            'Epoch: [%d/%d]\t'
            'loss: %.4f\t'
            'lossd: %.4f\t'
            'lossc: %.4f\t'% (
                epoch + 1, args.epochs, loss, loss_d, loss_c))

        # eval
        model_mtl.eval()
        model_mlp_d.eval()
        model_mlp_c.eval()

        with torch.no_grad():
            out_val_mtl = model_mtl(x_val)
            out_val_d = model_mlp_d(out_val_mtl)
            out_val_c = model_mlp_c(out_val_mtl)

        loss_vd = F.cross_entropy(out_val_d, yd_val)
        loss_vc = F.cross_entropy(out_val_c, yc_val)
        loss_v = awl(loss_vd, loss_vc)


        auc_val_d, fpr_val_d, tpr_val_d = utils_ml.roc_auc(out_val_d, yd_val)
        auc_val_c, fpr_val_c, tpr_val_c = utils_ml.roc_auc(out_val_c, yc_val)
        f1_val_d = utils_ml.f1(out_val_d, yd_val)
        f1_val_c = utils_ml.f1(out_val_c, yc_val)

        if f1_val_c > max_f1:
            max_f1 = f1_val_c
            patience = 0
            best_param_mtl = copy.deepcopy(model_mtl.state_dict())
            best_param_mlpd = copy.deepcopy(model_mlp_d.state_dict())
            best_param_mlpc = copy.deepcopy(model_mlp_c.state_dict())

            print('Valida...\t'
                'Loss val: %.4f\t'
                'Lossvd: %.4f\t'
                'Lossvc: %.4f\t'
                'aucd val: %.4f\t'
                'f1d val: %.4f\t'
                'aucc val: %.4f\t'
                'f1c val: %.4f\t' % (
                    loss_v, loss_vd, loss_vc, auc_val_d, f1_val_d, auc_val_c, f1_val_c))
                    
        else:
            patience += 1

        if patience>20:
            model_mtl.load_state_dict(best_param_mtl)
            model_mlp_d.load_state_dict(best_param_mlpd)
            model_mlp_c.load_state_dict(best_param_mlpc)

            model_mtl.eval()
            model_mlp_d.eval()
            model_mlp_c.eval()

            with torch.no_grad():
                out_test_mtl = model_mtl(x_test)
                out_test_d = model_mlp_d(out_test_mtl)
                out_test_c = model_mlp_c(out_test_mtl)

            lossc_test = F.cross_entropy(out_test_c, yc_test)
            lossd_test = F.cross_entropy(out_test_d, yd_test)
            loss_test = awl(lossd_test, lossc_test)


            acc_test = utils_ml.accuracy(out_test_c, yc_test)
            auc_test, fpr_test, tpr_test = utils_ml.roc_auc(out_test_c, yc_test)
            f1_test = utils_ml.f1(out_test_c, yc_test)
            aupr_test  = utils_ml.aupr(out_test_c, yc_test)
            print('Base Test...\t'
                'Loss: %.4f\t'
                'acc: %.4f\t'
                'auc: %.4f\t'
                'aupr: %.4f\t'
                'f1: %.4f\t' % (
                    loss_test, acc_test, auc_test, aupr_test, f1_test))

            prob_test = out_test_c.detach().cpu().numpy()
            prob_test = prob_test - np.max(prob_test,axis=1).reshape((-1,1))
            prob_test = np.exp(prob_test)
            prob_test = prob_test/np.sum(prob_test,axis=1).reshape((-1,1))

            yc_pred = np.argmax(prob_test, axis=1)
            yc_true = yc_test
            print('yc_pred untrain: ', yc_pred[:20])
            print('yc_true untrain: ', yc_true[:20])
            y = np.array(yc_true.cpu())
            cmcv = confusion_matrix(y, yc_pred)
            print('confu_matrix:\n ', cmcv)
            print('wrong/right: ',cmcv[0][1]+cmcv[1][0], cmcv[0][0]+cmcv[1][1])

            return loss_test, acc_test, auc_test, aupr_test, f1_test, cmcv


def main(): 
    acc_nod2MLP_list, auc_nod2MLP_list, aupr_nod2MLP_list, f1_nod2MLP_list = [],[],[],[]
    acc_nod2MTL_list, auc_nod2MTL_list, aupr_nod2MTL_list, f1_nod2MTL_list = [],[],[],[]
    acc_knn_list, auc_knn_list, aupr_knn_list, f1_knn_list = [],[],[],[]
    acc_svm_list, auc_svm_list, aupr_svm_list, f1_svm_list = [],[],[],[]
    acc_GCNMLP_list, aupr_GCNMLP_list, aupr_GCNMLP_list, F1_GCNMLP_list = [],[],[],[]
    acc_GCNMTL_list, auc_GCNMTL_list, aupr_GCNMTL_list, f1_GCNMTL_list = [],[],[],[]
    
    num00_test_nod2MLP, num01_test_nod2MLP, num10_test_nod2MLP, num11_test_nod2MLP = [],[],[],[]
    num00_test_nod2MTL, num01_test_nod2MTL, num10_test_nod2MTL, num11_test_nod2MTL = [],[],[],[]
    num00_test_knn, num01_test_knn, num10_test_knn, num11_test_knn = [],[],[],[]
    num00_test_svm, num01_test_svm, num10_test_svm, num11_test_svm = [],[],[],[]
    num00_test_GCNMLP, num01_test_GCNMLP, num10_test_GCNMLP, num11_test_GCNMLP = [],[],[],[]
    num00_test_GCNMTL, num01_test_GCNMTL, num10_test_GCNMTL, num11_test_GCNMTL = [],[],[],[]

    gene_nod2vec, feature_nod2vec, yd_nod2vec, yc_nod2vec = utils_ml.load_data_nod2vec('data_out/feature_with_label_nod2vec_Beta.txt')
    df, nodes_common, load_label = utils_ml.load_data()

    max_mt_f1 = 0

    for i in range(5):
        print(i)
        nodes_common1 = copy.deepcopy(nodes_common)
        nodes_df, features, adj, label_d, label_c, idx_train_order, idx_train, idx_val, idx_test  = utils_ml.split_train_val_test(df, nodes_common1, load_label)       

        features = features.cuda()
        adj = adj.cuda()
        label_d = label_d.cuda()
        label_c = label_c.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()

        print('________________* GCNMLP *_______________')
        model_gcn_mlp = get_GCN_MLP(features.shape[1], nclass=2)
        model_gcn_mlp.cuda()
        loss_GCNMLP, acc_GCNMLP, auc_GCNMLP, aupr_GCNMLP, f1_GCNMLP, confumatrix_GCNMLP = train_GCNMLP(model_gcn_mlp, idx_train, idx_val, idx_test, features, label_c, adj)
        num00_test_GCNMLP.append(confumatrix_GCNMLP[0][0])
        num01_test_GCNMLP.append(confumatrix_GCNMLP[0][1])
        num10_test_GCNMLP.append(confumatrix_GCNMLP[1][0])
        num11_test_GCNMLP.append(confumatrix_GCNMLP[1][1])

        acc_GCNMLP_list.append(acc_GCNMLP)
        aupr_GCNMLP_list.append(auc_GCNMLP)
        aupr_GCNMLP_list.append(aupr_GCNMLP)
        F1_GCNMLP_list.append(f1_GCNMLP)

        print('________________* GCNMTL *_______________')
        model_gcn = get_GCN(features.shape[1])
        model_mlp_d = get_MLP(1024, 2)
        model_mlp_c = get_MLP(1024, 2)
        model_gcn.cuda()
        model_mlp_d.cuda()
        model_mlp_c.cuda()

        awl = AutomaticWeightedLoss(2)
        awl = awl.cuda()

        loss_GCNMTL, acc_GCNMTL, auc_GCNMTL, aupr_GCNMTL, f1_GCNMTL, confumatrix_GCNMTL, pred_prob_c = train_GCNMTL(model_gcn, model_mlp_d, model_mlp_c, awl, features, adj, idx_train, idx_val, idx_test, label_d, label_c)

        '''Save the genes in the test set whose true label is 0 and whose predicted value is 1 in the file'''
        '''Arrange all genes in the test set according to cell type-specific probability from large to small and output it to a file'''
        if f1_GCNMTL >= max_mt_f1:
            max_mt_f1 = f1_GCNMTL

            # try to find genes which label_c=0 & pred_c=1
            # sort genes in test dataset accroding to pred_prob
            
            verify_genes = []
            verify_prob_genes = []
            for idx in idx_test:
                pred_c = torch.argmax(pred_prob_c, axis=1)
                if label_c[idx]==0 and pred_c[idx]==1:
                    verify_genes.append(nodes_df[idx])
            col1 = "gene"
            data_gene_01_test = pd.DataFrame({col1:verify_genes})
            data_gene_01_test.to_excel('data_out/verify01_beta.xlsx',sheet_name='sheet1', index=False)
            print('*****finish save file 1********')

            m=nn.Softmax(dim=1)
            pred_prob_c = m(pred_prob_c)
            pred_prob_c_test = pred_prob_c[idx_test][:, 1].tolist()
            dict_prob_idx = dict(zip(pred_prob_c_test, idx_test))
            dict_prob_idx = sorted(dict_prob_idx.items(),reverse=True)
            sorted_prob_test = []
            sorted_idx_test = [] # Sort idx_test according to prob_c to get sorted_idx_test
            true_label_c = []
            pred_label_c = []
            label_c_cpu = label_c.clone().cpu()
            pred_c_cpu = pred_c.clone().cpu()
            for i in dict_prob_idx:
                sorted_prob_test.append(i[0])
                sorted_idx_test.append(i[1])
            for idx in sorted_idx_test:
                verify_prob_genes.append(nodes_df[idx])
                true_label_c.append(label_c_cpu[idx].item())
                pred_label_c.append(pred_c_cpu[idx].item())

            col2 = "prob"
            col3 = "true_label_c"
            col4 = "pred_label_c"
            data_gene_prob_test = pd.DataFrame({col1:verify_prob_genes,col2:sorted_prob_test,col3:true_label_c,col4:pred_label_c})
            data_gene_prob_test.to_excel('data_out/verify_beta_sortprob_true_pred.xlsx', sheet_name='sheet1', index=False)
            print('*****finish save file 2********')

        num00_test_GCNMTL.append(confumatrix_GCNMTL[0][0])
        num01_test_GCNMTL.append(confumatrix_GCNMTL[0][1])
        num10_test_GCNMTL.append(confumatrix_GCNMTL[1][0])
        num11_test_GCNMTL.append(confumatrix_GCNMTL[1][1])

        acc_GCNMTL_list.append(acc_GCNMTL)
        auc_GCNMTL_list.append(auc_GCNMTL)
        aupr_GCNMTL_list.append(aupr_GCNMTL)
        f1_GCNMTL_list.append(f1_GCNMTL)


        # prepare data for KNN/MLP/SVM... use node2vec to get features from PPI
        gene_train = []
        gene_test = []
        gene_val = []
        for idx in idx_train:
            gene_train.append(nodes_df[idx])
        for idx in idx_test:
            gene_test.append(nodes_df[idx])
        for idx in idx_val:
            gene_val.append(nodes_df[idx])

        feature_train = []
        feature_test = []
        feature_val = []
        labelc_train = []
        labelc_test = []
        labelc_val = []
        labeld_train = []
        labeld_test = []
        labeld_val = []

        for i in range(len(gene_nod2vec)):
            if gene_nod2vec[i] in gene_train:
                feature_train.append(feature_nod2vec[i])
                labelc_train.append(yc_nod2vec[i])
                labeld_train.append(yd_nod2vec[i])
            elif gene_nod2vec[i] in gene_test:
                feature_test.append(feature_nod2vec[i])
                labelc_test.append(yc_nod2vec[i])
                labeld_test.append(yd_nod2vec[i])
            elif gene_nod2vec[i] in gene_val:
                feature_val.append(feature_nod2vec[i])
                labelc_val.append(yc_nod2vec[i])
                labeld_val.append(yd_nod2vec[i])


        feature_train = torch.FloatTensor(feature_train)
        labelc_train = torch.LongTensor(labelc_train)
        labeld_train = torch.LongTensor(labeld_train)
        feature_val = torch.FloatTensor(feature_val)
        labelc_val = torch.LongTensor(labelc_val)
        labeld_val = torch.LongTensor(labeld_val)
        feature_test = torch.FloatTensor(feature_test)
        labelc_test = torch.LongTensor(labelc_test)
        labeld_test = torch.LongTensor(labeld_test)


        print('________________* KNN *_______________')
        loss_knn, acc_knn, auc_knn, aupr_knn, f1_knn, confumatrix_knn = classify_knn(feature_train, labelc_train, feature_test, labelc_test, 3)
        num00_test_knn.append(confumatrix_knn[0][0])
        num01_test_knn.append(confumatrix_knn[0][1])
        num10_test_knn.append(confumatrix_knn[1][0])
        num11_test_knn.append(confumatrix_knn[1][1])

        acc_knn_list.append(acc_knn)
        auc_knn_list.append(auc_knn)
        aupr_knn_list.append(aupr_knn)
        f1_knn_list.append(f1_knn)


        print('________________* SVM *_______________')
        loss_svm, acc_svm, auc_svm, aupr_svm, f1_svm, confumatrix_svm = classifier_SVM(feature_train, labelc_train, feature_test, labelc_test)
        num00_test_svm.append(confumatrix_svm[0][0])
        num01_test_svm.append(confumatrix_svm[0][1])
        num10_test_svm.append(confumatrix_svm[1][0])
        num11_test_svm.append(confumatrix_svm[1][1])

        acc_svm_list.append(acc_svm)
        auc_svm_list.append(auc_svm)
        aupr_svm_list.append(aupr_svm)
        f1_svm_list.append(f1_svm)


        print('________________* nod2MLP *_______________')
        feature_train = feature_train.cuda()
        labelc_train = labelc_train.cuda()
        labeld_train = labeld_train.cuda()
        feature_val = feature_val.cuda()
        labelc_val = labelc_val.cuda()
        labeld_val = labeld_val.cuda()
        feature_test = feature_test.cuda()
        labelc_test = labelc_test.cuda()
        labeld_test = labeld_test.cuda()


        mymodel= get_nod2MLP(feature_train.shape[1], nclass=2)
        mymodel = mymodel.cuda()
        loss_nod2MLP, acc_nod2MLP, auc_nod2MLP, aupr_nod2MLP, f1_nod2MLP, confumatrix_nod2MLP = train_nod2MLP(mymodel, feature_train, labelc_train, feature_val, labelc_val, feature_test, labelc_test)

        num00_test_nod2MLP.append(confumatrix_nod2MLP[0][0])
        num01_test_nod2MLP.append(confumatrix_nod2MLP[0][1])
        num10_test_nod2MLP.append(confumatrix_nod2MLP[1][0])
        num11_test_nod2MLP.append(confumatrix_nod2MLP[1][1])

        acc_nod2MLP_list.append(acc_nod2MLP)
        auc_nod2MLP_list.append(auc_nod2MLP)
        aupr_nod2MLP_list.append(aupr_nod2MLP)
        f1_nod2MLP_list.append(f1_nod2MLP)


        print('________________* nod2MTL *_______________')
        model_mtl1 = get_MTL1(feature_train.shape[1])
        model_mlp2_d = get_MLP(1024, 2)
        model_mlp2_c = get_MLP(1024, 2)
        model_mtl1.cuda()
        model_mlp2_d.cuda()
        model_mlp2_c.cuda()

        awl = AutomaticWeightedLoss(2)
        awl = awl.cuda()

        loss_nod2MTL, acc_nod2MTL, auc_nod2MTL, aupr_nod2MTL, f1_nod2MTL, confumatrix_nod2MTL = train_nod2MTL(model_mtl1, model_mlp2_d, model_mlp2_c, awl, feature_train, labelc_train, labeld_train, feature_val, labelc_val, labeld_val, feature_test, labelc_test, labeld_test)

        num00_test_nod2MTL.append(confumatrix_nod2MTL[0][0])
        num01_test_nod2MTL.append(confumatrix_nod2MTL[0][1])
        num10_test_nod2MTL.append(confumatrix_nod2MTL[1][0])
        num11_test_nod2MTL.append(confumatrix_nod2MTL[1][1])

        acc_nod2MTL_list.append(acc_nod2MTL)
        auc_nod2MTL_list.append(auc_nod2MTL)
        aupr_nod2MTL_list.append(aupr_nod2MTL)
        f1_nod2MTL_list.append(f1_nod2MTL)


        print('^^^^^^^^^^^^^* KNN test *^^^^^^^^^^^^^')
        print('confu_matrix:\n',confumatrix_knn)
        print('acc: %.4f\t'
            'auc: %.4f\t'
            'aupr: %.4f\t'
            'f1: %.4f\t' % (
                acc_knn, auc_knn, aupr_knn, f1_knn))


        print('^^^^^^^^^^^^^* SVM test *^^^^^^^^^^^^^')
        print('confu_matrix:\n',confumatrix_svm)
        print('acc: %.4f\t'
            'auc: %.4f\t'
            'aupr: %.4f\t'
            'f1: %.4f\t' % (
                acc_svm, auc_svm, aupr_svm, f1_svm))

        print('^^^^^^^^^^^^^* nod2MLP test *^^^^^^^^^^^^^')
        print('confu_matrix:\n',confumatrix_nod2MLP)
        print('acc: %.4f\t'
            'auc: %.4f\t'
            'aupr: %.4f\t'
            'f1: %.4f\t' % (
                acc_nod2MLP, auc_nod2MLP, aupr_nod2MLP, f1_nod2MLP))

        print('^^^^^^^^^^^^^* nod2MTL test *^^^^^^^^^^^^^')
        print('confu_matrix:\n',confumatrix_nod2MTL)
        print('acc: %.4f\t'
            'auc: %.4f\t'
            'aupr: %.4f\t'
            'f1: %.4f\t' % (
                acc_nod2MTL, auc_nod2MTL, aupr_nod2MTL, f1_nod2MTL))
        
        print('^^^^^^^^^^^^^* GCNMLP test *^^^^^^^^^^^^^')
        print('confu_matrix:\n',confumatrix_GCNMLP)
        print('acc: %.4f\t'
            'auc: %.4f\t'
            'aupr: %.4f\t'
            'f1: %.4f\t' % (
                acc_GCNMLP, auc_GCNMLP, aupr_GCNMLP, f1_GCNMLP))
        
        print('^^^^^^^^^^^^^* GCNMTL test *^^^^^^^^^^^^^')
        print('confu_matrix:\n',confumatrix_GCNMTL)
        print('acc: %.4f\t'
            'auc: %.4f\t'
            'aupr: %.4f\t'
            'f1: %.4f\t' % (
                acc_GCNMTL, auc_GCNMTL, aupr_GCNMTL, f1_GCNMTL))


    print("*mean KNN* acc = {:.4f}".format(np.mean(acc_knn_list)),
        "auc = {:.4f}".format(np.mean(auc_knn_list)),
        "aupr = {:.4f}".format(np.mean(aupr_knn_list)),
        "f_score = {:.4f}".format(np.mean(f1_knn_list)))


    print("*mean SVM* acc = {:.4f}".format(np.mean(acc_svm_list)),
        "auc = {:.4f}".format(np.mean(auc_svm_list)),
        "aupr = {:.4f}".format(np.mean(aupr_svm_list)),
        "f_score = {:.4f}".format(np.mean(f1_svm_list)))


    print("*mean nod2MLP* acc = {:.4f}".format(np.mean(acc_nod2MLP_list)),
        "auc = {:.4f}".format(np.mean(auc_nod2MLP_list)),
        "aupr = {:.4f}".format(np.mean(aupr_nod2MLP_list)),
        "f_score = {:.4f}".format(np.mean(f1_nod2MLP_list)))

    print("*mean nod2MTL* acc = {:.4f}".format(np.mean(acc_nod2MTL_list)),
        "auc = {:.4f}".format(np.mean(auc_nod2MTL_list)),
        "aupr = {:.4f}".format(np.mean(aupr_nod2MTL_list)),
        "f_score = {:.4f}".format(np.mean(f1_nod2MTL_list)))

    print("*mean GCNMLP* acc = {:.4f}".format(np.mean(acc_GCNMLP_list)),
        "auc = {:.4f}".format(np.mean(aupr_GCNMLP_list)),
        "aupr = {:.4f}".format(np.mean(aupr_GCNMLP_list)),
        "f_score = {:.4f}".format(np.mean(F1_GCNMLP_list)))

    print("*mean GCNMTL* acc = {:.4f}".format(np.mean(acc_GCNMTL_list)),
        "auc = {:.4f}".format(np.mean(auc_GCNMTL_list)),
        "aupr = {:.4f}".format(np.mean(aupr_GCNMTL_list)),
        "f_score = {:.4f}".format(np.mean(f1_GCNMTL_list)))

    print('confumatrix:')


    print('KNN__')
    print(num00_test_knn)
    print(num01_test_knn)
    print(num10_test_knn)
    print(num11_test_knn)
    print(np.mean(num00_test_knn), np.mean(num01_test_knn), np.mean(num10_test_knn), np.mean(num11_test_knn))


    print('SVM__')
    print(num00_test_svm)
    print(num01_test_svm)
    print(num10_test_svm)
    print(num11_test_svm)
    print(np.mean(num00_test_svm), np.mean(num01_test_svm), np.mean(num10_test_svm), np.mean(num11_test_svm))


    print('nod2MLP__')
    print(num00_test_nod2MLP)
    print(num01_test_nod2MLP)
    print(num10_test_nod2MLP)
    print(num11_test_nod2MLP)
    print(np.mean(num00_test_nod2MLP), np.mean(num01_test_nod2MLP), np.mean(num10_test_nod2MLP), np.mean(num11_test_nod2MLP))

    print('nod2MTL__')
    print(num00_test_nod2MTL)
    print(num01_test_nod2MTL)
    print(num10_test_nod2MTL)
    print(num11_test_nod2MTL)
    print(np.mean(num00_test_nod2MTL), np.mean(num01_test_nod2MTL), np.mean(num10_test_nod2MTL), np.mean(num11_test_nod2MTL))

    print('GCNMLP__')
    print(num00_test_GCNMLP)
    print(num01_test_GCNMLP)
    print(num10_test_GCNMLP)
    print(num11_test_GCNMLP)
    print(np.mean(num00_test_GCNMLP), np.mean(num01_test_GCNMLP), np.mean(num10_test_GCNMLP), np.mean(num11_test_GCNMLP))

    print('GCNMTL__')
    print(num00_test_GCNMTL)
    print(num01_test_GCNMTL)
    print(num10_test_GCNMTL)
    print(num11_test_GCNMTL)
    print(np.mean(num00_test_GCNMTL), np.mean(num01_test_GCNMTL), np.mean(num10_test_GCNMTL), np.mean(num11_test_GCNMTL))

if __name__ == "__main__":
    main()