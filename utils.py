import argparse
import scipy.sparse as sp
import numpy as np
import torch
import networkx as nx
import multiprocessing as mp
import torch.nn.functional as F
import math
import random
random.seed(3407)
from sklearn import metrics
from sklearn.metrics import f1_score, precision_recall_curve, auc, confusion_matrix
from copy import deepcopy
from scipy.spatial.distance import pdist,squareform
import pandas as pd

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=500,
                help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lr_d', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=5e-4)     
    parser.add_argument('--dropout', type=float, default=0.5)

    return parser

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape).to_dense()


def load_data():
    dg = pd.read_csv('data/9606.protein.info.v11.0.txt', sep='\t')
    print('read graph')
    print(len(dg))
    ensp2hgnc = {i: j for i, j in dg[['protein_external_id', 'preferred_name']].itertuples(index=False, name=None)}
    df=pd.read_csv('data/9606.protein.links.detailed.v11.0.txt', sep=' ')

    df['protein1'] = df['protein1'].apply(lambda x: ensp2hgnc[x])
    df['protein2'] = df['protein2'].apply(lambda x: ensp2hgnc[x])
    prot1 = df['protein1'].astype(str) 

    # load label
    load_label = pd.read_excel('data_out/label_Beta.xlsx')
    label_genes = load_label['gene'].astype(str) 
    label_genes = np.array(label_genes)
    label_genes = label_genes.tolist()
    nodes_common = list(set(prot1).intersection(set(label_genes)))
    print('****')
    print(len(nodes_common))

    df = nx.from_pandas_edgelist(df, 'protein1', 'protein2', 'combined_score')
    
    return df, nodes_common, load_label

def load_data_nod2vec(FileName):
    f=open(FileName,'r')
    feature=[]
    genes = []
    label_d=[]
    label_c=[]
    gene_list = []
    feature_list=[]
    label_c_list=[]
    label_d_list=[]
    feature_label=f.readlines()
    for lines in feature_label:
        line1=lines.rsplit(' ', 2)
        label_d.append(line1[1])
        label_c.append(line1[2])
        line2=line1[0].split(' ', 1)
        genes.append(line2[0])
        feature.append(line2[1])

    for g in genes:
        gene_list.append(g)
    for l in feature:
        item_list=[]
        items=l.split(' ')
        for i in items:
            item_list.append(float(i))
        feature_list.append(item_list)

    for d in label_d:
        label_d_list.append(int(d))

    for c in label_c:
        yc=c.split('\\n')
        label_c_list.append(int(yc[0]))
    
    return gene_list, feature_list, label_d_list, label_c_list


def split_train_val_test(df, nodes_common, load_label):
    nodes = np.array(df.nodes)
    nodes = list(nodes)
    random.shuffle(nodes)
    label_d = []
    label_c = []

    for i in range(len(nodes)):
        if nodes[i] in nodes_common:
            genename = nodes[i]
            n_index = load_label[load_label.gene == genename].index.tolist()[0]
            yd_n = load_label['yd'][n_index]
            yc_n = load_label['yc'][n_index]
            label_d.append(yd_n)
            label_c.append(yc_n)
        else:
            label_d.append(2)
            label_c.append(2)

    label_d = np.array(label_d)
    label_c = np.array(label_c)
    adj = sp.coo_matrix(nx.to_numpy_matrix(df, nodelist=nodes))
    features = np.eye(len(nodes))
    features = torch.FloatTensor(features)
    label_d = torch.LongTensor(label_d)
    label_c = torch.LongTensor(label_c)
    adj = sparse_mx_to_torch_sparse_tensor(adj)


    '''split training set, test set, verification set'''
    num_00 = 0
    num_10 = 0
    num_11 = 0
    num_2 = 0           #unknow
    for i in range(len(label_d)):
        if label_d[i]==0 and label_c[i]==0:
            num_00 += 1
        elif label_d[i]==0 and label_c[i]==1:
            label_c[i]=0
            num_00 += 1
        elif label_d[i]==1 and label_c[i]==0:
            num_10 += 1
        elif label_d[i]==1 and label_c[i]==1:
            num_11 += 1
        else:
            num_2 += 1

    print('***num of 00, 10, 11, unknow:')
    print(num_00)
    print(num_10)
    print(num_11)
    print(num_2)

    train_00_num = int(0.7*num_00)
    val_00_num = int(0.2*num_00)

    train_10_num = int(0.7*num_10)
    val_10_num = int(0.2*num_10)

    train_11_num = int(0.7*num_11)
    val_11_num = int(0.2*num_11)

    n_00_train = 0
    n_00_val = 0
    n_10_train = 0
    n_10_val = 0
    n_11_train = 0
    n_11_val = 0

    idx_train_00 = []
    idx_train_11 = []
    idx_train_10 = []
    idx_val = []
    idx_test = []

    for l in range(len(label_c)):
        if label_d[l]==0 and label_c[l]==0:
            if n_00_train<train_00_num:
                n_00_train+=1
                idx_train_00.append(l)
            elif n_00_val<val_00_num:
                n_00_val+=1
                idx_val.append(l)
            else:
                idx_test.append(l)

        elif label_d[l]==1 and label_c[l]==1:
            if n_11_train<train_11_num:
                n_11_train+=1
                idx_train_11.append(l)
            elif n_11_val<val_11_num:
                n_11_val+=1
                idx_val.append(l)
            else:
                idx_test.append(l)

        elif label_d[l]==1 and label_c[l]==0:
            if n_10_train<train_10_num:
                n_10_train+=1
                idx_train_10.append(l)
            elif n_10_val<val_10_num:
                n_10_val+=1
                idx_val.append(l)
            else:
                idx_test.append(l)
    
    idx_train_11_00 = idx_train_11 + idx_train_00
    random.shuffle(idx_train_11_00)
    random.shuffle(idx_train_10)
    idx_train_order = idx_train_11_00 + idx_train_10
    idx_train = idx_train_11_00 + idx_train_10
    random.shuffle(idx_train)
    random.shuffle(idx_val)
    random.shuffle(idx_test)

    idx_train = torch.LongTensor(idx_train)
    idx_train_order = torch.LongTensor(idx_train_order)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    print('len of traindata: ',len(idx_train))
    print('len of valdata: ',len(idx_val))
    print('len of testdata: ',len(idx_test))

    return nodes, features, adj, label_d, label_c, idx_train_order, idx_train, idx_val, idx_test


def split_train_untrain(label, train_idx, nclass, train_ratio=0.2):
    """
    split train-dataset to train_set and untrain_set
    """
    assert 0 <= train_ratio <= 1
    idx_train = []
    idx_untrain = []

    num_0_1 = [0,0]
    idx_0_1 = [[],[]]
    
    for idx in train_idx:
        if label[idx]==0:
            idx_0_1[0].append(idx.item())
            num_0_1[0]+=1
        else:
            idx_0_1[1].append(idx.item())
            num_0_1[1]+=1

    for c in range(nclass):
        train_num = int(num_0_1[c]*train_ratio)
        idx_train += idx_0_1[c][:train_num]
        idx_untrain += idx_0_1[c][train_num:]

    np.random.shuffle(idx_train)
    np.random.shuffle(idx_untrain)

    idx_train = torch.LongTensor(idx_train)
    idx_untrain = torch.LongTensor(idx_untrain)

    return idx_train, idx_untrain

def sel_idx_wspl(score,untrain_idx,label,add_ratio=0.5):
    label_untrain = label
    y = np.array(label_untrain.cpu())
    add_indices = []
    nclss = 2
    avgLoss = np.zeros(nclss)
    num_sort = np.zeros(nclss)
    num_sort_ = np.zeros(nclss)
    Hi = np.zeros(nclss)
    Ci = np.zeros(nclss)
    curri_num = np.zeros(nclss)
    curri_lack_num = np.zeros(nclss)
    assert score.shape[1] == nclss
    count_per_class = [sum(y == c) for c in range(nclss)]
    #pred_y = np.argmax(score,axis=1)
    for cls in range(nclss):
        indices = np.where(y==cls)[0]
        cls_score = score[indices, cls]
        idx_sort = np.argsort(cls_score)
        add_num = min(int(np.ceil(count_per_class[cls] * add_ratio)),
                      indices.shape[0])
        add_indices += list(indices[idx_sort[-add_num:]])
        
        avgLoss[cls] = np.sum(-np.log(cls_score[idx_sort[-add_num:]])) / add_num
        num_sort[cls] = count_per_class[cls]
        num_sort_[cls] = add_num 
        Hi[cls] = num_sort[cls] / np.exp(avgLoss[cls]) # 1 / H

    print('num_sort: ',num_sort)
    print('num_sort_: ',num_sort_)

    add_indices = torch.LongTensor(add_indices)
    add_idx = untrain_idx[add_indices]

    # normalize H to [min, max]
    min_ = min(num_sort)
    max_ = max(num_sort)
    maxHi = max(Hi)
    minHi = min(Hi)
    for cls in range(len(Hi)):
        Hi[cls] = (max_ - min_) * (Hi[cls] - minHi) / (maxHi - minHi) + min_
        Hi[cls] = np.ceil(Hi[cls])
    
    yc_pred = np.argmax(score, axis=1)
    yc_true = label_untrain
    cmcv = confusion_matrix(y, yc_pred)

    cv = []
    cv.append(cmcv[0][1])
    cv.append(cmcv[1][0])
    cv = np.array(cv)

    w = cv*(np.sum(Hi)/np.sum(cv*Hi))
    w = w/(min(w))

    for i in range(2):
        if (w[i] < 1) or (math.isnan(w[i])):
            w[i] = 1.0
        elif w[i] > 3:
            w[i] = 3.0

    sortNum = np.sort(num_sort_)
    indexSortHi = np.argsort(-Hi)
    newCurriculumNum = np.zeros(nclss)
    for i, n in enumerate(sortNum):
        newCurriculumNum[indexSortHi[i]] = n

    subCuccriculumNum = newCurriculumNum - num_sort_

    w = w.astype(float)
    return add_idx, w, subCuccriculumNum


def accuracy(output, labels):
    label_cpu = labels.detach().cpu()
    output_cpu = output.detach().cpu()
    preds = output_cpu.max(1)[1].type_as(label_cpu)
    correct = preds.eq(label_cpu).double()
    correct = correct.sum()
    acc = np.array(correct / len(label_cpu))
    return acc

def roc_auc(output, labels):
    fpr, tpr, thresholds = metrics.roc_curve(labels.detach().cpu(), F.softmax(output, dim=-1)[:,1].detach().cpu())
    auc_score = auc(fpr, tpr)
    return auc_score, fpr, tpr

def aupr(output, labels):
    precision, recall, thresholds = precision_recall_curve(labels.detach().cpu(), F.softmax(output, dim=-1)[:,1].detach().cpu())
    aupr_score = auc(recall, precision)
    return aupr_score

def f1(output, labels):
    return f1_score(labels.detach().cpu(), torch.argmax(output, dim=-1).detach().cpu(), average='macro')

def Evaluation_test(output, labels):
    acc = accuracy(output, labels)
    auc_score, fpr, tpr = roc_auc(output, labels)
    aupr_score = aupr(output, labels)
    macro_F = f1(output,labels)

    return acc, auc_score, aupr_score, macro_F, fpr, tpr


def src_smote(adj,features,labels,idx_train,portion=0,im_class_num=2):
    c_largest = 1
    adj_back = adj
    chosen = None
    new_features = None

    c0_idx = []
    c1_idx = []
    c2_idx = []
    for idx in idx_train:
        if labels[idx]==0:
            c0_idx.append(idx)
        elif labels[idx]==1:
            c1_idx.append(idx)
        else:
            c2_idx.append(idx)

    avg_number = int((len(c0_idx)+len(c1_idx))/(c_largest+1))

    for i in range(im_class_num):
        new_chosen = idx_train[(labels==(c_largest-i))[idx_train]]
        print('nums of class {} is {}'.format(c_largest-i, new_chosen.shape[0]))
        if portion == 0:
            c_portion = int(avg_number/new_chosen.shape[0])
            portion_rest = (avg_number/new_chosen.shape[0]) - c_portion
        else:
            c_portion = int(portion)
            portion_rest = portion-c_portion

        for j in range(c_portion):
            num = int(new_chosen.shape[0])
            new_chosen = new_chosen[:num]

            chosen_embed = features[new_chosen,:]
            distance = squareform(pdist(chosen_embed.detach().cpu()))
            np.fill_diagonal(distance,distance.max()+100)

            idx_neighbor = distance.argmin(axis=-1)

            print('^^^')
            print(features.shape)
            print(chosen_embed.shape)
            print(idx_neighbor)
            print(len(idx_neighbor))

            interp_place = random.random()
            embed = chosen_embed + (chosen_embed[idx_neighbor,:]-chosen_embed)*interp_place

            if chosen is None:
                chosen = new_chosen
                new_features = embed
            else:
                chosen = torch.cat((chosen, new_chosen), 0)
                new_features = torch.cat((new_features, embed),0)

        num = int(new_chosen.shape[0]*portion_rest)
        new_chosen = new_chosen[:num]

        chosen_embed = features[new_chosen,:]
        distance = squareform(pdist(chosen_embed.detach().cpu()))
        np.fill_diagonal(distance,distance.max()+100)

        idx_neighbor = distance.argmin(axis=-1)
            
        interp_place = random.random()
        embed = chosen_embed + (chosen_embed[idx_neighbor,:]-chosen_embed)*interp_place

        if chosen is None:
            chosen = new_chosen
            new_features = embed
        else:
            chosen = torch.cat((chosen, new_chosen), 0)
            new_features = torch.cat((new_features, embed),0)

    add_num = chosen.shape[0]
    new_adj = adj_back.new(torch.Size((adj_back.shape[0]+add_num, adj_back.shape[0]+add_num)))
    new_adj[:adj_back.shape[0], :adj_back.shape[0]] = adj_back[:,:]
    new_adj[adj_back.shape[0]:, :adj_back.shape[0]] = adj_back[chosen,:]
    new_adj[:adj_back.shape[0], adj_back.shape[0]:] = adj_back[:,chosen]
    new_adj[adj_back.shape[0]:, adj_back.shape[0]:] = adj_back[chosen,:][:,chosen]

    features_append = deepcopy(new_features)
    labels_append = deepcopy(labels[chosen])

    idx_new = np.arange(adj_back.shape[0], adj_back.shape[0]+add_num)
    idx_train_append = idx_train.new(idx_new)

    features = torch.cat((features,features_append), 0)
    labels = torch.cat((labels,labels_append), 0)
    idx_train = torch.cat((idx_train,idx_train_append), 0)
    adj = new_adj.to_sparse()

    b=torch.randperm(idx_train.size(0))
    idx_train = idx_train[b]    

    c0_idx = []
    c1_idx = []
    c2_idx = []
    for idx in idx_train:
        if labels[idx]==0:
            c0_idx.append(idx)
        elif labels[idx]==1:
            c1_idx.append(idx)
        else:
            c2_idx.append(idx) 
    print('len of c0,c1,c2:')
    print(len(c0_idx))
    print(len(c1_idx))
    print(len(c2_idx))

    return adj, features, labels, idx_train