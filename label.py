import pandas as pd
import numpy as np

celltype='Delta cells'

data_t2d = pd.read_excel('data/t2dall.xlsx')
load_yd = pd.read_excel('data/GeneCards-T2D-selected.xlsx')

# Get the disease-related gene list related_genes_list from load_yd
related_genes_list = []
for i in range(len(load_yd)):
    related_genes_list.append(load_yd['Gene Symbol'][i])

# Find all gene sets containing celltype and |value|>=1 in the t2d data
selected_t2d = data_t2d[((data_t2d['Cell type 1'] == celltype) | (data_t2d['Cell type 2'] == celltype)) & (abs(data_t2d['value']) >= 1)]
geneset_t2d = selected_t2d['Gene'].values.tolist()

gene_t2d=list(set(data_t2d['Gene']))
l=len(gene_t2d)

matrix = pd.DataFrame(0, index=range(l), columns=["gene","yd","yc"])
matrix['gene']=gene_t2d

yd_list=[0]*l
yc_list=[0]*l

for i in range(l):
    g = matrix['gene'][i]
    if g in related_genes_list:
        yd_list[i]=1
    if g in geneset_t2d:
        yc_list[i]=1


matrix['yd']=yd_list
matrix['yc']=yc_list

print(matrix)
matrix.to_excel('data_out/label_Delta.xlsx', index=False)
