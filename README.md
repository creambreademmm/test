# test
**data:**
t2dall.xlsx: Gene differential expression data for various pancreatic cell types in T2D.

Download the T2D-related gene data GeneCards-T2D.csv from GeneCards, filter out the data whose Category is Protein Coding, and get GeneCards-T2D-selected.xlsx.


**run node_2_vec.py:**
Using node2vec to get the embedding of nodes in the PPI network (keep edges with combined_score<400) and save the embeddings in PPI_400.emb.

**run label.py:**
Specify celltype(for example, Delta), load data t2dall.xlsx and GeneCards-T2D-selected.xlsx, get label_Delta.xlsx including gene labels and cell type-specific labels of each cell in t2dall.xlsx.

**run combine.py：**
input label_Delta.xlsx and PPI_400.emb，get feature_with_label_nod2vec_Delta.txt.

**run DiGCellNet.py**
Get the results of all methods, record the results. Then run figure.py to get Figure2.

**run Enrichment.py** performs enrichment analysis and gets the result Figure4.

Verify the results on an independent data set, download the independent data set human_islets.xlsx(ID code GEO: GSE81608), **run violin.py **to get Figure4.
