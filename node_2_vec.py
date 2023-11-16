import numpy as np
import pandas as pd
import networkx as nx
import argparse
import node2vec
from gensim.models import Word2Vec
import matplotlib.pyplot as plt

def Get_Graph():
    # Extract PPI network to mat matrix
    dg = pd.read_csv('data/9606.protein.info.v11.0.txt', sep='\t')
    print('read graph')
    ensp2hgnc = {i: j for i, j in dg[['protein_external_id', 'preferred_name']].itertuples(index=False, name=None)}
    df=pd.read_csv('data/9606.protein.links.detailed.v11.0.txt', sep=' ')
    df = df.drop(df[df['combined_score']<400].index)
    df['protein1'] = df['protein1'].apply(lambda x: ensp2hgnc[x])
    df['protein2'] = df['protein2'].apply(lambda x: ensp2hgnc[x])
    print('Replace name and number')

    nt = nx.from_pandas_edgelist(df, 'protein1', 'protein2', 'combined_score')
    mat = nx.to_numpy_matrix(nt, weight='combined_score')

    mat = mat.astype('int16') 
    nodes = np.array(nt.nodes)

    ppi_dict = dict(zip(nodes, range(len(nodes))))

    nt_cg = nx.from_pandas_edgelist(df, 'protein1', 'protein2', 'combined_score')
    return nt_cg
    

def parse_args():
	parser = argparse.ArgumentParser(description="Run node2vec.")

	parser.add_argument('--output', nargs='?', default=r'data/PPI_400.emb',
	                    help='Embeddings path')
	parser.add_argument('--dimensions', type=int, default=64,
	                    help='Number of dimensions. Default is X.')
	parser.add_argument('--walk-length', type=int, default=80,
	                    help='Length of walk per source. Default is X.')
	parser.add_argument('--num-walks', type=int, default=10,
	                    help='Number of walks per source. Default is 5.')
	parser.add_argument('--window-size', type=int, default=5,
                    	help='Context size for optimization. Default is 10.')
	parser.add_argument('--iter', default=1, type=int,
                      help='Number of epochs in SGD')
	parser.add_argument('--workers', type=int, default=8,
	                    help='Number of parallel workers. Default is 8.')
	parser.add_argument('--p', type=float, default=2.0,
	                    help='Return hyperparameter. Default is XX.')
	parser.add_argument('--q', type=float, default=0.25,
	                    help='Inout hyperparameter. Default is XX.')
	parser.add_argument('--weighted', dest='weighted', action='store_true',
	                    help='Boolean specifying (un)weighted. Default is unweighted.')
	parser.add_argument('--unweighted', dest='unweighted', action='store_false')
	parser.set_defaults(weighted=True)
	parser.add_argument('--directed', dest='directed', action='store_true',
	                    help='Graph is (un)directed. Default is undirected.')
	parser.add_argument('--undirected', dest='undirected', action='store_false')
	parser.set_defaults(directed=False)

	return parser.parse_args()



def learn_embeddings(walks):
	for walk in walks:
		walk=str(walk)

	model = Word2Vec(walks, vector_size=args.dimensions, window=args.window_size, min_count=0, sg=1, workers=args.workers)
	model.wv.save_word2vec_format(args.output)
	# The parameters specified during training are as follows:
	# walks is a sample sequence of random walks
	# sg=1 is the skip-gram algorithm, which inputs the target word and outputs the context. sensitive to low frequency words
	# size is the dimension of the output word vector, 512
	# window is the maximum distance between the current word and the target word in the sentence
	# min_count is to filter words. Words with a frequency less than min-count will be ignored.
	# workers controls the parallelism of training. This parameter is only valid after Cpython is installed. Otherwise, only a single core can be used.
	# iter is Number of iterations
	return


def main(args):
	'''
	Pipeline for representational learning for all nodes in a graph.
	'''
	print('start')
	nx_G = Get_Graph()	
	print('get graph')
	G = node2vec.Graph(nx_G, args.directed, args.p, args.q)
	G.preprocess_transition_probs()

	walks = G.simulate_walks(args.num_walks, args.walk_length)
	print('walks')
	
	learn_embeddings(walks)
	print('learn embedding')	

if __name__ == "__main__":
	args = parse_args()
	main(args)

