import matplotlib.pyplot as plt
import numpy as np

# alpha
y_knn_alpha = [0.5222,0.5250,0.5191]
y_svm_alpha = [0.5525,0.5652,0.5089]
y_mlp_alpha = [0.5541,0.5601,0.5339]        #[0.5558,0.5639,0.5340]
y_mt_alpha = [0.6472,0.6198,0.5853]         #[0.6577,0.6249,0.5905]

# beta
y_knn_beta = [0.5437,0.5468,0.5363]
y_svm_beta = [0.5763,0.5812,0.5020]
y_mlp_beta = [0.5726,0.5789,0.5240]         #[0.5716,0.5761,0.5239]
y_mt_beta = [0.6526,0.6354,0.5893]           #[0.6519,0.6345,0.6006]

# delta
y_knn_delta = [0.5175,0.5205,0.5161]
y_svm_delta = [0.5561,0.5642,0.5283]
y_mlp_delta = [0.5464,0.5616,0.5299]          #[0.5476,0.5626,0.5275]
y_mt_delta = [0.6578,0.6287,0.5919]              #[0.6505,0.6237,0.5929]

# gamma
y_knn_gamma = [0.5252,0.5256,0.5167]
y_svm_gamma = [0.5679,0.5775,0.4776]
y_mlp_gamma = [0.5661,0.5687,0.5202]          #[0.5710,0.5668,0.5223]
y_mt_gamma = [0.6456,0.6361,0.5905]           #[0.6550,0.6387,0.5881]

labels = ["ACC","AUC","F1-Score"]

data_alpha = np.array([y_knn_alpha, y_svm_alpha, y_mlp_alpha, y_mt_alpha])
data_beta = np.array([y_knn_beta, y_svm_beta, y_mlp_beta, y_mt_beta])
data_delta = np.array([y_knn_delta, y_svm_delta, y_mlp_delta, y_mt_delta])
data_gamma = np.array([y_knn_gamma, y_svm_gamma, y_mlp_gamma, y_mt_gamma])

colors = ["lightgreen","lightblue","gold","lightcoral"]

x = np.arange(len(labels))

fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))

# Draw the first subplot
for i in range(data_alpha.shape[0]):
    axs[0, 0].bar(x + i * 0.2 - 0.3, data_alpha[i], width=0.2, color=colors[i], alpha=1)
axs[0, 0].set_title("Alpha cells")

# Draw the second subplot
for i in range(data_beta.shape[0]):
    axs[0, 1].bar(x + i * 0.2 - 0.3, data_beta[i], width=0.2, color=colors[i], alpha=1)
axs[0, 1].set_title("Beta cells")

# Draw the third subplot
for i in range(data_delta.shape[0]):
    axs[1, 0].bar(x + i * 0.2 - 0.3, data_delta[i], width=0.2, color=colors[i], alpha=1)
axs[1, 0].set_title("Delta cells")

# Draw the fourth subplot
for i in range(data_gamma.shape[0]):
    axs[1, 1].bar(x + i * 0.2 - 0.3, data_gamma[i], width=0.2, color=colors[i], alpha=1)
axs[1, 1].set_title("Gamma cells")

fig.legend(["KNN", "SVM", "MLP", "DiGCellNet"], loc="upper center", ncol=4)

for ax in axs.flat:
    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

plt.subplots_adjust(wspace=0.3, hspace=0.4)

plt.savefig("figures/figure1.pdf")
