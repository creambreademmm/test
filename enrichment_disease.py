'''Gene enrichment analysis - disease analysis'''
import matplotlib.pyplot as plt

# data
diseases = ['Diabetes Mellitus, Non-Insulin-Dependent', 'Schizophrenia', 'Mental Depression', 'Unipolar Depression', 'Major Depressive Disorder', 'Diabetes Mellitus, Experimental', 'Depressive disorder', 'Bipolar Disorder', 'Peripheral Neuropathy', 'Precancerous Conditions']
enrichment_ratios = [5.8140, 2.5977, 5.2326, 4.0250, 4.0250, 5.8140, 2.9070, 2.9070, 4.3605, 5.2326]
fdrs = [0.024805, 0.024805, 0.024805, 0.092839, 0.092839, 0.098644, 0.21006, 0.21006, 0.25765, 0.33617]
# Sort by Enrichment ratio from large to small
sorted_indices = sorted(range(len(enrichment_ratios)), key=lambda k: enrichment_ratios[k], reverse=False)
diseases = [diseases[i] for i in sorted_indices]
enrichment_ratios = [enrichment_ratios[i] for i in sorted_indices]
fdrs = [fdrs[i] for i in sorted_indices]

# plot
fig, ax = plt.subplots(figsize=(11, 3.5))
colors = ['steelblue' if fdr <= 0.1 else 'lightsteelblue' for fdr in fdrs]
bar_plot = ax.barh(diseases, enrichment_ratios, color=colors)

# Set horizontal and vertical axis labels
ax.set_xlabel('Enrichment ratio')

# Set the horizontal axis scale range
ax.set_xlim([0, max(enrichment_ratios) * 1.2])

# Remove top and right borders
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

legend_elements = [bar_plot[0], bar_plot[1]]
legend_labels = ['FDR ≤ 0.1', 'FDR > 0.1']
legend = ax.legend(legend_elements, legend_labels, loc='upper right')

# Adjust the position and size of the legend
legend.get_frame().set_edgecolor('white')
plt.setp(legend.get_texts(), color='black')
plt.setp(legend.get_title(), color='black')

plt.tight_layout()
plt.savefig('figures/diseaseEnrichment_alpha.png', dpi=300)
plt.show()
