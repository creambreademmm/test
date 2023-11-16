'''Gene enrichment analysis ---pathway'''
import matplotlib.pyplot as plt

# data
description = ['Thyroid hormone signaling pathway', 'cGMP-PKG signaling pathway', 'Carbohydrate digestion and absorption', 'Prostate cancer', 'Apelin signaling pathway', 'Pancreatic cancer', 'Insulin secretion', 'Thyroid hormone synthesis', 'Glucagon signaling pathway', 'AGE-RAGE signaling pathway in diabetic complications', ]
enrichment_ratios = [1.1311, 1.1311, 0.75410, 0.75410, 1.2568, 1.0055, 0.75410, 0.75410, 0.75410, 0.75410]
fdrs = [2.8764e-7, 0.0000095762, 0.00013984, 0.00013984, 0.00041138, 0.0010700, 0.0027396, 0.0027396, 0.0027396, 0.0027396]
# Sort by Enrichment ratio from large to small
sorted_indices = sorted(range(len(enrichment_ratios)), key=lambda k: enrichment_ratios[k], reverse=False)
description = [description[i] for i in sorted_indices]
enrichment_ratios = [enrichment_ratios[i] for i in sorted_indices]
fdrs = [fdrs[i] for i in sorted_indices]

# plot
fig, ax = plt.subplots(figsize=(13, 4))
colors = ['steelblue' if fdr <= 0.05 else 'lightsteelblue' for fdr in fdrs]
bar_plot = ax.barh(description, enrichment_ratios, color=colors)

# Set horizontal and vertical axis labels
ax.set_xlabel('Enrichment ratio')

# Set the horizontal axis scale range
ax.set_xlim([0, max(enrichment_ratios) * 1.2])

# Remove top and right borders
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

legend_elements = [bar_plot[0], bar_plot[1]]
legend_labels = ['FDR ≤ 0.05']
legend = ax.legend(legend_elements, legend_labels, loc='upper right')

# Adjust the position and size of the legend
legend.get_frame().set_edgecolor('white')
plt.setp(legend.get_texts(), color='black')
plt.setp(legend.get_title(), color='black')

plt.tight_layout()
plt.savefig('figures/pathwayEnrichment_alpha.png', dpi=300)
plt.show()
