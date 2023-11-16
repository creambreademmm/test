import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

print('reading data')
df = pd.read_excel('data/human_islets.xlsx', index_col=0)
print('read data finished')

# Extract data with gene id 1
gene1 = df.loc[801]

beta_data = gene1.iloc[651:947].values.flatten()
alpha_data = gene1.iloc[947:1516].values.flatten()
delta_data = gene1.iloc[1516:1546].values.flatten()
gamma_data = gene1.iloc[1546:1600].values.flatten()

# calculate p value
print('----------- beta VS alpha -----------')
u_statistic, p_value_mw = stats.mannwhitneyu(beta_data, alpha_data) 
print("Mann-Whitney U test results:") 
print("                         U Statistics:", u_statistic) 
print("                         p value:", p_value_mw) # Kruskal-Wallis test 
h_statistic, p_value_kw = stats.kruskal(beta_data, alpha_data) 
print("Kruskal-Wallis test results:") 
print("                         H Statistics:", h_statistic) 
print("                         p value:", p_value_kw) # independent samples t test 
t_statistic, p_value_t = stats.ttest_ind(beta_data, alpha_data) 
print("independent samples t test results:") 
print("                         t Statistics:", t_statistic) 
print("                         p value:", p_value_t) # Analysis of Variance (ANOVA) 
f_statistic, p_value_anova = stats.f_oneway(beta_data, alpha_data) 
print("Analysis of Variance (ANOVA) Results:") 
print("                         F Statistics:", f_statistic) 
print("                         p value:", p_value_anova)

print('----------- beta VS delta -----------')
u_statistic, p_value_mw = stats.mannwhitneyu(beta_data, delta_data) 
print("Mann-Whitney U test results:") 
print("                         U Statistics:", u_statistic) 
print("                         p value:", p_value_mw) # Kruskal-Wallis test 
h_statistic, p_value_kw = stats.kruskal(beta_data, delta_data) 
print("Kruskal-Wallis test results:") 
print("                         H Statistics:", h_statistic) 
print("                         p value:", p_value_kw) # independent samples t test 
t_statistic, p_value_t = stats.ttest_ind(beta_data, delta_data) 
print("independent samples t test results:") 
print("                         t Statistics:", t_statistic) 
print("                         p value:", p_value_t) # Analysis of Variance (ANOVA) 
f_statistic, p_value_anova = stats.f_oneway(beta_data, delta_data) 
print("Analysis of Variance (ANOVA) Results:") 
print("                         F Statistics:", f_statistic) 
print("                         p value:", p_value_anova)

print('----------- beta VS gamma -----------')
u_statistic, p_value_mw = stats.mannwhitneyu(beta_data, gamma_data) 
print("Mann-Whitney U test results:") 
print("                         U Statistics:", u_statistic) 
print("                         p value:", p_value_mw) # Kruskal-Wallis test 
h_statistic, p_value_kw = stats.kruskal(beta_data, gamma_data) 
print("Kruskal-Wallis test results:") 
print("                         H Statistics:", h_statistic) 
print("                         p value:", p_value_kw) # independent samples t test 
t_statistic, p_value_t = stats.ttest_ind(beta_data, gamma_data) 
print("independent samples t test results:") 
print("                         t Statistics:", t_statistic) 
print("                         p value:", p_value_t) # Analysis of Variance (ANOVA) 
f_statistic, p_value_anova = stats.f_oneway(beta_data, gamma_data) 
print("Analysis of Variance (ANOVA) Results:") 
print("                         F Statistics:", f_statistic) 
print("                         p value:", p_value_anova)


# Create a new DataFrame for plotting a violin plot
data = pd.DataFrame({'cell_type': ['alpha'] * len(alpha_data) + ['beta'] * len(beta_data) +  ['delta'] * len(delta_data) + ['gamma'] * len(gamma_data),
                     'expression': list(alpha_data) +  list(beta_data) + list(delta_data) + list(gamma_data)})

sns.violinplot(x='cell_type', y='expression', data=data)
sns.despine()
sns.stripplot(x='cell_type', y='expression', data=data, color='black', size=1)

plt.xlabel('Cell type')
plt.ylabel('Expression')

plt.savefig('figures/beta_CALM1.pdf')
plt.show()

