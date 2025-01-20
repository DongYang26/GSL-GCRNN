import pandas as pd

# Calculate adjacency matrix through correlation coefficient
df = pd.read_csv('./Data/trainData/bcn_L/bcn-L_202106010800-202106080800_1mins_2MHz.csv', header=None)  # first row represents header

corr_matrix = df.corr(method='spearman',min_periods=1)

adj_matrix = (corr_matrix > 0.9).astype(int)
print("adj_matrix.shape:{}".format(adj_matrix.shape))

adj_matrix.to_csv('./Data/trainData/bcn_L/101_adj_bcn_L__0.9_spearman.csv', index=False)

