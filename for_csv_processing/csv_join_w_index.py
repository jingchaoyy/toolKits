"""
Created on  2/19/2020
@author: Jingchao Yang
"""
import pandas as pd
import os
from functools import reduce

file_path = r'data_path'

dfs = []
for f in os.listdir(file_path):
    if f.endswith(".csv"):
        data = pd.read_csv(os.path.join(file_path, f), usecols=['key', 'prediction'])
        new_col_name = f.split('.')[0]
        data = data.rename(index=str, columns={"prediction": new_col_name})
        dfs.append(data)

df_final = reduce(lambda left, right: pd.merge(left, right, on='key'), dfs)
df_final.to_csv(os.path.join(file_path, 'merged\merged.csv'))