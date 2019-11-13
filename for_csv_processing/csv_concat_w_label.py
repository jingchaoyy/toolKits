"""
Created on  11/4/19
@author: Jingchao Yang
"""
import pandas as pd
import glob
import os

path = '/Users/jc/Desktop/test'  # use your path
all_files = glob.glob(path + "/*.csv")

li = []

for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    head, tail = os.path.split(filename)
    fn = tail.split('.')[0]
    dt = pd.to_datetime(fn, format='%Y%m%d%H')
    df['datetime'] = dt
    li.append(df)

frame = pd.concat(li, axis=0, ignore_index=True)
frame.to_csv('/Users/jc/Desktop/test/merged.csv')
