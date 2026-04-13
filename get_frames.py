import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

df = pd.read_csv('final_dataset.csv')
test_word = df.iloc[4]

test_word_list = test_word.to_list()

# mask = df.map(lambda x: isinstance(x, int) and x == 0)
# first_zero_col = mask.idxmax(axis=1).where(mask.any(axis=1), None)

# print(first_zero_col)

for i in range(1, len(test_word_list)):
    if test_word_list[i] == 0.0:
        print(df.columns[i])
        break

print(f"last column: {df.columns[-1]}")