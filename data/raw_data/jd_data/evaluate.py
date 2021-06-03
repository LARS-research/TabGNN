import pdb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score



data1=pd.read_csv('./kaggle_submission.csv')
data2=pd.read_csv('./test.csv')

data1=list(data1['label'])
data2=list(data2['label'])

print(roc_auc_score(data2,data1))




