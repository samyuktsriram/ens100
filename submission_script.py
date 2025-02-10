import pandas as pd
import numpy as np

df = pd.read_csv('trial.csv') #this should have a few columns with the anomaly predictions, consistent with inference logic

def inference_logic(row):
    if row['forest_anom'] > 0:
        return row['class_preds']
    elif row['forest_anom'] < 0:
        return 6

# Apply logic
df['Label'] = df.apply(inference_logic, axis=1)
df.drop(['forest_anom', 'class_preds', 'Unnamed: 0.1','Unnamed: 0'], axis=1, inplace=True)
df.to_csv('submission_file.csv')