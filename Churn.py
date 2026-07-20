import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score
)

df = pd.read_csv("Churn.csv")

#print(df.head(5))
#print(df.shape)
#print(df.columns)
#df.duplicated
#print(df.drop_duplicates)
#print(df.dropna())
#print(df.isnull())

x= df["gender"]
y= df["SeniorCitizen"]
plt.bar(x.index,y.value_counts, color = "blue")
plt.xlabel("gender")
plt.ylabel("Seniorcitizens")
plt.title("Churn Data Collections")
plt.show()