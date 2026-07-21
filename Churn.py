import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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
#print(df.value_counts())


df["Churn"] = df["Churn"].map({
    "No": 0,
    "Yes": 1
})

print("Churn")

df["Contract"] = df["Contract"].map({
    "Month-to-month": 0,
    "One year": 1,
    "Two year": 2
})

#print(df["Contract"])


df["InternetService"] = df["InternetService"].map({
    "DSL" : 0,
    "Fiber optic": 1,
    "No": 2
})
print(df["InternetService"])

df["PaymentMethod"] = df["PaymentMethod"].map({
    "Electronic check" : 0,
   "Bank transfer (automatic)": 1,
   "Credit card (automatic)": 2,
    "Mailed check": 3
})
print(df["PaymentMethod"])

df["InternetService"] = df["InternetService"].map({
    "DSL" : 0,
    "Fiber optic": 1,
    "No": 2
})
print(df["InternetService"])




#sns.histplot(data= df, x="gender" ,color = "blue", multiple="stack")
#plt.xlabel("gender")
#plt.show()

#sns.countplot( x="Churn",data=df,  hue="Churn", palette="Set2")
#plt.show()

sns.countplot(x="PaymentMethod", data=df, color="green",
                                            edgecolor="black")
plt.show()

#sns.boxplot(x="Churn", y="MonthlyCharges", data=df)
#plt.title("Monthly Charges by Churn")
#plt.show()


#sns.pairplot(df[["tenure", "MonthlyCharges", "TotalCharges", "Churn"]])
#plt.show()