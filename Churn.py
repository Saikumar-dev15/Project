import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
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

df = df.drop("customerID", axis=1)
#print(df)

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

#print(df["Churn"])

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
#print(df["InternetService"])

df["PaymentMethod"] = df["PaymentMethod"].map({
    "Electronic check" : 0,
   "Bank transfer (automatic)": 1,
   "Credit card (automatic)": 2,
    "Mailed check": 3
})
#print(df["PaymentMethod"])


df["DeviceProtection"] = df["DeviceProtection"].map({
    "Yes": 0,
    "No": 1
})
#print(df["DeviceProtection"])

df["OnlineSecurity"] = df["OnlineSecurity"].map({
    "Yes": 0,
    "No": 1,
    "No internet service" : 2
})
#print(df["OnlineSecurity"])

df["TotalCharges"] = pd.to_numeric(                         #to convert into int numerical because median works only in numerical values are present
    df["TotalCharges"],
    errors="coerce"
)

df["TotalCharges"] = df["TotalCharges"].fillna(
    df["TotalCharges"].median()                             
)




x = df[["tenure",
        "Contract",
        "MonthlyCharges",
        "TotalCharges",
        "InternetService",
        "PaymentMethod"]]

y = df["Churn"]

x_train , x_test , y_train, y_test = train_test_split(
    x,y, test_size=0.2 , random_state=42
)


print("NaN in x_train:", np.isnan(x_train).sum())
print("NaN in x_test:", np.isnan(x_test).sum())

print("Infinity in x_train:", np.isinf(x_train).sum())
print("Infinity in x_test:", np.isinf(x_test).sum())

x_train = np.nan_to_num(x_train,
                        nan=0.0,
                        posinf=0.0,
                        neginf=0.0)

x_test = np.nan_to_num(x_test , nan=0.0,
                       posinf=0.0,
                       neginf=0.0)

model = LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

print("Accuracy Score: ", accuracy_score(y_test, y_pred))

#sns.histplot(data= df, x="gender" ,color = "blue", multiple="stack")
#plt.xlabel("gender")
#plt.show()

#sns.countplot( x="Churn",data=df,  hue="Churn", palette="Set2")
#plt.show()


#sns.countplot(x="InternetService", data=df, color="Orange",
#                                            edgecolor="black")
#plt.show()

#sns.countplot(x="DeviceProtection", data=df, color="SkyBlue",
#                                            edgecolor="black")
#plt.show()


#sns.boxplot(x="Churn", y="MonthlyCharges", data=df)
#plt.title("Monthly Charges by Churn")
#plt.show()


#sns.pairplot(df[["tenure", "MonthlyCharges", "TotalCharges", "Churn"]])
#plt.show()