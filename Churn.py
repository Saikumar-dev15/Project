import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time 
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
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



df["TotalCharges"] = pd.to_numeric(                         #to convert into int numerical because median works only in numerical values are present
    df["TotalCharges"],
    errors="coerce"
)

df["TotalCharges"] = df["TotalCharges"].fillna(
    df["TotalCharges"].median()                             
)

#Preprocessing
cat_columns =[ "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod"
]

num_columns = ["tenure", "MonthlyCharges", "TotalCharges"]

x = df.drop("Churn", axis=1)
y = df["Churn"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_columns),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_columns)
    ]
)

x_train , x_test , y_train, y_test = train_test_split(
    x,y, test_size=0.2 , random_state=42
)

X_train = preprocessor.fit_transform(x_train)
X_test = preprocessor.transform(x_test)

#print(f"X_Trained from Preprocessor: {X_train}")
#print(f"X_Tested from Preprocessor: {X_test}")


#print("NaN in x_train:", np.isnan(x_train).sum())
#print("NaN in x_test:", np.isnan(x_test).sum())

#print("Infinity in x_train:", np.isinf(x_train).sum())
#print("Infinity in x_test:", np.isinf(x_test).sum())

X_train = np.nan_to_num(X_train,
                        nan=0.0,
                        posinf=0.0,
                        neginf=0.0)

X_test = np.nan_to_num(X_test , nan=0.0,
                       posinf=0.0,
                       neginf=0.0)

start = time.time()
model = LogisticRegression( class_weight="balanced",max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
end = time.time()

sk_time = end - start
print(f"Time to Predict Sk Logestic Model: {sk_time}")

print("Recall Score: ", recall_score(y_test, y_pred))
print("Roc Aug Score: ", roc_auc_score(y_test, y_pred))
print("Accuracy Score: ", accuracy_score(y_test, y_pred))
print(f"Precision Score: {precision_score(y_test, y_pred)}")
print("f1 Score: ", f1_score(y_test, y_pred))
print("Confusion Matrix: ", confusion_matrix(y_test, y_pred))


#Numpy Logistic Regression






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