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
    roc_auc_score,
    precision_recall_curve,
    roc_curve
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
print(f"Precision Recall Curve: {precision_recall_curve(y_test, y_pred)}")
print(f"Roc Curve: {roc_curve(y_test, y_pred)}")
print("Confusion Matrix: ", confusion_matrix(y_test, y_pred))


#Numpy Logistic Regression

class LogisticRegressionNumpy:
    
    def __init__(self, lr=0.01, epochs=2000):
        self.lr = lr
        self.epochs = epochs
        
    
    def sigmoid(self, z):
        return 1/(1+np.exp(-z))
    
    def fit(self, x,y):
        
        x = np.array(x)
        y = np.array(y)
        
        samples, features = x.shape

        self.weights = np.zeros(features)
        self.bias = 0
        
        for i in range(self.epochs):
            linear = np.dot(x,self.weights)+ self.bias
            
            predicition = self.sigmoid(linear)
            
            prediction = np.clip(predicition, 1e-15, 1-1e-15)
            
            loss = -(1/samples) * np.sum(
                y*np.log(prediction) + (1-y)* np.log(1- prediction)
            )
            
            dw = (1/samples)*np.dot(x.T,(predicition - y))
            db = (1/samples)*np.sum(predicition-y)
            
            self.weights -= self.lr*dw
            self.bias  -= self.lr*db
            
            if i %100 ==0:
                print(f"Epoch {i}: Loss = {loss:.4f}")
            
    def predict_probability(self, x):
        linear = np.dot(x,self.weights)+ self.bias
        return self.sigmoid(linear)
    
    def predict(self,x):
        prob = self.predict_probability(x)
        return np.where(prob>=0.5,1,0)  
    

start_num = time.time()

scratch = LogisticRegressionNumpy(lr= 0.01, epochs=2000)

scratch.fit(X_train, y_train)
np_pred = scratch.predict(X_test) 

end_num = time.time()

np_time = end_num - start_num
        

print("Training Time Taken by Numpy Logistic:",np_time)
print("Accuracy of Numpy Logistic:",accuracy_score(y_test,np_pred))
print("Precision Score of Numpy Logistic:",precision_score(y_test,np_pred))
print("Recall Score of Numpy Logistic  :",recall_score(y_test,np_pred))
print("F1 Score Score of Numpy Logistic:",f1_score(y_test,np_pred))
print("ROC AUC  Score of Numpy Logistic:",roc_auc_score(y_test,np_pred))
print(f"Precision Recall Curve: {precision_recall_curve(y_test, np_pred)}")
print(f"Roc Curve: {roc_curve(y_test, np_pred)}")
print("Confusion matrix :", confusion_matrix(y_test, np_pred))

sns.histplot(data= df, x="gender" ,color = "blue", multiple="stack")
plt.xlabel("gender")
#plt.show()

sns.countplot( x="Churn",data=df,  hue="Churn", palette="Set2")
#plt.show()


sns.countplot(x="InternetService", data=df, color="Orange",
                                            edgecolor="black")
#plt.show()

sns.countplot(x="DeviceProtection", data=df, color="SkyBlue",
                                            edgecolor="black")
#plt.show()


sns.pairplot(df[["tenure", "MonthlyCharges", "TotalCharges", "Churn"]])
#plt.show()