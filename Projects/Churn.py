import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time 
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
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
    roc_curve,
    average_precision_score
)



df = pd.read_csv("Projects/Churn.csv")


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


start = time.time()
model = LogisticRegression( class_weight="balanced",max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_prob_sk = model.predict_proba(X_test)[:,1]
end = time.time()

sk_time = end - start
recall_sk = recall_score(y_test, y_pred)
rocscore = roc_auc_score(y_test, y_prob_sk) 
accuracy = accuracy_score(y_test, y_pred)
precision_sk = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
precision_recall = precision_recall_curve(y_test, y_prob_sk)
print(f"Precision Recall Curve: {precision_recall}")
average_precision = average_precision_score(y_test, y_prob_sk)
print("------------------------ Confusion Matrix -------------------------------------------")
print(confusion_matrix(y_test, y_pred))
print("--------------------------------------------------------------------------------------")


#Numpy Logistic Regression

class LogisticRegressionNumpy:
    
    def __init__(self, lr=0.01, epochs=2000, class_weight=None):
        self.lr = lr
        self.epochs = epochs
        self.class_weight = class_weight        
    
    def sigmoid(self, z):
        return np.where(z>=0 , 1/(1+np.exp(-np.abs(z))), np.exp(-np.abs(z))/(1+np.exp(-np.abs(z))))
    
    
    def fit(self, x,y):
        
        x = np.array(x , dtype=float)
        y = np.array(y , dtype=float)
        
        samples, features = x.shape

        self.weights = np.zeros(features)
        self.bias = 0
        self.loss_history = []
        
        if self.class_weight == "balanced":
            n_pos = y.sum()
            n_neg = samples - n_pos
            sw = np.where(y ==1, samples/(2* n_pos), samples/(2*n_neg))
            
        else:
            sw = np.ones(samples)
        sw_sum = sw.sum()
        
        
        for i in range(self.epochs):
            linear = np.dot(x,self.weights)+ self.bias
            
            predicition = self.sigmoid(linear)
            
            p_safe = np.clip(predicition, 1e-15, 1-1e-15)
            
            loss = -np.sum(
                sw*(y*np.log(p_safe) + (1-y)* np.log(1-p_safe))
            )/ sw_sum 
            
            self.loss_history.append(loss)
            
            error = sw * (predicition - y)                         # Weighted residual
            dw = np.dot(x.T , error)/ sw_sum
            db = np.sum(error)/ sw_sum
            
            self.weights -= self.lr*dw
            self.bias  -= self.lr*db
            
            if i %500 ==0:
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
np_pred_prob = scratch.predict_probability(X_test)
np_time = end_num - start_num
        

accuracy_num = accuracy_score(y_test,np_pred)
precision_num = precision_score(y_test,np_pred)
recall_num = recall_score(y_test,np_pred)
f1_num = f1_score(y_test,np_pred)
roc_auc_num = roc_auc_score(y_test,np_pred_prob)
precision_recall_num = precision_recall_curve(y_test, np_pred_prob)
print(f"Precision Recall Curve: {precision_recall_num}")
average_precision_num = average_precision_score(y_test, np_pred_prob)
print("------------------------ Confusion Matrix -------------------------------------------")
print(confusion_matrix(y_test, np_pred))
print("--------------------------------------------------------------------------------------")


sns.histplot(data= df, x="gender" ,color = "#87cbf5", multiple="stack")
plt.title("Category Based Members")
plt.xlabel("Gender")
plt.ylabel("Total")
plt.show()

sns.countplot( x="Churn",data=df,  hue="Churn", palette="Set2")
plt.title("CHURN")
plt.xlabel("Churn (0=No , 1=Yes)")
plt.show()


sns.boxplot(x="Churn", y="tenure",data=df,  hue="Churn", legend=False)
plt.title("Churn VS Tenure")
plt.xlabel("Churn(0=No , 1=Yes)")
plt.ylabel("Tenure")
plt.show()

sns.countplot(x="InternetService", data=df, color="#edb940",
                                            edgecolor="black")
plt.title("INTERNET SERVICE")
plt.xlabel("Services")
plt.ylabel("Total")
plt.show()

sns.countplot(x="DeviceProtection", data=df, color="#92f467e8",
                                            edgecolor="black")
plt.title("DEVICE PROTECTION")
plt.xlabel("Devices")
plt.show()


#Hyperparameter
warnings.filterwarnings("ignore", category=ConvergenceWarning)
model_1 = LogisticRegression(random_state=42)

param_grid = {
     'C': [0.001, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100],
    'solver': ['lbfgs', 'liblinear', 'saga'],
    'class_weight' : [None, 'balanced']
}

grid = GridSearchCV(
    estimator = model_1,
    param_grid =param_grid,
    cv= 5,
    scoring = 'f1',
    refit=True
)
start_grid = time.time()

grid.fit(X_train, y_train)

grid_model = grid.predict(X_test)

print("Best Parameters: ", grid.best_params_)
print("Best Accuracy: ", grid.best_score_)

best_model = grid.best_estimator_

y_pred_Grid = best_model.predict(X_test)
y_pred_prob_grid = best_model.predict_proba(X_test)[:,1]
end_grid = time.time()

time_grid = end_grid - start_grid

accuracy_grid = accuracy_score(y_test, y_pred_Grid)
precision_grid = precision_score(y_test, y_pred_Grid)
recall_grid = recall_score(y_test, y_pred_Grid)
f1_grid = f1_score(y_test, y_pred_Grid)
average_precision_grid = average_precision_score(y_test, y_pred_prob_grid)

precision_recall_grid = precision_recall_curve(y_test, y_pred_prob_grid)
print(f"Precision Recall Curve of Tuned Model: {precision_recall_grid}")
roc_auc_grid = roc_auc_score(y_test, y_pred_prob_grid)
print(f"Test Roc AUC Score of Tuned Model: {roc_auc_grid}")

corr = df.corr(numeric_only=True)
print(corr)

sns.heatmap(corr, annot=True)
plt.title("HeatMap of Numerical Columns")
plt.show(block=True)


cm = confusion_matrix(y_test, y_pred)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=["No Churn", "Churn"],
    yticklabels=["No Churn", "Churn"]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix Of SK Model")
plt.show()


precision, recall, thresholds = precision_recall_curve(
    y_test,
    y_prob_sk
)
plt.plot(recall, precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.show(block=True)


results = pd.DataFrame({
    "Models"            : ["SK Logistic", "Num Logistic", "Grid Model"],
    "Time"              : [sk_time, np_time, time_grid],
    "Recall"            : [recall_sk, recall_num , recall_grid],
    "Accuracy"          : [accuracy, accuracy_num, accuracy_grid ],
    "Precision"         : [precision_sk , precision_num , precision_grid ],
    "F1"                : [f1, f1_num, f1_grid],
    "ROC"               : [rocscore, roc_auc_num, roc_auc_grid],
    "Average Precision" : [average_precision, average_precision_num, average_precision_grid]
})

print("*********************************************** REPORT *******************************************************************************")

print(results)


customer_churn_probability = best_model.predict_proba(X_test)[:, 1]

def risk_level(probability):                                          #risk levels predictions
    if probability >= 0.70:
        return "High Risk"
    elif probability >= 0.40:
        return "Medium Risk"
    else :
        return "Low Risk"
    

risk = [risk_level(p) for p in customer_churn_probability]

def recommendations(risk):
    if risk == "High Risk":
        return "Immediate retention offer"
    elif risk == "Medium Risk":
        return "Monitor and Contact for further deals"
    else : 
        return "Normal Customer"
    
    
actions = [recommendations(r) for r in risk]



customer_risk_report = x_test.copy()

customer_risk_report["Churn Probability"]  = customer_churn_probability
customer_risk_report["Risk Level"]         = risk
customer_risk_report["Recommended Action"] = actions



customer_risk_report = customer_risk_report[
    ["customerID",
     "tenure",
     "MonthlyCharges",
     "TotalCharges",
     "Contract",
     "InternetService",
     "Churn Probability",
     "Risk Level",
     "Recommended Action"
    ]
]
##
#Sort the highest Risk LEvel
customer_risk_report = customer_risk_report.sort_values(
    by="Churn Probability",
    ascending = False
)

print("******************************** Customer Risk Report************************************************")
print(customer_risk_report.head(25))