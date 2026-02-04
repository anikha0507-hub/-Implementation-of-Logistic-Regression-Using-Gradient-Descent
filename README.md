# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load and preprocess the placement dataset.
2. Initialize weights and bias and apply feature scaling.
3. Train Logistic Regression using Gradient Descent.
4. Predict placement status and evaluate model performance.

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: 25009524
RegisterNumber: Anikha Pillai
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data = pd.read_csv("C:\Users\acer\Downloads\Placement_Data (1).csv")
data.head()
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

categorical_cols = [
    'gender', 'ssc_b', 'hsc_b', 'hsc_s',
    'degree_t', 'workex', 'specialisation', 'status'
]

for col in categorical_cols:
    data[col] = le.fit_transform(data[col])

data.head()
X = data.drop(['status', 'salary'], axis=1).values
y = data['status'].values.reshape(-1, 1)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X = scaler.fit_transform(X)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
m, n = X_train.shape   # samples, features
w = np.zeros((n, 1))   # weights
b = 0                 # bias

alpha = 0.01           # learning rate
iterations = 3000      # number of iterations
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
losses = []

for i in range(iterations):
    z = np.dot(X_train, w) + b
    y_hat = sigmoid(z)
    
    dw = (1/m) * np.dot(X_train.T, (y_hat - y_train))
    db = (1/m) * np.sum(y_hat - y_train)
    
    w -= alpha * dw
    b -= alpha * db
    
    loss = -(1/m) * np.sum(
        y_train * np.log(y_hat + 1e-9) + 
        (1 - y_train) * np.log(1 - y_hat + 1e-9)
    )
    losses.append(loss)

print("Training completed")
plt.plot(losses)
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.title("Loss vs Iterations (Gradient Descent)")
plt.show()
def predict(X):
    z = np.dot(X, w) + b
    y_pred = sigmoid(z)
    return (y_pred >= 0.5).astype(int)
y_pred = predict(X_test)

accuracy = np.mean(y_pred == y_test)
print("Accuracy:", accuracy)
from sklearn.metrics import confusion_matrix, classification_report

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

```

## Output:
<img width="818" height="552" alt="image" src="https://github.com/user-attachments/assets/907df00f-4b88-4bac-b623-009fb22e5b8f" />
<img width="700" height="318" alt="image" src="https://github.com/user-attachments/assets/efa43d54-4655-4ec6-9954-28ada570ccf7" />




## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

