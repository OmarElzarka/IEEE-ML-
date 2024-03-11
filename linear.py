import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression




dataset = pd.read_csv(r"C:\Users\Omar\Downloads\Data\data.csv")

X=dataset.drop(labels=['Bankrupt?'], axis=1)
y=dataset['Bankrupt?']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model = LinearRegression()
model.fit(X_train, y_train)

log_reg=LogisticRegression(max_iter=1000)
log_reg.fit(X_train,y_train)
