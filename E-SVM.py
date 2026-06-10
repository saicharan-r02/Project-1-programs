import pandas as pd
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

l=load_digits()
plt.gray() 

for i in range(6,12):
    plt.matshow(l.images[i],cmap='inferno') 
plt.show()    

print(dir(l))
print(l.feature_names)
print(l.target_names)
print(l.target)
print(l.data)

d=pd.DataFrame(l.data,l.target)
print(d)
d["target"]=l.target
print(d)

x=d.drop(d[["target"]],axis="columns")
y=d.target

from sklearn.model_selection import train_test_split

X_tr,X_t,y_tr,y_t=train_test_split(x,y,train_size=0.8)
print(len(X_tr))
print(len(X_t))

from sklearn.svm import SVC

m=SVC()
m.fit(X_tr,y_tr)
print(m.score(X_t,y_t))

m_c=SVC(C=10)
m_c.fit(X_tr,y_tr)
print("Regularization Accuracy score: \n",m_c.score(X_t,y_t))

m_g=SVC(gamma=0.001)
m_g.fit(X_tr,y_tr)
print("Gamma Accuracy score: \n",m_g.score(X_t,y_t))

m_k=SVC(kernel="rbf")
m_k.fit(X_tr,y_tr)
print("kernel of rbf Accuracy score: \n",m_k.score(X_t,y_t))

m_k=SVC(kernel="linear")
m_k.fit(X_tr,y_tr)
print("Kernel of linear Accuracy score: \n",m_k.score(X_t,y_t))
