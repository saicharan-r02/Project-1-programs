import pandas as pd
import warnings
warnings.filterwarnings("ignore")

d=pd.read_csv("Salaries.csv")
print(d)

n=d.drop(d[["salary_more_then_100k"]],axis="columns")
t=d["salary_more_then_100k"]

from sklearn.preprocessing import LabelEncoder

l_c=LabelEncoder()
l_j=LabelEncoder()
l_d=LabelEncoder()

n["company_N"]=l_c.fit_transform(n["company"])
n["job_N"]=l_j.fit_transform(n["job"])
n["degree_N"]=l_d.fit_transform(n["degree"])
print("\n",n)

n=n.drop(n[["company","job","degree"]],axis="columns")
print("\n",n)
print("\n",t)

from sklearn import tree

m=tree.DecisionTreeClassifier()
m.fit(n,t)

print(m.score(n,t))
print(m.predict([[2,1,0]]))
print(m.predict([[1,1,1]]))
print(m.predict([[0,0,0]]))
print(m.predict([[0,1,0]]))
print(m.predict([[1,1,0]]))
print(m.predict([[2,2,2]]))
