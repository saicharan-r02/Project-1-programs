import numpy as np
import pandas as pd

s=pd.Series([10,29,32,4,52,20])
print(s)

D={
    "Name": ["Sai","Ram","Rolex","Vikram"],
    "Marks":[4,67,89,90],
    "Age":[34,6,89,84]
}

s=pd.DataFrame(D)
print(s)
print(s.index)
print(s.info())
print(s.shape)
print(s.columns)
print(s.loc[0,"Age"])
print(s.head(3))
print(s.tail(3))
print(s.info())
print(s.describe())
print(s.loc[1])

d={
    "Name": ["Sai","Ram",np.nan,"Vikram"],
    "Marks":[4,np.nan,89,90],
    "Age":[34,6,89,np.nan]
}

s=pd.DataFrame(d)
print(s)
print(s.isnull())
print(s.fillna(2))
print(s.dropna())
print(s[s["Marks"]>20])
print(s[(s["Marks"]>15)& (s["Age"]>20)])
s["Passed"]=s["Marks"]>13
print(s)
print("Dropped column \n",s.drop("Passed",axis=1))
print("Rename columns method\n",s.rename(columns={"Marks":"Score"}))
print("Sorting method\n",s.sort_values(by="Marks",ascending=True))
print("Sorting method\n",s.sort_values(by="Age",ascending=False))
print("Group Method\n",s.groupby("Age")["Marks"].mean())
