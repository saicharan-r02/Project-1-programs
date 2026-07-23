import pandas as pd
from sklearn.linear_model import LinearRegression

#loading the dataset Car_P.csv file
d=pd.read_csv("Car_P.csv")
print(d)

#getting dummiies in dataset
dum=pd.get_dummies(d.CarModel,dtype=int)
print(dum)

#concat data and dummies
m=pd.concat([d,dum],axis="columns")
print(m)

#dropping CarModel 
f=m.drop(["CarModel"],axis="columns")
print(f)

#dropping Mercedez Benz C class
f=f.drop(["Mercedez Benz C class"],axis="columns")
print(f)

#dropping Sell Price($) columns
X=f.drop(["Sell Price($)"],axis="columns")
print(X)

#dropping Mileage,Age(yrs),Audi A5,BMW X5 columns 
y=f.drop(["Mileage" , "Age(yrs)" , "Audi A5" ,"BMW X5"],axis="columns")
print(y)

#LinearRegression model training
md=LinearRegression()
md.fit(X,y)
md.predict(X)

print(md.score(X,y))
print(md.feature_names_in_)
print(md.predict(pd.DataFrame([[20000,9,0,1]],columns=['Mileage','Age(yrs)','Audi A5','BMW X5'])))
print(md.predict(pd.DataFrame([[49300,4,1,0]],columns=['Mileage','Age(yrs)','Audi A5','BMW X5'])))

print(md.predict(pd.DataFrame([[29438,9,0,0]],columns=['Mileage','Age(yrs)','Audi A5','BMW X5'])))

print(md.predict(pd.DataFrame([[38652,23,1,0]],columns=['Mileage','Age(yrs)','Audi A5','BMW X5'])))
