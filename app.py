import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

data=pd.DataFrame({
"study_hours":[1,2,2,3,3,4,4,5,5,6,6,7,7,8,9,10],
"attendance":[45,50,55,60,62,65,68,70,72,75,78,80,83,86,90,95],
"previous_marks":[35,42,45,50,55,58,60,62,66,70,72,76,80,82,88,92],
"pass":[0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1]})
X=data[["study_hours","attendance","previous_marks"]];y=data["pass"]
a,b,c,d=train_test_split(X,y,test_size=.25,random_state=42,stratify=y)
model=RandomForestClassifier(n_estimators=100,random_state=42).fit(a,c)
print("Accuracy:",round(accuracy_score(d,model.predict(b))*100,2),"%")
hours=float(input("Study hours/day: "));att=float(input("Attendance %: "));marks=float(input("Previous marks %: "))
p=model.predict([[hours,att,marks]])[0];prob=model.predict_proba([[hours,att,marks]])[0][1]
print("Prediction:","PASS" if p else "AT RISK");print("Pass probability:",round(prob*100,1),"%")
