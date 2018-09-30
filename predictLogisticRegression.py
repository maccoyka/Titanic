import pandas as pd
import cleanDataFunction
from sklearn import linear_regression

train = pd.read_csv("Titanic/train.csv")
cleanDataFunction.clean_data(train)

target = train["Survived"].values 
features = train[["Pclass","Age","Sex","SibSp","Parch"]].values

classifier = linear_model.LogisticRegression()
classifier_ = classifier.fit(features, target)

print(classifier_score(features,target))


