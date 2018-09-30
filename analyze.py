import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

train = pd.read_csv('Titanic/train.csv')
fig = plt.figure(figsize=(18,6))

# plot percent survived
plt.subplot2grid((2,3),(0,0))
train.Survived.value_counts(normalize=True).plot(kind="bar", alpha=.75)
plt.title("Survived")

# plot ages who died
plt.subplot2grid((2,3),(0,1))
plt.scatter(train.Survived, train.Age, alpha=0.1)
plt.title("Age wrt Survived")

# plot percent of class on boat
plt.subplot2grid((2,3),(0,2))
train.Pclass.value_counts(normalize=True).plot(kind="bar", alpha=.75)
plt.title("Class")

# plot correlatiob of class and age to death
plt.subplot2grid((2,3), (1,0), colspan=2)
for x in [1,2,3]:
	# plotting by age and ['s mean filtering by class
	train.Age[train.Pclass == x].plot(kind="kde")
plt.title("Class wrt Age")
plt.legend(("1st","2nd","3rd"))

plt.subplot2grid((2,3),(1,2))
train.Embarked.value_counts(normalize=True).plot(kind="bar", alpha=.75)
plt.title("Embarked")

plt.show()

print(train.head())



