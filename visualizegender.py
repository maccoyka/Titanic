import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

female_color = "#FA0000"

train = pd.read_csv('Titanic/train.csv')
fig = plt.figure(figsize=(18,6))

# plot percent survived
plt.subplot2grid((3,4),(0,0))
train.Survived.value_counts(normalize=True).plot(kind="bar", alpha=.75)
plt.title("Survived")

# plot Men who survived
plt.subplot2grid((3,4),(0,1))
train.Survived[train.Sex == "male"].value_counts(normalize=True).plot(kind="bar", alpha=.75)
plt.title("Men Survived")

# plot Women who survived
plt.subplot2grid((3,4),(0,2))
train.Survived[train.Sex == "female"].value_counts(normalize=True).plot(kind="bar", alpha=.75, color=female_color)
plt.title("Women Survived")

# comparison of survivor by sex
plt.subplot2grid((3,4),(0,3))
train.Sex[train.Survived == 1].value_counts(normalize=True).plot(kind="bar", alpha=.75, color=[female_color, 'b'])
plt.title("Sex of Survived")


plt.subplot2grid((3,4), (1,0), colspan=4)
for x in [1,2,3]:
	# plotting by age and ['s mean filtering by class
	train.Survived[train.Pclass == x].plot(kind="kde")
plt.title("Class wrt Survived")
plt.legend(("1st","2nd","3rd"))

# rich man who survived
plt.subplot2grid((3,4),(2,0))
train.Survived[(train.Sex == "male") & (train.Pclass == 1)].value_counts(normalize=True).plot(kind="bar", alpha=.75)
plt.title("Rich Men Survived")

# poor men who survived
plt.subplot2grid((3,4),(2,1))
train.Survived[(train.Sex == "male") & (train.Pclass == 3)].value_counts(normalize=True).plot(kind="bar", alpha=.75)
plt.title("Poor Men Survived")

# rich women who survived
plt.subplot2grid((3,4),(2,2))
train.Survived[(train.Sex == "female") & (train.Pclass == 1)].value_counts(normalize=True).plot(kind="bar", alpha=.75)
plt.title("Rich Women Survived")

# poor women who survived
plt.subplot2grid((3,4),(2,3))
train.Survived[(train.Sex == "female") & (train.Pclass == 3)].value_counts(normalize=True).plot(kind="bar", alpha=.75)
plt.title("Poor Women Survived")



plt.show()


