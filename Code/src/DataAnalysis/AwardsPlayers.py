import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

awards_players_dataSet = pd.read_csv('../../Other/basketballPlayoffs/awards_players.csv')

# general info about the dataset
print(awards_players_dataSet.info())

# verify if there are some missing values
print(awards_players_dataSet.isnull().sum())

# description
print(awards_players_dataSet.describe())

# Types of data for each column
print(awards_players_dataSet.dtypes)

# Filter the players that have won some award
filtered_df = awards_players_dataSet[awards_players_dataSet['award'] == ' ']
print(filtered_df)


# visualize the dataset
plt.figure(figsize=(21, 17))
sns.countplot(data=awards_players_dataSet, x='award')
plt.title('Distribuzione dei Premi')
plt.xticks(rotation=45)
plt.show()

