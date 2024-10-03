import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

# Carica il dataset
teams_df = pd.read_csv('../../Other/basketballPlayoffs/teams.csv')

# Seleziona le colonne di interesse
columns_of_interest = ['year', 'rank', 'won', 'name']
teams_df = teams_df[columns_of_interest]

# Divisione in caratteristiche (X) e target (y)
X = teams_df.drop(['won', 'rank', 'name'], axis=1)  # Usa tutte le colonne eccetto 'won', 'rank' e 'name'
y = teams_df['rank']  # Usa 'rank' come target

# Divisione in training set e test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Modello di Decision Tree
model = DecisionTreeRegressor()
model.fit(X_train, y_train)

# Predizione per tutto il dataset
teams_df['predicted_rank'] = model.predict(X)

# Calcola la media dei ranking previsti per ogni squadra
avg_predicted_ranks = teams_df.groupby('name')['predicted_rank'].mean().reset_index()

# Ordina per rank previsto medio e prendi i primi 8 team
top_teams = avg_predicted_ranks.nsmallest(8, 'predicted_rank')

# Mostra i team previsti
print(top_teams)
