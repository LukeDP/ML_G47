import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Carica il dataset
teams_df = pd.read_csv('../../Other/basketballPlayoffs/teams.csv')

# Seleziona le colonne di interesse
columns_of_interest = ['year', 'rank', 'won', 'o_pts', 'd_pts', 'GP', 'playoff', 'name']  # Aggiungiamo 'playoff'
teams_df = teams_df[columns_of_interest]

# Filtra i dati per gli ultimi 10 anni
current_year = teams_df['year'].max()
recent_years_df = teams_df[teams_df['year'] >= current_year - 10]

# Contare quante volte una squadra ha partecipato ai playoff negli ultimi 10 anni
participation_count = recent_years_df[recent_years_df['playoff'] == 'Y'].groupby('name')['playoff'].count().reset_index()
participation_count.columns = ['name', 'playoff_participations']  # Rinominiamo la colonna

# Uniamo il conteggio delle partecipazioni al dataset originale
recent_years_df = pd.merge(recent_years_df, participation_count, on='name', how='left')

# Riempire i NaN con 0 per le squadre che non hanno partecipato
recent_years_df['playoff_participations'] = recent_years_df['playoff_participations'].fillna(0)

# Divisione in caratteristiche (X) e target (y)
X = recent_years_df.drop(['rank', 'name', 'playoff'], axis=1)  # Rimuoviamo 'rank', 'name' e 'playoff'
y = recent_years_df['rank']  # Usa 'rank' come target

# Divisione in training set e test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modello di regressione lineare
model = LinearRegression()
model.fit(X_train, y_train)

# Predizione per il prossimo anno
next_year_data = recent_years_df[recent_years_df['year'] == current_year].copy()
next_year_data.drop(['rank', 'name', 'playoff'], axis=1, inplace=True)  # Rimuoviamo le colonne non necessarie
predicted_ranks = model.predict(next_year_data)

# Aggiungi le previsioni ai dati
next_year_data['predicted_rank'] = predicted_ranks

# Unisci i risultati con i nomi dei team
predicted_teams = next_year_data.join(recent_years_df[['name']].reset_index(drop=True))

# Ordina per rank previsto e prendi i primi 8 team
top_teams = predicted_teams.nsmallest(8, 'predicted_rank')

# Visualizza i team previsti
print(top_teams[['name', 'predicted_rank', 'playoff_participations']])
