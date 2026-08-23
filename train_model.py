"""
Reproduces the notebook's data pipeline exactly to train and pickle the
win-probability model. Run once locally after placing matches.csv and
deliveries.csv.zip in this folder.
"""
import pickle

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

match = pd.read_csv('matches.csv')
delivery = pd.read_csv('deliveries.csv.zip')

total_score_df = delivery.groupby(['match_id', 'inning']).sum()['total_runs'].reset_index()
total_score_df = total_score_df[total_score_df['inning'] == 1]

match_df = match.merge(total_score_df[['match_id', 'total_runs']], left_on='id', right_on='match_id')

teams = [
    'Sunrisers Hyderabad',
    'Mumbai Indians',
    'Royal Challengers Bangalore',
    'Kolkata Knight Riders',
    'Kings XI Punjab',
    'Chennai Super Kings',
    'Rajasthan Royals',
    'Delhi Capitals'
]
match_df['team1'] = match_df['team1'].str.replace('Delhi Daredevils', 'Delhi Capitals')
match_df['team2'] = match_df['team2'].str.replace('Delhi Daredevils', 'Delhi Capitals')
match_df['team1'] = match_df['team1'].str.replace('Deccan Chargers', 'Sunrisers Hyderabad')
match_df['team2'] = match_df['team2'].str.replace('Deccan Chargers', 'Sunrisers Hyderabad')

match_df = match_df[match_df['team1'].isin(teams)]
match_df = match_df[match_df['team2'].isin(teams)]

match_df = match_df[match_df['dl_applied'] == 0]
match_df = match_df[['match_id', 'city', 'winner', 'total_runs']]
delivery_df = match_df.merge(delivery, on='match_id')
delivery_df = delivery_df[delivery_df['inning'] == 2]

delivery_df['current_score'] = delivery_df.groupby('match_id')['total_runs_y'].cumsum()
delivery_df['runs_left'] = delivery_df['total_runs_x'] - delivery_df['current_score']
delivery_df['balls_left'] = 126 - (delivery_df['over'] * 6 + delivery_df['ball'])

delivery_df['player_dismissed'] = delivery_df['dismissal_kind'].apply(lambda x: 1 if pd.notnull(x) else 0)
delivery_df['wickets'] = 10 - delivery_df.groupby('match_id')['player_dismissed'].cumsum()

delivery_df['crr'] = (delivery_df['current_score'] * 6) / (120 - delivery_df['balls_left'])
delivery_df['rrr'] = (delivery_df['runs_left'] * 6) / delivery_df['balls_left']


def result(row):
    return 1 if row['batting_team'] == row['winner'] else 0


delivery_df['result'] = delivery_df.apply(result, axis=1)

final_df = delivery_df[['batting_team', 'bowling_team', 'city', 'runs_left', 'balls_left',
                         'wickets', 'total_runs_x', 'crr', 'rrr', 'result']]
final_df = final_df.sample(final_df.shape[0])

final_df.dropna(inplace=True)
final_df = final_df[final_df['balls_left'] != 0]
X = final_df.iloc[:, :-1]
y = final_df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

trf = ColumnTransformer(
    [('trf', OneHotEncoder(sparse_output=False, drop='first'), ['batting_team', 'bowling_team', 'city'])],
    remainder='passthrough'
)

pipe = Pipeline(steps=[
    ('step1', trf),
    ('step2', LogisticRegression(solver='liblinear'))
])

pipe.fit(X_train, y_train)

y_pred = pipe.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Test accuracy: {acc:.4f}")

# Pull the *exact* categories the fitted OneHotEncoder knows about, so the
# website's dropdowns can never send a team/city value that would blow up
# predict_proba with an "unknown category" error.
encoder = pipe.named_steps['step1'].named_transformers_['trf']
batting_teams, _bowling_teams, cities = [list(c) for c in encoder.categories_]
# batting_team and bowling_team share the same category set (encoded from
# the same column list), so batting_teams covers both.

import json
with open('teams_cities.json', 'w') as f:
    json.dump({'teams': sorted(batting_teams), 'cities': sorted(cities)}, f, indent=2)

print("Teams:", sorted(batting_teams))
print("Cities:", sorted(cities))

pickle.dump(pipe, open('pipe.pkl', 'wb'))
print("Saved pipe.pkl and teams_cities.json")
