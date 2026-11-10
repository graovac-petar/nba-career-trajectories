"""Step 0b - retrieve the NBA.com records used by build_dataset.py, with the nba_api package (pip install nba_api==1.11.4).

Writes
  data/nbaapi_player_season_per_with_draft.csv  one row per player and regular season (columns SEASON_ID, PLAYER_ID,
                                                 FULL_NAME are the ones build_dataset.py uses: they identify careers
                                                 that began before 1979-80)
  data/nbaapi_draft_history.csv                 the NBA draft history (DraftHistory endpoint)

The files used for the paper were retrieved in July 2026. They are not redistributed in this repository because of
NBA.com's terms of use; this script rebuilds them. It makes one request per player (about 5,000 requests, roughly an
hour with the pause below). Season records can change slightly as NBA.com revises its data.
NOTE: this script reconstructs the retrieval step for the public repository and has not been run in this exact form;
check the row counts it prints against the numbers in results/sample_flow.csv after running build_dataset.py.
"""
import time, os
import pandas as pd
from nba_api.stats.static import players
from nba_api.stats.endpoints import playercareerstats, drafthistory

os.makedirs('data', exist_ok=True)
PAUSE = 0.7

dh = drafthistory.DraftHistory(league_id='00').get_data_frames()[0]
dh.to_csv('data/nbaapi_draft_history.csv', index=False)
print('draft history rows:', len(dh))

rows = []
for i, p in enumerate(players.get_players()):
    for attempt in range(3):
        try:
            df = playercareerstats.PlayerCareerStats(player_id=p['id'], per_mode36='Totals').get_data_frames()[0]
            break
        except Exception:
            time.sleep(5 * (attempt + 1))
    else:
        print('failed:', p['id'], p['full_name']); continue
    if len(df):
        df = df[['SEASON_ID', 'PLAYER_ID', 'GP', 'MIN']].copy(); df['FULL_NAME'] = p['full_name']
        rows.append(df)
    time.sleep(PAUSE)
    if i % 250 == 0: print(i, 'players')
out = pd.concat(rows, ignore_index=True)
out = out.merge(dh[['PERSON_ID', 'SEASON', 'ROUND_NUMBER', 'ROUND_PICK', 'OVERALL_PICK']], left_on='PLAYER_ID', right_on='PERSON_ID', how='left')
out.to_csv('data/nbaapi_player_season_per_with_draft.csv', index=False)
print('player-season rows:', len(out))
