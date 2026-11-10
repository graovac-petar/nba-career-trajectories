# Raw data (not included)

The raw tables are not redistributed here because the terms of use of Basketball-Reference (Sports Reference LLC)
and NBA.com do not allow republishing complete data tables. They can be rebuilt as follows:

| File | How to obtain it |
|---|---|
| `nba_advanced_1980_2026.csv` | run `code/00_scrape_bbref.js` in the browser console on basketball-reference.com (one request every 3.5 s, within the site's limit of 20 requests per minute); the paper used tables retrieved on 4 September 2026 |
| `nbaapi_player_season_per_with_draft.csv`, `nbaapi_draft_history.csv` | `python3 code/00_fetch_nba_api.py` (nba_api 1.11.4); the paper used records retrieved in July 2026 |

Then `python3 code/build_dataset.py` rebuilds `results/player_seasons.csv` and the analysis samples.

Everything downstream of these files (the analysis samples, PER trajectories, cluster labels and all other results in
`results/`) is included, so every number, table and figure of the paper can be regenerated without the raw tables
(see the main README).
