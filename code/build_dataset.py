"""
Step 1 - Build the analysis dataset from Basketball-Reference season-level advanced tables.

Input : data/nba_advanced_1980_2026.csv  (one row per player-season-team, retrieved 4 Sep 2026)
Output: results/player_seasons.csv       (deduplicated player-season table)
        results/players.csv              (one row per player with metadata and external variables)
        results/trajectories_<spec>.npz  (padded calendar-axis PER matrices + masks, per specification)
        results/sample_flow.csv          (exclusion flow for every specification)

Specifications (sample-restriction variants) are defined in SPECS and reused by the
sensitivity analyses.  The primary specification is 'primary'.
"""
import numpy as np, pandas as pd, json, os, re
os.makedirs('results', exist_ok=True)

RAW = 'data/nba_advanced_1980_2026.csv'

def load_player_seasons():
    d = pd.read_csv(RAW)
    # Players who changed team within a season have one aggregate row (2TM/3TM/...) plus one row
    # per team.  Keep the aggregate row only.
    d['multi'] = d['team'].astype(str).str.contains('TM')
    d = (d.sort_values(['season_end', 'player_id', 'multi'], ascending=[True, True, False])
           .drop_duplicates(['season_end', 'player_id']).drop(columns='multi'))
    d = d[d['per'].notna()].copy()            # 50 player-seasons without PER (0 minutes)
    d['awards'] = d['awards'].fillna('')
    d['all_star'] = d['awards'].str.split(';').apply(lambda a: 'AS' in a)
    d['all_nba'] = d['awards'].str.contains(r'NBA[123]')
    d['mvp_top5'] = d['awards'].str.extract(r'MVP-(\d+)')[0].astype(float).le(5).fillna(False)
    return d.sort_values(['player_id', 'season_end']).reset_index(drop=True)

# ---------------------------------------------------------------------------------------------------
# True first NBA season. The Basketball-Reference download starts in 1979-80, so a player's first season
# in the download is his NBA debut only if he had not played before. Earlier NBA seasons are taken from
# the NBA's own season records (nba_api, data/nbaapi_player_season_per_with_draft.csv, 1951-52 onwards),
# matched by normalised name; when several NBA players share a name, the one whose seasons overlap the
# Basketball-Reference seasons most is used. Players absent from the NBA records were checked by hand.
MANUAL_FIRST_SEASON = {'Tiny Archibald': 1971, 'World B. Free': 1976, 'Joe Hassett': 1978, 'Gar Heard': 1971,
                       'Jo Jo White': 1970, 'Michael Ray Richardson': 1979, 'Billy Ray Bates': 1980,
                       'Dave Greenwood': 1980, 'Cliff Robinson': 1980}

def _norm(name):
    from unidecode import unidecode
    s = unidecode(str(name)).lower(); s = re.sub(r"[^a-z ]", "", s); return re.sub(r"\s+", " ", s).strip()

_CACHE = {}
def true_first_season(ps, return_ids=False):
    if 'first' in _CACHE: return (_CACHE['first'], _CACHE['ids']) if return_ids else _CACHE['first']
    na = pd.read_csv('data/nbaapi_player_season_per_with_draft.csv', usecols=['SEASON_ID', 'PLAYER_ID', 'FULL_NAME'])
    na['season_end'] = na['SEASON_ID'].str[:4].astype(int) + 1
    na['key'] = na['FULL_NAME'].map(_norm)
    seasons_by_id = na.groupby('PLAYER_ID')['season_end'].apply(set)
    ids_by_key = na.groupby('key')['PLAYER_ID'].apply(lambda x: sorted(set(x)))
    out = {}; ids = {}
    for pid, g in ps.groupby('player_id'):
        name = g['name'].iloc[0]; own = set(g['season_end'])
        if name in MANUAL_FIRST_SEASON:
            out[pid] = MANUAL_FIRST_SEASON[name]; ids[pid] = None; continue
        cands = ids_by_key.get(_norm(name), [])
        best, best_ov = None, 0
        for c in cands:
            ov = len(seasons_by_id[c] & own)
            if ov > best_ov: best, best_ov = c, ov
        first_bbref = min(own); ids[pid] = best
        out[pid] = min(first_bbref, min(seasons_by_id[best])) if best is not None else first_bbref
    _CACHE['first'], _CACHE['ids'] = pd.Series(out), pd.Series(ids)
    return (_CACHE['first'], _CACHE['ids']) if return_ids else _CACHE['first']

SPECS = {
    # name: dict(min_mp, min_seasons, min_games, per_cap, debut_min, debut_max)
    'primary':       dict(min_mp=100, min_seasons=2, min_games=42, per_cap=None, debut=(1980, 2011)),
    'original2025':  dict(min_mp=0,   min_seasons=2, min_games=42, per_cap=45,   debut=(1980, 2011)),
    'mp250':         dict(min_mp=250, min_seasons=2, min_games=42, per_cap=None, debut=(1980, 2011)),
    'nomp_cap45':    dict(min_mp=0,   min_seasons=2, min_games=42, per_cap=45,   debut=(1980, 2011)),
    'min3seasons':   dict(min_mp=100, min_seasons=3, min_games=42, per_cap=None, debut=(1980, 2011)),
    'nogames':       dict(min_mp=100, min_seasons=2, min_games=0,  per_cap=None, debut=(1980, 2011)),
    'debut1980_2005':dict(min_mp=100, min_seasons=2, min_games=42, per_cap=None, debut=(1980, 2006)),
}

def build(spec_name, ps):
    s = SPECS[spec_name]
    flow = []
    debut = true_first_season(ps)          # first NBA season, including seasons before 1979-80
    pid_window = debut[(debut >= s['debut'][0]) & (debut <= s['debut'][1])].index
    flow.append(('debut season in window', len(pid_window)))
    x = ps[ps['player_id'].isin(pid_window)].copy()
    # A season counts as a valid observation only if the player logged >= min_mp minutes
    x['valid'] = x['mp'] >= s['min_mp']
    if s['per_cap'] is not None:
        x.loc[x['per'] > s['per_cap'], 'valid'] = False
    games = x.groupby('player_id')['g'].sum()
    keep = games[games >= s['min_games']].index
    x = x[x['player_id'].isin(keep)]
    flow.append(('>= %d career games' % s['min_games'], x['player_id'].nunique()))
    nvalid = x[x['valid']].groupby('player_id').size()
    keep = nvalid[nvalid >= s['min_seasons']].index
    x = x[x['player_id'].isin(keep)]
    flow.append(('>= %d valid seasons' % s['min_seasons'], x['player_id'].nunique()))

    # Calendar-axis trajectories: from first valid season to last valid season, one slot per season
    players, seqs, masks, meta = [], [], [], []
    for pid, g in x.groupby('player_id'):
        gv = g[g['valid']]
        y0, y1 = gv['season_end'].min(), gv['season_end'].max()
        L = y1 - y0 + 1
        seq = np.zeros(L); m = np.zeros(L, bool)
        for _, r in gv.iterrows():
            seq[r['season_end'] - y0] = r['per']; m[r['season_end'] - y0] = True
        players.append(pid); seqs.append(seq); masks.append(m)
        meta.append(dict(player_id=pid, name=g['name'].iloc[0], debut=y0, last=y1,
                         span=L, n_valid=int(m.sum()), n_gaps=int(L - m.sum()),
                         career_games=int(g['g'].sum()), career_mp=int(g['mp'].sum()),
                         mean_per=float(gv['per'].mean()), peak_per=float(gv['per'].max()),
                         sd_per=float(gv['per'].std(ddof=0)),
                         min_mp_valid=float(gv['mp'].min()),
                         pos_mode=gv['pos'].mode().iloc[0],
                         all_star_n=int(gv['all_star'].sum()), all_nba_n=int(gv['all_nba'].sum()),
                         mvp_top5_n=int(gv['mvp_top5'].sum()),
                         mean_bpm=float(gv['bpm'].mean()), mean_ws=float(gv['ws'].mean()),
                         sum_ws=float(gv['ws'].sum()), sum_vorp=float(gv['vorp'].sum()),
                         mean_usg=float(gv['usg_pct'].mean()), mean_ts=float(gv['ts_pct'].mean()),
                         debut_age=float(gv['age'].iloc[0]) if pd.notna(gv['age'].iloc[0]) else np.nan))
    T = max(len(s_) for s_ in seqs)
    X = np.zeros((len(seqs), T)); M = np.zeros((len(seqs), T), bool)
    for i, (s_, m_) in enumerate(zip(seqs, masks)):
        X[i, :len(s_)] = s_; M[i, :len(m_)] = m_
    meta = pd.DataFrame(meta)
    np.savez_compressed(f'results/trajectories_{spec_name}.npz', X=X, M=M, player_id=np.array(players))
    meta.to_csv(f'results/players_{spec_name}.csv', index=False)
    flow.append(('final players', len(meta)))
    flow.append(('valid PER observations', int(M.sum())))
    flow.append(('players with >=1 missing season inside span', int((meta['n_gaps'] > 0).sum())))
    return flow, meta

if __name__ == '__main__':
    ps = load_player_seasons()
    ps.to_csv('results/player_seasons.csv', index=False)
    print('player-seasons:', len(ps), 'players:', ps.player_id.nunique(),
          'seasons:', ps.season_end.min(), '-', ps.season_end.max())
    rows = []
    for name in SPECS:
        flow, meta = build(name, ps)
        for step, n in flow:
            rows.append(dict(spec=name, step=step, n=n))
        print(name, flow)
    pd.DataFrame(rows).to_csv('results/sample_flow.csv', index=False)
    # Draft status: the NBA draft record of the matched NBA player (nba_api person id); no record = undrafted
    first, ids = true_first_season(ps, return_ids=True)
    dh = pd.read_csv('data/nbaapi_draft_history.csv'); dh = dh[dh['DRAFT_TYPE'] == 'Draft'].drop_duplicates('PERSON_ID')
    dh = dh.set_index('PERSON_ID')
    rows = []
    for pid in ps['player_id'].unique():
        nid = ids.get(pid); r = dh.loc[nid] if (nid is not None and not pd.isna(nid) and nid in dh.index) else None
        rows.append(dict(player_id=pid, nba_id=nid, first_nba_season=int(first[pid]),
                         draft_year=(r['SEASON'] if r is not None else np.nan), pick=(r['OVERALL_PICK'] if r is not None else np.nan),
                         round=(r['ROUND_NUMBER'] if r is not None else np.nan)))
    pd.DataFrame(rows).to_csv('results/draft_match.csv', index=False)
    print('draft records matched:', int(pd.DataFrame(rows)['pick'].notna().sum()))
