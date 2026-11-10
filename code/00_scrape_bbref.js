// Retrieval of Basketball-Reference "Advanced" season tables (run in a browser console on basketball-reference.com).
// Retrieved 4 September 2026. Produces nba_advanced_1980_2026.csv (one row per player-season-team).
// Requests are spaced 3.5 s apart to respect the site's rate limits.
window.__adv = {};
window.__fetchSeason = async function (y) {
  const r = await fetch(`https://www.basketball-reference.com/leagues/NBA_${y}_advanced.html`);
  if (!r.ok) return 'HTTP ' + r.status;
  const doc = new DOMParser().parseFromString(await r.text(), 'text/html');
  const t = doc.querySelector('table#advanced'); if (!t) return 'NOTABLE';
  const cols = ['name_display','age','team_name_abbr','pos','games','games_started','mp','per','ts_pct','usg_pct','ws','bpm','vorp','awards'];
  const rows = [...t.querySelectorAll('tbody tr')].filter(r => !r.classList.contains('thead'));
  window.__adv[y] = rows.map(r => {
    const a = r.querySelector('td[data-stat="name_display"] a');
    const id = a ? a.getAttribute('href').split('/').pop().replace('.html', '') : '';
    return [y, id].concat(cols.map(c => { const td = r.querySelector(`[data-stat="${c}"]`); return td ? td.innerText.replace(/[\t\n]/g, ' ').replace(/,/g, ';') : ''; })).join(',');
  });
  return window.__adv[y].length;
};
(async () => {
  for (let y = 1980; y <= 2026; y++) { await window.__fetchSeason(y); await new Promise(r => setTimeout(r, 3500)); }
  const hdr = 'season_end,player_id,name,age,team,pos,g,gs,mp,per,ts_pct,usg_pct,ws,bpm,vorp,awards';
  const csv = [hdr].concat(Object.keys(window.__adv).sort().flatMap(k => window.__adv[k])).join('\n');
  const b = new Blob([csv], { type: 'text/csv' }); const a = document.createElement('a');
  a.href = URL.createObjectURL(b); a.download = 'nba_advanced_1980_2026.csv'; document.body.appendChild(a); a.click(); a.remove();
})();
