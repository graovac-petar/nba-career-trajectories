"""
Step 2 - Transformer autoencoder for variable-length PER trajectories.

All training hyper-parameters are fixed here (see CONFIG) and reported in the manuscript.
Usage:  python code/autoencoder.py --spec primary --latent 64 --seed 0
Writes: results/ae_<spec>_d<latent>_s<seed>.npz  (embeddings, reconstructions, split indices)
        results/ae_<spec>_d<latent>_s<seed>.json  (training log + reconstruction metrics)
"""
import argparse, json, time, os, math
import numpy as np, torch, torch.nn as nn

CONFIG = dict(
    d_model=64, nhead=4, enc_layers=2, dec_layers=2, ff=128, dropout=0.1,
    optimizer='AdamW', lr=1e-3, weight_decay=1e-4, batch_size=64,
    max_epochs=400, patience=40, lr_patience=15, lr_factor=0.5, grad_clip=1.0,
    val_fraction=0.20, split_seed=2026,
)

class TransformerAE(nn.Module):
    """Encoder: [CLS] + season tokens -> self-attention; latent = linear projection of CLS state.
       Decoder: learned positional queries attend (cross-attention) to the latent vector."""
    def __init__(self, latent_dim, max_len, d_model=64, nhead=4, enc_layers=2, dec_layers=2, ff=128, dropout=0.1):
        super().__init__()
        self.d_model, self.max_len = d_model, max_len
        self.input_proj = nn.Linear(2, d_model)              # (PER_z, observed flag)
        self.enc_pos = nn.Embedding(max_len + 1, d_model)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        enc = nn.TransformerEncoderLayer(d_model, nhead, ff, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, enc_layers)
        self.to_latent = nn.Linear(d_model, latent_dim)
        self.from_latent = nn.Linear(latent_dim, d_model)
        self.dec_pos = nn.Embedding(max_len, d_model)
        dec = nn.TransformerDecoderLayer(d_model, nhead, ff, dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(dec, dec_layers)
        self.out = nn.Linear(d_model, 1)
        self.reset_parameters()

    def reset_parameters(self):
        # Xavier-uniform for all matrices, zeros for biases; CLS token N(0, 0.02)
        for n, p in self.named_parameters():
            if p.dim() > 1: nn.init.xavier_uniform_(p)
            elif 'bias' in n: nn.init.zeros_(p)
        nn.init.normal_(self.cls, std=0.02)

    def encode(self, x, obs):
        B, T = x.shape
        tok = self.input_proj(torch.stack([x, obs.float()], -1))
        tok = tok + self.enc_pos(torch.arange(1, T + 1, device=x.device)).unsqueeze(0)
        tok = torch.cat([self.cls.expand(B, 1, -1), tok], 1)
        pad = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=x.device), ~obs], 1)
        h = self.encoder(tok, src_key_padding_mask=pad)
        return self.to_latent(h[:, 0])

    def decode(self, z, T):
        B = z.shape[0]
        q = self.dec_pos(torch.arange(T, device=z.device)).unsqueeze(0).expand(B, T, -1)
        mem = self.from_latent(z).unsqueeze(1)
        return self.out(self.decoder(q, mem)).squeeze(-1)

    def forward(self, x, obs):
        z = self.encode(x, obs)
        return self.decode(z, x.shape[1]), z

def masked_mse(yhat, y, m):
    return ((yhat - y) ** 2 * m).sum() / m.sum()

def make_split(n, lengths, val_fraction, seed):
    """Stratified (by career-length tertile) random split into development / held-out players."""
    rng = np.random.RandomState(seed)
    tert = np.digitize(lengths, np.quantile(lengths, [1/3, 2/3]))
    val = np.zeros(n, bool)
    for t in np.unique(tert):
        idx = np.where(tert == t)[0]; rng.shuffle(idx)
        val[idx[:int(round(len(idx) * val_fraction))]] = True
    return ~val, val

def train(spec, latent, seed, cfg=CONFIG, verbose=True, tag=None, three_way=False):
    torch.manual_seed(seed); np.random.seed(seed)
    torch.use_deterministic_algorithms(True)
    d = np.load(f'results/trajectories_{spec}.npz')
    X, M = d['X'], d['M']
    n, T = X.shape
    lengths = M.sum(1)
    tr, va = make_split(n, lengths, cfg['val_fraction'], cfg['split_seed'])
    te = None
    if three_way:
        # 70/10/20: the 20 % test players are never used for training or early stopping; early stopping uses a
        # 10 % validation subset carved out of the development players (same stratified procedure, seed+1)
        te = va.copy(); dev_idx = np.where(tr)[0]
        tr2, va2 = make_split(len(dev_idx), lengths[dev_idx], 0.125, cfg['split_seed'] + 1)
        tr = np.zeros(n, bool); va = np.zeros(n, bool); tr[dev_idx[tr2]] = True; va[dev_idx[va2]] = True
    # z-scoring with development-set statistics (valid observations only)
    mu, sd = X[tr][M[tr]].mean(), X[tr][M[tr]].std()
    Xz = np.where(M, (X - mu) / sd, 0.0)
    Xt = torch.tensor(Xz, dtype=torch.float32); Mt = torch.tensor(M)
    model = TransformerAE(latent, T, cfg['d_model'], cfg['nhead'], cfg['enc_layers'], cfg['dec_layers'], cfg['ff'], cfg['dropout'])
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=cfg['lr_factor'], patience=cfg['lr_patience'])
    tr_idx = np.where(tr)[0]; va_idx = np.where(va)[0]
    g = torch.Generator().manual_seed(seed)
    best, best_state, best_epoch, wait, log = np.inf, None, -1, 0, []
    t0 = time.time()
    for ep in range(cfg['max_epochs']):
        model.train(); perm = tr_idx[torch.randperm(len(tr_idx), generator=g).numpy()]
        tl = 0.0
        for i in range(0, len(perm), cfg['batch_size']):
            b = perm[i:i + cfg['batch_size']]
            x, m = Xt[b], Mt[b]
            opt.zero_grad()
            yhat, _ = model(x, m)
            loss = masked_mse(yhat, x, m)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'])
            opt.step(); tl += loss.item() * len(b)
        model.eval()
        with torch.no_grad():
            yv, _ = model(Xt[va_idx], Mt[va_idx]); vl = masked_mse(yv, Xt[va_idx], Mt[va_idx]).item()
        sched.step(vl)
        log.append(dict(epoch=ep + 1, train_mse_z=tl / len(perm), val_mse_z=vl, lr=opt.param_groups[0]['lr']))
        if vl < best - 1e-5:
            best, best_epoch, wait = vl, ep + 1, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= cfg['patience']: break
        if verbose and (ep % 25 == 0): print(f'ep {ep+1} train {tl/len(perm):.4f} val {vl:.4f}')
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        Yz, Z = model(Xt, Mt)
    Y = Yz.numpy() * sd + mu
    res = (Y - X) ** 2
    def rmse(idx): return float(np.sqrt(res[idx][M[idx]].mean()))
    # Baselines for reconstruction (all in PER units, evaluated on the same valid observations)
    def base_rmse(idx, kind):
        out = []
        for i in idx:
            y = X[i][M[i]]
            if kind == 'global_mean': pred = np.full_like(y, mu)
            elif kind == 'player_mean': pred = np.full_like(y, y.mean())
            elif kind == 'player_linear':
                t = np.where(M[i])[0]; A = np.vstack([np.ones_like(t), t]).T
                pred = A @ np.linalg.lstsq(A, y, rcond=None)[0]
            out.append(((pred - y) ** 2).sum())
        return float(np.sqrt(sum(out) / M[idx].sum()))
    metrics = dict(spec=spec, latent=latent, seed=seed, n_players=int(n), T=int(T), n_train=int(tr.sum()), n_val=int(va.sum()),
                   mu=float(mu), sd=float(sd), best_epoch=best_epoch, epochs_run=len(log), train_seconds=time.time() - t0,
                   rmse_train=rmse(tr_idx), rmse_val=rmse(va_idx), rmse_all=rmse(np.arange(n)),
                   rmse_test=(rmse(np.where(te)[0]) if te is not None else None), n_test=(int(te.sum()) if te is not None else None),
                   mse_val=rmse(va_idx) ** 2,
                   baseline_global_mean_val=base_rmse(va_idx, 'global_mean'),
                   baseline_player_mean_val=base_rmse(va_idx, 'player_mean'),
                   baseline_player_linear_val=base_rmse(va_idx, 'player_linear'),
                   n_params=int(sum(p.numel() for p in model.parameters())), config=cfg)
    tag = tag or f'{spec}_d{latent}_s{seed}'
    np.savez_compressed(f'results/ae_{tag}.npz', Z=Z.numpy(), Y=Y, train=tr, val=va, test=(te if te is not None else np.zeros(n, bool)), player_id=d['player_id'])
    json.dump(dict(metrics=metrics, log=log), open(f'results/ae_{tag}.json', 'w'), indent=1)
    torch.save(model.state_dict(), f'results/ae_{tag}.pt')
    if verbose: print(json.dumps({k: v for k, v in metrics.items() if k != 'config'}, indent=1))
    return metrics

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--spec', default='primary'); ap.add_argument('--latent', type=int, default=64)
    ap.add_argument('--seed', type=int, default=0); ap.add_argument('--three_way', action='store_true')
    a = ap.parse_args()
    torch.set_num_threads(2)
    train(a.spec, a.latent, a.seed, tag=(f'{a.spec}_d{a.latent}_s{a.seed}_3way' if a.three_way else None), three_way=a.three_way)
