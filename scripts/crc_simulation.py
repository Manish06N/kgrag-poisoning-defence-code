import numpy as np, sys, warnings
warnings.filterwarnings('ignore'); sys.path.insert(0,'.')
from kgrag import defend as D, learned as L

def load(split):
    recs=D.load_paths(split); X=L.features(recs); y=D.flat(recs,'poisoned').astype(int)
    qi=D.qindex(recs); gold=D.flat(recs,'gold_hit').astype(bool)
    return dict(recs=recs,X=X,y=y,qi=qi,gold=gold)
def pool(*sets):
    out=dict(recs=[],X=[],y=[],qi=[],gold=[]); off=0
    for s in sets:
        out['recs']+=s['recs']; out['X'].append(s['X']); out['y'].append(s['y']); out['gold'].append(s['gold']); out['qi'].append(s['qi']+off); off+=len(s['recs'])
    for k in ('X','y','qi','gold'): out[k]=np.concatenate(out[k]) if k!='X' else np.vstack(out[k])
    return out
def subset(P,qs):
    m=np.isin(P['qi'],qs); remap={q:i for i,q in enumerate(qs)}
    return dict(X=P['X'][m],y=P['y'][m],gold=P['gold'][m],qi=np.array([remap[q] for q in P['qi'][m]]),nq=len(qs))
GRID=np.concatenate([np.linspace(0,0.2,81)[1:],np.linspace(0.2,1,81)])  # candidate thresholds on the detector probability
def loss_matrix(S,s):
    """per answerable question: 1 if every clean gold-supporting path has score > lambda (removed), for each lambda"""
    rows=[]
    for q in range(S['nq']):
        m=(S['qi']==q)&S['gold']&(S['y']==0)
        if m.any(): rows.append((s[m][:,None]>GRID[None,:]).all(0).astype(float))   # all gold paths removed
    return np.array(rows)
def crc(Lm,alpha):
    n=len(Lm); risk=Lm.mean(0)
    ok=np.where(n/(n+1)*risk+1/(n+1)<=alpha)[0]
    return GRID[ok[0]] if len(ok) else None            # smallest threshold (most aggressive removal) meeting the bound
def fit(S): return L._model().fit(S['X'],S['y'])
def stats(S,s,thr):
    rem=s>thr; Lm=loss_matrix(S,s); k=np.searchsorted(GRID,thr)
    return dict(risk=float(loss_matrix(S,s)[:,min(np.searchsorted(GRID,thr),len(GRID)-1)].mean()) if len(Lm) else np.nan,
                poison_out=float(rem[S['y']==1].mean()),clean_out=float(rem[S['y']==0].mean()),
                left_per_q=float(((~rem)&(S['y']==1)).sum()/S['nq']))

paper=pool(load('dev'),load('test'))          # 300 dev + 500 test questions, paper attack
adapt=pool(load('ad_dev'),load('ad_test'))    # same questions, adaptive attacker
rng=np.random.default_rng(0)
print('=== A. does the guarantee hold? paper attack, 100 random splits (150 train / 250 calibration / 400 test questions) ===')
print('alpha   realised risk (mean over splits)   splits with risk<=alpha   poison removed   poison left/q   threshold(mean)')
res={a:[] for a in (0.02,0.05,0.10,0.20)}
for rep in range(100):
    q=rng.permutation(800); tr,ca,te=q[:150],q[150:400],q[400:]
    Str,Sca,Ste=subset(paper,tr),subset(paper,ca),subset(paper,te)
    m=fit(Str); sca,ste=m.predict_proba(Sca['X'])[:,1],m.predict_proba(Ste['X'])[:,1]
    Lca=loss_matrix(Sca,sca)
    for a in res:
        thr=crc(Lca,a)
        if thr is None: res[a].append((np.nan,)*5); continue
        st=stats(Ste,ste,thr); res[a].append((st['risk'],st['poison_out'],st['left_per_q'],st['clean_out'],thr))
for a,v in res.items():
    v=np.array(v); ok=~np.isnan(v[:,0])
    print('%.2f    %.3f                              %3.0f%%                     %5.1f%%         %5.2f          %.3f   (no valid threshold in %d splits)'%(a,np.nanmean(v[:,0]),100*np.mean(v[ok,0]<=a),100*np.nanmean(v[:,1]),np.nanmean(v[:,2]),np.nanmean(v[:,4]),(~ok).sum()))
print('\n=== B. attacker changes after calibration: calibrate on PAPER attack, test on ADAPTIVE attack (same questions, same detector) ===')
print('alpha   realised risk on adaptive attack   (paper-attack risk for comparison)   poison removed (adaptive)')
rb={a:[] for a in (0.05,0.10)}
for rep in range(50):
    q=rng.permutation(800); tr,ca,te=q[:150],q[150:400],q[400:]
    m=fit(subset(paper,tr)); Sca=subset(paper,ca); Lca=loss_matrix(Sca,m.predict_proba(Sca['X'])[:,1])
    Sp,Sa=subset(paper,te),subset(adapt,te)
    for a in rb:
        thr=crc(Lca,a)
        if thr is None: continue
        sp,sa=stats(Sp,m.predict_proba(Sp['X'])[:,1],thr),stats(Sa,m.predict_proba(Sa['X'])[:,1],thr)
        rb[a].append((sa['risk'],sp['risk'],sa['poison_out']))
for a,v in rb.items():
    v=np.array(v); print('%.2f    %.3f                              (%.3f)                              %5.1f%%'%(a,v[:,0].mean(),v[:,1].mean(),100*v[:,2].mean()))
print('\n=== C. calibrate on a MIX of both attacks, test on each ===')
rc={a:[] for a in (0.05,0.10)}
mix=pool(load('dev'),load('test'),load('ad_dev'),load('ad_test'))
for rep in range(50):
    q=rng.permutation(800); tr,ca,te=q[:150],q[150:400],q[400:]
    trm=np.concatenate([tr,tr+800]); cam=np.concatenate([ca,ca+800])
    m=fit(subset(mix,trm)); Sca=subset(mix,cam); Lca=loss_matrix(Sca,m.predict_proba(Sca['X'])[:,1])
    Sp,Sa=subset(paper,te),subset(adapt,te)
    for a in rc:
        thr=crc(Lca,a)
        if thr is None: continue
        sp,sa=stats(Sp,m.predict_proba(Sp['X'])[:,1],thr),stats(Sa,m.predict_proba(Sa['X'])[:,1],thr)
        rc[a].append((sp['risk'],sa['risk'],sp['poison_out'],sa['poison_out']))
for a,v in rc.items():
    v=np.array(v); print('%.2f    risk paper %.3f / adaptive %.3f    poison removed paper %.1f%% / adaptive %.1f%%'%(a,v[:,0].mean(),v[:,1].mean(),100*v[:,2].mean(),100*v[:,3].mean()))
