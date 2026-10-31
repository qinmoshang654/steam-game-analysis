"""Independent audits against saved CSVs and the original Kaggle files."""
from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import silhouette_score

ROOT=Path(__file__).resolve().parent
out=ROOT/'results'
m=json.loads((out/'metrics.json').read_text(encoding='utf8'))
d=pd.read_csv(out/'clean_games.csv'); raw=pd.read_csv(ROOT/'data/raw/steam.csv')
splits=pd.read_csv(out/'splits.csv'); tags=pd.read_csv(ROOT/'data/raw/steamspy_tag_data.csv').set_index('appid')
assert len(d)==len(raw)==27075 and d.appid.is_unique
assert d.appid.tolist()==raw.appid.tolist()==splits.appid.tolist()
assert np.array_equal(d.reviews,raw.positive_ratings+raw.negative_ratings)
assert hashlib.sha256((ROOT/'data/raw/steam.csv').read_bytes()).hexdigest()==m['audit']['source_sha256']
for a,b in [('train','validation'),('train','test'),('validation','test')]:
    da=set(splits.loc[splits.split.eq(a),'developer'].str.lower().str.strip())-{'未知开发商'}
    db=set(splits.loc[splits.split.eq(b),'developer'].str.lower().str.strip())-{'未知开发商'}
    assert da.isdisjoint(db)
bundle=joblib.load(out/'market_cluster.joblib'); cols=bundle['features']; tr=d.split.eq('train')
assert np.allclose(bundle['scaler'].mean_,d.loc[tr,cols].mean(),rtol=1e-12)
x=bundle['scaler'].transform(d[cols]); assert np.array_equal(bundle['model'].predict(x),d.cluster)
cv=pd.read_csv(out/'cluster_validation.csv'); selected=cv[cv.validation_min_share.ge(.02)].sort_values(['silhouette','davies_bouldin'],ascending=[False,True]).iloc[0]
assert selected['method']==m['selected_cluster']['method'] and selected.k==m['selected_cluster']['k']
q=pd.read_csv(out/'retrieval_queries.csv'); ranks=pd.read_csv(out/'retrieval_test_ranks.csv')
train_ids=set(d.loc[tr,'appid']); test_queries=set(q.loc[q.split.eq('test'),'appid'])
assert set(ranks.candidate_appid)<=train_ids and set(ranks.query_appid)==test_queries
assert train_ids.isdisjoint(test_queries)
assert all((g['rank'].tolist()==list(range(1,11))) for _,g in ranks.groupby(['method','query_appid'],sort=False))
truth=np.array([int(tags.loc[row.candidate_appid,row.hidden_tag]>0) for row in ranks.itertuples()])
assert np.array_equal(truth,ranks.hidden_tag_match)
recomputed=[]
for method,g in ranks.groupby('method',sort=False):
    p=g.groupby('query_appid').hidden_tag_match.mean().mean()
    hit=g.groupby('query_appid').hidden_tag_match.max().mean()
    coverage=g.candidate_appid.nunique()/len(train_ids)
    rec=next(r for r in m['retrieval'] if r['split']=='test' and r['method']==method)
    assert np.isclose(p,rec['precision_at_10']) and np.isclose(hit,rec['hit_at_10']) and np.isclose(coverage,rec['catalog_coverage'])
    recomputed.append({'method':method,'precision_at_10':p,'hit_at_10':hit,'coverage':coverage})
val=[r for r in m['retrieval'] if r['split']=='validation']
assert max(val,key=lambda r:r['precision_at_10'])['method']==m['retrieval_selected_on_validation']
# Confirm the withheld tag is not directly present under another feature-field name.
for row in q.itertuples():
    query=d[d.appid.eq(row.appid)].iloc[0]
    forbidden={s.strip().lower().replace(' ','_') for field in ['genres','categories'] for s in str(query[field]).split(';')}
    assert row.hidden_tag not in forbidden and tags.loc[row.appid,row.hidden_tag]>0
result={'passed':True,'games':len(d),'developer_groups_disjoint':True,'scaler_train_only':True,
        'model_labels_reproduced':True,'selection_uses_validation_only':True,
        'test_retrieval_rows_checked':len(ranks),'retrieval_recomputed':recomputed}
(out/'audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps(result,ensure_ascii=False,indent=2))
