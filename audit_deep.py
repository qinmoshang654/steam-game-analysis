"""Independent recomputation of the saved exploratory extension outputs."""
from pathlib import Path
import json,hashlib
import pandas as pd
import numpy as np
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/deep'

def main():
    protocol=json.loads((OUT/'protocol.json').read_text(encoding='utf8'))
    for f,digest in protocol['input_sha256'].items():
        assert hashlib.sha256((ROOT/'data/raw'/f).read_bytes()).hexdigest()==digest
    d=pd.read_csv(ROOT/'results/clean_games.csv').set_index('appid')
    tags=pd.read_csv(ROOT/'data/raw/steamspy_tag_data.csv').set_index('appid')
    ranks=pd.read_csv(OUT/'test_ranks.csv')
    assert len(ranks)==50000 and not ranks.duplicated(['method','query_appid','rank']).any()
    assert ranks.groupby(['method','query_appid']).size().eq(10).all()
    assert ranks.candidate_appid.ne(ranks.query_appid).all()
    assert d.loc[ranks.candidate_appid,'split'].eq('train').all()
    assert d.loc[ranks.query_appid,'split'].eq('test').all()
    raw=tags.loc[ranks.candidate_appid].to_numpy()
    col=tags.columns.get_indexer(ranks.hidden_tag)
    matches=(raw[np.arange(len(ranks)),col]>0).astype(int)
    assert np.array_equal(matches,ranks['match'].to_numpy())
    metrics=pd.read_csv(OUT/'retrieval_metrics.csv')
    recomputed=[]
    for name,rows in ranks.groupby('method',sort=False):
        per=rows.groupby('query_appid')['match'].agg(['mean','max'])
        row=metrics[(metrics.split=='test')&(metrics.method==name)].iloc[0]
        assert np.isclose(row.p10,per['mean'].mean())
        assert np.isclose(row.hit10,per['max'].mean())
        assert np.isclose(row.coverage,rows.candidate_appid.nunique()/d.split.eq('train').sum())
        recomputed.append({'method':name,'p10':float(row.p10),'hit10':float(row.hit10)})
    # Reconstruct original variants exactly; a changed preprocessing path must
    # never silently become a new baseline.
    original=pd.read_csv(ROOT/'results/retrieval_test_ranks.csv')
    for name in ['Popularity','Genres','CoreContent','FullTags']:
        a=original[original.method.eq(name)].sort_values(['query_appid','rank'])
        b=ranks[ranks.method.eq(name)].sort_values(['query_appid','rank'])
        assert np.array_equal(a.candidate_appid.to_numpy(),b.candidate_appid.to_numpy()),name
    chosen=json.loads((OUT/'selected_before_extended_test.json').read_text(encoding='utf8'))
    val=metrics[metrics.split.eq('validation')]
    assert chosen['method']==val.loc[val.p10.idxmax(),'method']
    weights=val[val.method.str.startswith('BlockTags')]
    assert chosen['weighted_method']==weights.loc[weights.p10.idxmax(),'method']
    q=pd.read_csv(OUT/'per_query_metrics.csv');train=d[d.split.eq('train')].index
    freq=(tags.loc[train]>0).mean()
    assert np.allclose(q.tag_prevalence,freq.loc[q.hidden_tag])
    for row in json.loads((OUT/'paired_developer_bootstrap.json').read_text(encoding='utf8')):
        assert row['wins']+row['ties']+row['losses']==500
    boot=pd.read_csv(OUT/'cluster_group_bootstrap.csv');assert len(boot)==30
    sparse=pd.read_csv(OUT/'tag_missingness.csv');assert len(sparse)==80
    assert np.isfinite(sparse[['p10','hit10']]).all().all()
    result={'passed':True,'test_rank_rows':len(ranks),'original_four_rankings_reproduced':True,
        'raw_relevance_recomputed':True,'candidate_partition_train_only':True,
        'extension_selection_validation_only':True,'exploratory_test_reuse_disclosed':True,
        'developer_bootstrap_refits':len(boot),'tag_missingness_conditions':len(sparse),
        'recomputed':recomputed}
    (OUT/'audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
