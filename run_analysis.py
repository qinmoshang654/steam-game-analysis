"""Reproducible Steam market clustering and masked-tag retrieval experiments.

Run from this directory with the packages in requirements.txt installed.
The original Kaggle CSV files must be in data/raw. No network calls are made.
"""
from pathlib import Path
import os
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parent/'results/.mpl-cache'))
import json, hashlib, platform, time, itertools
import numpy as np
import pandas as pd
import scipy
from scipy.stats import spearmanr
from scipy import sparse
import sklearn
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
RAW = ROOT/'data/raw'
OUT = ROOT/'results'; OUT.mkdir(exist_ok=True)
FIG = ROOT/'figures'; FIG.mkdir(exist_ok=True)
SEED = 42
ANCHOR = pd.Timestamp('2019-05-01')
KS = [2, 3, 4, 5, 6, 8]
METHODS = ['KMeans', 'MiniBatchKMeans', 'GaussianMixture']
MARKET_COLS = ['log_price', 'log_reviews', 'wilson_lower', 'log_release_age']
FEATURE_NAMES = ['价格对数', '评价数量对数', '好评率下界', '发行时长对数']
plt.rcParams.update({'font.family':['Times New Roman','SimSun'],
                     'font.size':12, 'axes.unicode_minus':False, 'figure.dpi':140,
                     'axes.spines.top':False, 'axes.spines.right':False,
                     'text.color':'black', 'axes.labelcolor':'black'})

def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=lambda v:
        v.item() if hasattr(v,'item') else str(v)), encoding='utf8')

def parts(v):
    return sorted(set(x.strip().lower().replace(' ','_') for x in str(v).split(';') if x.strip()))

def model_for(method, k, seed=SEED):
    if method == 'KMeans': return KMeans(k, n_init=20, max_iter=500, random_state=seed)
    if method == 'MiniBatchKMeans':
        return MiniBatchKMeans(k, n_init=20, batch_size=1024, max_iter=500, random_state=seed)
    return GaussianMixture(k, covariance_type='full', n_init=5, max_iter=500,
                           reg_covar=1e-5, random_state=seed)

def cluster_scores(x, labels):
    return {'silhouette':float(silhouette_score(x,labels)),
            'davies_bouldin':float(davies_bouldin_score(x,labels)),
            'min_cluster_share':float(pd.Series(labels).value_counts(normalize=True).min())}

def savefig(name):
    plt.tight_layout(); plt.savefig(FIG/name, bbox_inches='tight', dpi=180); plt.close()
    print('figure',name,flush=True)

def make_figures(d,x,comparison,ablations,retrieval,genres):
    labels=d.cluster.to_numpy()
    # Figures are generated from saved experimental quantities.
    fig,ax=plt.subplots(1,2,figsize=(7.2,2.8))
    ax[0].hist(d.loc[~d.free,'price'],bins=np.geomspace(.01,max(d.price)+1,36),color='#737373',edgecolor='white'); ax[0].set_xscale('log'); ax[0].xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda value,pos: f'{value:g}')); ax[0].set_xlabel('付费游戏价格 英镑 对数坐标'); ax[0].set_ylabel('游戏数量')
    gg=genres.head(8).sort_values(); ax[1].barh(gg.index,gg.values,color='#777777'); ax[1].set_xlabel('包含该类型的游戏数量'); savefig('01_market_overview.png')
    fig,ax=plt.subplots(1,2,figsize=(7.2,2.8))
    sr=d.sample(5000,random_state=42)
    ax[0].scatter(sr.reviews+1,sr.approval,s=5,alpha=.22,color='#555555'); ax[0].set_xscale('log'); ax[0].set_xlabel('评价总数加一 对数坐标'); ax[0].set_ylabel('原始好评比例')
    ax[1].scatter(sr.reviews+1,sr.wilson_lower,s=5,alpha=.22,color='#555555'); ax[1].set_xscale('log'); ax[1].set_xlabel('评价总数加一 对数坐标'); ax[1].set_ylabel('Wilson 好评比例下界'); savefig('02_review_uncertainty.png')
    fig,ax=plt.subplots(figsize=(7.2,3.0))
    for method in METHODS:
        r=comparison[comparison.method.eq(method)]; ax.plot(r.k,r.silhouette,marker='o',label=method)
    ax.set_xlabel('聚类数量 K'); ax.set_ylabel('验证集轮廓系数'); ax.legend(fontsize=11); savefig('03_cluster_selection.png')
    fig,ax=plt.subplots(1,2,figsize=(7.2,3.1))
    sample=d.sample(5000,random_state=42)
    sc=ax[0].scatter(sample.log_reviews,sample.wilson_lower,c=sample.cluster,cmap='tab10',s=5,alpha=.35)
    ax[0].set_xlabel('ln 评价总数加一'); ax[0].set_ylabel('Wilson 好评比例下界')
    means=pd.DataFrame(x,columns=FEATURE_NAMES).groupby(labels).mean()
    image=ax[1].imshow(means.to_numpy(),cmap='RdBu_r',aspect='auto',vmin=-2,vmax=2)
    ax[1].set_xticks(range(4),FEATURE_NAMES,rotation=25,ha='right'); ax[1].set_yticks(range(len(means)),[f'组{i}' for i in means.index])
    for row in range(len(means)):
        for col in range(4):ax[1].text(col,row,f'{means.iloc[row,col]:.2f}',ha='center',va='center',fontsize=11,color='black')
    fig.colorbar(image,ax=ax[1],shrink=.7,label='标准化特征均值'); savefig('04_cluster_profiles.png')
    fig,ax=plt.subplots(figsize=(7.2,3.0))
    abr=pd.DataFrame(ablations); names=['完整特征','去掉价格','去掉评价量','去掉好评下界','去掉发行时长']
    ax.bar(range(5),abr.silhouette,color='#777777'); ax.set_xticks(range(5),names,rotation=10); ax.set_ylabel('测试集轮廓系数 统一四维空间'); savefig('05_cluster_ablation.png')
    rt=pd.DataFrame(retrieval); fig,ax=plt.subplots(1,2,figsize=(7.2,2.8))
    names=['热门基线','仅类型','核心内容','完整标签']
    for axis,col,ylabel in [(ax[0],'precision_at_10','隐藏标签 Precision@10'),(ax[1],'catalog_coverage','训练目录覆盖率')]:
        vals=rt[rt.split.eq('test')][col].to_numpy(); axis.bar(range(4),vals,color=['#bdbdbd','#969696','#636363','#252525']); axis.set_xticks(range(4),names,rotation=12); axis.set_ylabel(ylabel)
        for i,v in enumerate(vals):axis.text(i,v+.01,f'{v:.3f}',ha='center',fontsize=10)
        axis.set_ylim(0,min(1,max(vals)*1.25+.03))
    savefig('06_retrieval_validation.png')

def main():
    started=time.perf_counter()
    d=pd.read_csv(RAW/'steam.csv')
    tags=pd.read_csv(RAW/'steamspy_tag_data.csv').set_index('appid').reindex(d.appid)
    assert d.appid.is_unique and tags.index.is_unique
    assert tags.notna().all().all(), 'Every core game must have full tag data'
    audit={'raw_shape':list(d.shape), 'missing':d.isna().sum().to_dict(),
           'duplicate_appid':int(d.appid.duplicated().sum()), 'duplicate_rows':int(d.duplicated().sum()),
           'full_tags_columns':len(tags.columns), 'tag_join_missing':int(tags.isna().any(axis=1).sum()),
           'source_sha256':hashlib.sha256((RAW/'steam.csv').read_bytes()).hexdigest(),
           'tags_sha256':hashlib.sha256((RAW/'steamspy_tag_data.csv').read_bytes()).hexdigest()}
    d['developer']=d.developer.fillna('未知开发商')
    d['publisher']=d.publisher.fillna('未知发行商')
    d['release_date']=pd.to_datetime(d.release_date, errors='coerce')
    critical=['price','positive_ratings','negative_ratings','median_playtime','average_playtime']
    invalid=d[critical].isna().any(axis=1)|(d[critical]<0).any(axis=1)|d.release_date.isna()|(d.release_date>ANCHOR)
    assert not invalid.any(), 'Do not silently remove invalid rows'
    d['reviews']=d.positive_ratings+d.negative_ratings
    n=d.reviews.to_numpy(float); p=np.divide(d.positive_ratings,n,out=np.zeros(len(d)),where=n>0)
    z=1.96
    d['approval']=np.where(n>0,p,np.nan)
    d['wilson_lower']=np.where(n>0,(p+z*z/(2*np.maximum(n,1))-z*np.sqrt((p*(1-p)+z*z/(4*np.maximum(n,1)))/np.maximum(n,1)))/(1+z*z/np.maximum(n,1)),0)
    d['release_age']=(ANCHOR-d.release_date).dt.days
    for name,source in [('log_price','price'),('log_reviews','reviews'),('log_release_age','release_age')]:
        d[name]=np.log1p(d[source])
    d['free']=d.price.eq(0)
    bounds=d.owners.str.extract(r'^(\d+)-(\d+)$').astype('int64')
    assert bounds.notna().all().all() and (bounds[1]>=bounds[0]).all()
    d['owners_low']=bounds[0]; d['owners_high']=bounds[1]
    # Split at the normalized developer-field level; same complete field never crosses splits.
    groups=d.developer.str.lower().str.strip()
    groups=groups.where(groups.ne('未知开发商'), 'unknown_'+d.appid.astype(str))
    all_idx=np.arange(len(d))
    tr,other=next(GroupShuffleSplit(n_splits=1,train_size=.6,random_state=42).split(d,groups=groups))
    vi,te=next(GroupShuffleSplit(n_splits=1,train_size=.5,random_state=43).split(other,groups=groups.iloc[other]))
    va=other[vi]; ts=other[te]
    split=np.full(len(d),'train',dtype=object); split[va]='validation'; split[ts]='test'
    d['split']=split
    for a,b in [(tr,va),(tr,ts),(va,ts)]: assert set(groups.iloc[a]).isdisjoint(groups.iloc[b])
    d[['appid','developer','split']].to_csv(OUT/'splits.csv',index=False,encoding='utf-8-sig')
    protocol={
      'seed':42,'split':'developer-field groups, 60/20/20 approximate game proportions; seeds42/43',
      'market_features':MARKET_COLS,'anchor_date':'2019-05-01 (maximum release date in frozen sample)',
      'cluster_candidates':{'algorithms':METHODS,'k':KS},
      'selection':'highest validation silhouette with every validation cluster >=2%; fixed sample2000',
      'test':'fixed independent test sample2000; no reselection',
      'recommendation':'training catalog only; 500 validation and 500 test queries; randomly hide one tag',
      'hidden_tag_rules':'train frequency50 to50% of train catalog; >=5 known tags; omit tags already in genres/categories',
      'recommender_candidates':['Popularity','Genres','CoreContent','FullTags'],
      'retrieval_metrics':'hidden-tag Precision@10, Hit@10, catalog coverage, list diversity; no real-user satisfaction claim',
      'split_counts':pd.Series(split).value_counts().to_dict()}
    dump(OUT/'protocol.json',protocol)  # frozen before any model fitting
    scaler=StandardScaler().fit(d.loc[tr,MARKET_COLS])
    x=scaler.transform(d[MARKET_COLS])
    rng=np.random.default_rng(2026)
    v_sample=np.sort(rng.choice(va,min(2000,len(va)),replace=False))
    t_sample=np.sort(rng.choice(ts,min(2000,len(ts)),replace=False))
    rows=[]; models={}
    for method in METHODS:
        for k in KS:
            start=time.perf_counter(); mod=model_for(method,k).fit(x[tr]); sec=time.perf_counter()-start
            score=cluster_scores(x[v_sample],mod.predict(x[v_sample]))
            share=np.bincount(mod.predict(x[va]),minlength=k)/len(va)
            row={'method':method,'k':k,'fit_seconds':sec,**score,'validation_min_share':float(share.min())}
            rows.append(row); models[(method,k)]=mod
            print('cluster',method,k,round(score['silhouette'],4),flush=True)
    comparison=pd.DataFrame(rows)
    comparison.to_csv(OUT/'cluster_validation.csv',index=False,encoding='utf-8-sig')
    eligible=comparison[comparison.validation_min_share.ge(.02)]
    assert len(eligible)>0
    winner=eligible.sort_values(['silhouette','davies_bouldin'],ascending=[False,True]).iloc[0]
    selected=models[(winner['method'],int(winner.k))]
    dump(OUT/'selected_cluster.json',winner.to_dict())  # selection saved before test evaluation
    test_scores=cluster_scores(x[t_sample],selected.predict(x[t_sample]))
    labels=selected.predict(x); d['cluster']=labels
    # Other methods at selected K are fixed comparators, not another test-driven selection.
    test_comparison=[]
    for method in METHODS:
        mod=models[(method,int(winner.k))]
        test_comparison.append({'method':method,'k':int(winner.k),**cluster_scores(x[t_sample],mod.predict(x[t_sample]))})
    pd.DataFrame(test_comparison).to_csv(OUT/'cluster_test_comparison.csv',index=False,encoding='utf-8-sig')
    seed_labels=[]; stability=[]
    for seed in [42,43,44,45,46]:
        mod=model_for(winner['method'],int(winner.k),seed).fit(x[tr])
        lab=mod.predict(x[ts]); seed_labels.append(lab)
        stability.append({'seed':seed,**cluster_scores(x[t_sample],mod.predict(x[t_sample]))})
    aris=[adjusted_rand_score(seed_labels[i],seed_labels[j]) for i,j in itertools.combinations(range(5),2)]
    pd.DataFrame(stability).to_csv(OUT/'cluster_stability.csv',index=False,encoding='utf-8-sig')
    # Ablate market feature blocks at frozen K/algorithm, evaluate in common full-feature geometry.
    ablations=[]
    for removed in [None,'log_price','log_reviews','wilson_lower','log_release_age']:
        keep=[i for i,c in enumerate(MARKET_COLS) if c!=removed]
        mod=model_for(winner['method'],int(winner.k)).fit(x[tr][:,keep])
        lab=mod.predict(x[t_sample][:,keep])
        ablations.append({'removed':removed or 'None',**cluster_scores(x[t_sample],lab)})
    pd.DataFrame(ablations).to_csv(OUT/'cluster_ablation.csv',index=False,encoding='utf-8-sig')
    # Stability of test silhouette under resampling, with the selected model fixed.
    base_labels=labels[t_sample]; dist=pairwise_distances(x[t_sample])
    boot=[]; brng=np.random.default_rng(2027)
    for _ in range(100):
        idx=brng.choice(len(t_sample),800,replace=True)
        boot.append(silhouette_score(dist[np.ix_(idx,idx)],base_labels[idx],metric='precomputed'))
    profile=[]
    genre_lists=d.genres.map(parts).tolist()
    for c in range(int(winner.k)):
        g=d[d.cluster.eq(c)]; genre_counts=pd.Series([v for i in g.index for v in genre_lists[i]]).value_counts()
        profile.append({'cluster':c,'n':len(g),'share':len(g)/len(d),'price_median':g.price.median(),
          'reviews_median':g.reviews.median(),'wilson_median':g.wilson_lower.median(),
          'approval_median':g.approval.median(),'age_median':g.release_age.median(),
          'free_share':g.free.mean(),'playtime_zero_share':g.median_playtime.eq(0).mean(),
          'top_genres':genre_counts.head(3).index.tolist(),
          'examples':g.sort_values('reviews',ascending=False).name.head(3).tolist()})
    dump(OUT/'cluster_profiles.json',profile)
    d.to_csv(OUT/'clean_games.csv',index=False,encoding='utf-8-sig')
    # Content-based recommendation. Full tags are binarized, so vote magnitude does not encode popularity.
    tag_cols=tags.columns.to_numpy()
    tag_presence=tags.to_numpy()>0
    train_freq=tag_presence[tr].sum(axis=0)
    eligible_tag=(train_freq>=50)&(train_freq<=len(tr)/2)
    cat_lists=d.categories.map(parts).tolist()
    core_tag_lists=d.steamspy_tags.map(parts).tolist()
    full_lists=[tag_cols[row].tolist() for row in tag_presence]
    def documents(mode,indices,hidden=None):
        result=[]
        for n,i in enumerate(indices):
            hidden_name=None if hidden is None else tag_cols[hidden[n]]
            tokens=['g:'+v for v in genre_lists[i] if v!=hidden_name]
            if mode!='Genres': tokens += ['c:'+v for v in cat_lists[i] if v!=hidden_name]
            if mode=='CoreContent': tokens += ['t:'+v for v in core_tag_lists[i] if v!=hidden_name]
            if mode=='FullTags': tokens += ['t:'+v for v in full_lists[i] if v!=hidden_name]
            result.append(' '.join(tokens))
        return result
    content={}; all_matrices={}
    for mode in ['Genres','CoreContent','FullTags']:
        cv=CountVectorizer(tokenizer=str.split,token_pattern=None,lowercase=False,binary=True,min_df=2)
        all_docs=documents(mode,all_idx)
        tf=TfidfTransformer()
        train_counts=cv.fit_transform([all_docs[i] for i in tr]); tf.fit(train_counts)
        matrix=tf.transform(cv.transform(all_docs)).astype(np.float32).tocsr()
        content[mode]=(cv,tf); all_matrices[mode]=matrix
    popularity=np.lexsort((d.appid.to_numpy()[tr],-d.reviews.to_numpy()[tr]))[:10]
    def make_queries(indices,seed):
        rg=np.random.default_rng(seed); valid=[]
        for i in indices:
            forbidden=set(genre_lists[i])|set(cat_lists[i])
            candidates=np.flatnonzero(tag_presence[i]&eligible_tag)
            candidates=[j for j in candidates if tag_cols[j] not in forbidden]
            if len(full_lists[i])>=5 and candidates:
                valid.append((int(i),int(rg.choice(candidates))))
        chosen=rg.choice(len(valid),min(500,len(valid)),replace=False)
        return [valid[j] for j in chosen]
    queries={'validation':make_queries(va,70),'test':make_queries(ts,71)}
    query_rows=[]
    for subset,q in queries.items():
        for i,h in q: query_rows.append({'split':subset,'row_index':i,'appid':int(d.appid.iloc[i]),'hidden_tag':tag_cols[h]})
    pd.DataFrame(query_rows).to_csv(OUT/'retrieval_queries.csv',index=False,encoding='utf-8-sig')
    retrieval=[]; rank_rows=[]; per_query={}; rec_indexes={}
    # Both folds are evaluated using identical frozen candidate settings.
    for subset,q in queries.items():
        qi=np.array([i for i,h in q]); hidden=np.array([h for i,h in q])
        for mode in ['Popularity','Genres','CoreContent','FullTags']:
            if mode=='Popularity': ranked=np.tile(popularity,(len(q),1))
            else:
                cv,tf=content[mode]
                qx=tf.transform(cv.transform(documents(mode,qi,hidden))).astype(np.float32)
                scores=cosine_similarity(qx,all_matrices[mode][tr])
                # Stable ties are resolved by ascending appid (tr is ascending CSV index).
                ranked=np.argsort(-scores,axis=1,kind='stable')[:,:10]
            rec_indexes[(subset,mode)]=ranked
            relevance=tag_presence[tr[ranked],hidden[:,None]]
            ps=relevance.mean(axis=1); hits=relevance.any(axis=1)
            per_query[(subset,mode)]=ps
            diversity=[]
            for j,ids in enumerate(ranked):
                # Hidden tag itself is excluded when measuring content diversity.
                cvf,tff=content['FullTags']
                cx=tff.transform(cvf.transform(documents('FullTags',tr[ids],np.repeat(hidden[j],10))))
                sim=cosine_similarity(cx)
                diversity.append(float((1-sim[np.triu_indices(10,1)]).mean()))
                if subset=='test':
                    for rank,cidx in enumerate(ids):
                        rank_rows.append({'method':mode,'query_appid':int(d.appid.iloc[qi[j]]),
                          'hidden_tag':tag_cols[hidden[j]],'rank':rank+1,'candidate_appid':int(d.appid.iloc[tr[cidx]]),
                          'hidden_tag_match':int(relevance[j,rank])})
            row={'split':subset,'method':mode,'queries':len(q),'precision_at_10':float(ps.mean()),
              'hit_at_10':float(hits.mean()),'catalog_coverage':float(len(np.unique(ranked))/len(tr)),
              'list_diversity':float(np.mean(diversity))}
            retrieval.append(row); print('retrieve',row,flush=True)
    pd.DataFrame(retrieval).to_csv(OUT/'retrieval_metrics.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(rank_rows).to_csv(OUT/'retrieval_test_ranks.csv',index=False,encoding='utf-8-sig')
    # This comparison is prespecified; the test result is not used to redesign any model.
    delta=per_query[('test','FullTags')]-per_query[('test','Genres')]
    brng=np.random.default_rng(72)
    delta_boot=np.mean(delta[brng.integers(0,len(delta),(2000,len(delta)))],axis=1)
    rec_winner=max([r for r in retrieval if r['split']=='validation'],key=lambda r:r['precision_at_10'])['method']
    examples=[]
    for appid in [413150,620]:
        match=d.index[d.appid.eq(appid)]
        if not len(match): continue
        i=int(match[0]); sim=cosine_similarity(all_matrices['FullTags'][i],all_matrices['FullTags']).ravel(); sim[i]=-1
        ids=np.argsort(-sim,kind='stable')[:5]
        examples.append({'query':d.name.iloc[i],'appid':appid,'recommendations':[
          {'appid':int(d.appid.iloc[j]),'name':d.name.iloc[j],'similarity':float(sim[j]),
           'common_tags':sorted(set(full_lists[i])&set(full_lists[j]))[:10],
           'price':float(d.price.iloc[j]),'reviews':int(d.reviews.iloc[j])} for j in ids]})
    dump(OUT/'recommendation_examples.json',examples)
    # Catalog summary is descriptive, using every audited game.
    genres=pd.Series([g for values in genre_lists for g in values]).value_counts()
    genre_rows=[]
    for genre,count in genres.head(10).items():
        ix=np.array([genre in v for v in genre_lists]); g=d[ix]
        genre_rows.append({'genre':genre,'count':int(count),'price_median':float(g.price.median()),
             'reviews_median':float(g.reviews.median()),'approval_median_n100':float(g.loc[g.reviews.ge(100),'approval'].median())})
    pd.DataFrame(genre_rows).to_csv(OUT/'genre_summary.csv',index=False,encoding='utf-8-sig')
    # Correlation is descriptive; not a causal price effect or a forecast of sales.
    enough=d.reviews.ge(100)
    rho=spearmanr(d.loc[enough,'price'],d.loc[enough,'approval'])
    audit.update({'retained_rows':len(d),'invalid_rows':int(invalid.sum()),
       'release_min':str(d.release_date.min().date()),'release_max':str(d.release_date.max().date()),
       'free_n':int(d.free.sum()),'no_reviews_n':int(d.reviews.eq(0).sum()),
       'median_playtime_zero_n':int(d.median_playtime.eq(0).sum()),
       'price_median':float(d.price.median()),'reviews_median':float(d.reviews.median()),
       'reviews_ge100_n':int(enough.sum()),'price_approval_spearman_ge100':float(rho.statistic),
       'developers_n':int(groups.nunique()),'numeric_feature_zero_variance':bool(np.any(scaler.var_==0))})
    metrics={'audit':audit,'protocol':protocol,'selected_cluster':winner.to_dict(),'cluster_test':test_scores,
      'cluster_silhouette_resample_interval':np.quantile(boot,[.025,.975]).tolist(),
      'stability':{'seeds':[42,43,44,45,46],'pairwise_ari':aris,'mean_ari':float(np.mean(aris)),'min_ari':float(np.min(aris))},
      'cluster_profiles':profile,'cluster_test_comparison':test_comparison,'cluster_ablation':ablations,
      'retrieval':retrieval,'retrieval_selected_on_validation':rec_winner,
      'full_minus_genres_p10':float(delta.mean()),'paired_query_bootstrap95':np.quantile(delta_boot,[.025,.975]).tolist(),
      'recommendation_examples':examples,'genre_summary':genre_rows,
      'versions':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,
                  'scipy':scipy.__version__,'sklearn':sklearn.__version__,'matplotlib':matplotlib.__version__},
      'seconds':time.perf_counter()-started}
    dump(OUT/'metrics.json',metrics)
    joblib.dump({'scaler':scaler,'model':selected,'features':MARKET_COLS},OUT/'market_cluster.joblib')
    joblib.dump(content,OUT/'content_vectorizers.joblib')
    for mode,matrix in all_matrices.items(): sparse.save_npz(OUT/f'{mode}_features.npz',matrix)
    make_figures(d,x,comparison,ablations,retrieval,genres)
    print('DONE',json.dumps({'winner':winner.to_dict(),'test':test_scores,'stability':metrics['stability'],'seconds':metrics['seconds']},ensure_ascii=False),flush=True)

if __name__=='__main__': main()
