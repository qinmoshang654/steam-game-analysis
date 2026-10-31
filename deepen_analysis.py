"""Exploratory extension of the original, already inspected Steam experiments.

No original results are overwritten. See results/deep/protocol.json for the
frozen extension protocol and the limitation from reusing an observed test set.
"""
from pathlib import Path
import os
ROOT=Path(__file__).resolve().parent
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'results/.mpl-cache'))
import json, itertools, time, hashlib
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler, RobustScaler, normalize
from sklearn.cluster import MiniBatchKMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import balanced_accuracy_score, adjusted_rand_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=ROOT/'results/deep'; OUT.mkdir(exist_ok=True)
FIG=ROOT/'figures'
plt.rcParams.update({'font.family':['Times New Roman','SimSun'],'font.size':12,
    'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False})

def dump(name,obj):
    (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,
       default=lambda v:v.item() if hasattr(v,'item') else str(v)),encoding='utf8')
def parts(v):return sorted(set(x.strip().lower().replace(' ','_') for x in str(v).split(';') if x.strip()))
def fitcluster(x,seed=42):return MiniBatchKMeans(2,n_init=20,batch_size=1024,max_iter=500,random_state=seed).fit(x)
def fig(name):
    plt.tight_layout();plt.savefig(FIG/name,dpi=180,bbox_inches='tight');plt.close()
def interval(v):return np.quantile(v,[.025,.5,.975]).tolist()

def main():
    start=time.perf_counter()
    protocol={
      'status':'exploratory extension after original test results were inspected; not a new untouched holdout',
      'data':'same frozen Kaggle sample and original developer-field train/validation/test split',
      'cluster_diagnostics':{'surrogates':'review-only depth1 and depth2; four-feature depth2',
        'bootstrap':'30 developer-group resamples; refit training scaler and fixed K2 MiniBatchKMeans; test ARI',
        'sensitivity':['90/95/99 percent Wilson','Laplace approval','raw approval','RobustScaler','reviews>=20']},
      'retrieval_candidates':['Popularity','GenrePopularity','Genres','CoreContent','FullTags','TagsOnly',
                              'BlockTags25','BlockTags50','BlockTags75','BlockTags100'],
      'block_formula':'cosine weighted sum: (1-w)*(.5 genre cosine + .5 category cosine) + w*full-tag cosine',
      'selection':'extension model selected using validation P@10 only; ties prefer simpler first-in-list',
      'frequency_strata':'hidden-tag training prevalence <=1%, (1%,5%], >5%',
      'review_strata':'query reviews <20,20..99,>=100; no real cold-start claim',
      'uncertainty':'2000 paired resamples of normalized developer-field groups, seeds 820..822',
      'sparsity':'0/25/50/75 percent independently deleted visible player tags, fixed masks shared by methods; five seeds 910..914',
      'input_sha256':{f:hashlib.sha256((ROOT/'data/raw'/f).read_bytes()).hexdigest()
                      for f in ['steam.csv','steamspy_tag_data.csv']}}
    dump('protocol.json',protocol)
    d=pd.read_csv(ROOT/'results/clean_games.csv')
    tr=np.flatnonzero(d.split.eq('train'));va=np.flatnonzero(d.split.eq('validation'));te=np.flatnonzero(d.split.eq('test'))
    cols=['log_price','log_reviews','wilson_lower','log_release_age']
    saved=joblib.load(ROOT/'results/market_cluster.joblib')
    x=saved['scaler'].transform(d[cols]);labels=saved['model'].predict(x)
    assert np.array_equal(labels,d.cluster)
    groups=d.developer.str.lower().str.strip().to_numpy()
    # Explain a fitted partition, not externally labelled game quality.
    surrogates=[]
    for name,features,depth in [('ReviewStump',['log_reviews'],1),('ReviewDepth2',['log_reviews'],2),('FourDepth2',cols,2)]:
        model=DecisionTreeClassifier(max_depth=depth,min_samples_leaf=100,random_state=42)
        model.fit(d.loc[tr,features],labels[tr]);pred=model.predict(d.loc[te,features])
        thresholds=[{'feature':features[int(f)],'threshold':float(t)} for f,t in zip(model.tree_.feature,model.tree_.threshold) if f>=0]
        surrogates.append({'model':name,'features':features,'depth':depth,'test_agreement':float((pred==labels[te]).mean()),
             'test_balanced_agreement':float(balanced_accuracy_score(labels[te],pred)),
             'test_ari':float(adjusted_rand_score(labels[te],pred)),'thresholds':thresholds})
    pd.DataFrame(surrogates).to_json(OUT/'cluster_surrogates.json',orient='records',force_ascii=False,indent=2)
    rho=spearmanr(d.loc[tr,'log_reviews'],d.loc[tr,'wilson_lower']).statistic
    # Conditional descriptive comparisons on the test partition.
    bins=[0,20,100,1000,np.inf];bn=['1–19','20–99','100–999','1000+'];conditional=[]
    for b,(low,high) in enumerate(zip(bins[:-1],bins[1:])):
        for c in [0,1]:
            ids=te[(d.reviews.to_numpy()[te]>=low)&(d.reviews.to_numpy()[te]<high)&(labels[te]==c)]
            if len(ids):conditional.append({'review_bin':bn[b],'cluster':c,'n':len(ids),
                'approval_median':float(d.approval.iloc[ids].median()),
                'wilson_median':float(d.wilson_lower.iloc[ids].median()),
                'age_median':float(d.release_age.iloc[ids].median())})
    pd.DataFrame(conditional).to_csv(OUT/'cluster_conditional.csv',index=False,encoding='utf-8-sig')
    # Developer bootstrap: refit both scaling and clustering, keep evaluation fixed.
    unique,inv=np.unique(groups[tr],return_inverse=True)
    group_rows=[tr[inv==j] for j in range(len(unique))]
    bootstrap=[]
    for rep in range(30):
        rg=np.random.default_rng(600+rep);chosen=rg.integers(0,len(unique),len(unique))
        ids=np.concatenate([group_rows[j] for j in chosen]);sc=StandardScaler().fit(d.loc[ids,cols])
        mod=fitcluster(sc.transform(d.loc[ids,cols]),42)
        lab=mod.predict(sc.transform(d.loc[te,cols]))
        bootstrap.append({'rep':rep,'sample_games':len(ids),'ari':float(adjusted_rand_score(labels[te],lab))})
    pd.DataFrame(bootstrap).to_csv(OUT/'cluster_group_bootstrap.csv',index=False,encoding='utf-8-sig')
    print('cluster bootstrap done',interval([r['ari'] for r in bootstrap]),flush=True)
    # Prespecified measurement alternatives, fixed K and algorithm; ARI comparison
    # avoids comparing silhouettes in different distance spaces.
    sensitivity=[];n=d.reviews.to_numpy(float);p=d.approval.to_numpy(float)
    for label in ['Wilson90','Wilson95','Wilson99','Laplace','RawApproval','RobustScaler','ReviewsGE20']:
        xx=d[cols].to_numpy().copy();trainids=tr;evalids=te
        if label.startswith('Wilson'):
            z={'Wilson90':1.6448536269514722,'Wilson95':1.96,'Wilson99':2.5758293035489004}[label]
            xx[:,2]=(p+z*z/(2*n)-z*np.sqrt((p*(1-p)+z*z/(4*n))/n))/(1+z*z/n)
        elif label=='Laplace':xx[:,2]=(d.positive_ratings.to_numpy()+1)/(n+2)
        elif label=='RawApproval':xx[:,2]=p
        elif label=='ReviewsGE20':
            trainids=tr[n[tr]>=20];evalids=te[n[te]>=20]
        sc=(RobustScaler() if label=='RobustScaler' else StandardScaler()).fit(xx[trainids])
        mod=fitcluster(sc.transform(xx[trainids]));lab=mod.predict(sc.transform(xx[evalids]))
        sensitivity.append({'variant':label,'train_n':len(trainids),'test_n':len(evalids),
          'ari_to_original':float(adjusted_rand_score(labels[evalids],lab)),
          'min_share':float(pd.Series(lab).value_counts(normalize=True).min())})
    pd.DataFrame(sensitivity).to_csv(OUT/'cluster_sensitivity.csv',index=False,encoding='utf-8-sig')
    # TF-IDF blocks fitted independently using only training games.
    tag=pd.read_csv(ROOT/'data/raw/steamspy_tag_data.csv').set_index('appid').reindex(d.appid)
    tag_names=tag.columns.to_numpy();presence=tag.to_numpy()>0;freq=presence[tr].mean(axis=0)
    genre=d.genres.map(parts).tolist();cat=d.categories.map(parts).tolist();core=d.steamspy_tags.map(parts).tolist()
    full=[tag_names[row].tolist() for row in presence]
    docs={'genre':genre,'category':cat,'core':core,'tags':full}
    block={};vectorizers={}
    for key,values in docs.items():
        cv=CountVectorizer(tokenizer=str.split,token_pattern=None,lowercase=False,binary=True,min_df=2)
        cv.fit([' '.join(values[i]) for i in tr]);tf=TfidfTransformer().fit(cv.transform([' '.join(values[i]) for i in tr]))
        matrix=tf.transform(cv.transform([' '.join(a) for a in values])).astype('float32').tocsr()
        vectorizers[key]=(cv,tf);block[key]=matrix
    original=joblib.load(ROOT/'results/content_vectorizers.joblib')
    originalmat={m:sparse.load_npz(ROOT/f'results/{m}_features.npz') for m in ['Genres','CoreContent','FullTags']}
    genre_count=CountVectorizer(tokenizer=str.split,token_pattern=None,lowercase=False,binary=True)
    genre_count.fit([' '.join(genre[i]) for i in tr])
    genre_binary=genre_count.transform([' '.join(a) for a in genre]).astype('float32')
    methods=protocol['retrieval_candidates']
    queries=pd.read_csv(ROOT/'results/retrieval_queries.csv')
    queries['tag_index']=[int(np.flatnonzero(tag_names==s)[0]) for s in queries.hidden_tag]
    cached={};query_masks={};pop_order=np.lexsort((d.appid.to_numpy()[tr],-d.reviews.to_numpy()[tr]))
    popscore=np.zeros(len(tr),dtype='float32');popscore[pop_order]=np.arange(len(tr),0,-1)/len(tr)
    def qscores(q,drop=0.,seed=0,wanted=None,mask_rng=None):
        wanted=methods if wanted is None else wanted
        qi=q.row_index.to_numpy();hidden=q.hidden_tag.tolist();rg=np.random.default_rng(seed) if mask_rng is None else mask_rng
        kept=[]
        for i,hid in zip(qi,hidden):
            # The separately collected three-core-tag field is not always a
            # strict subset of the full-tag export. Preserve both raw sources.
            visible=sorted((set(full[i])|set(core[i]))-{hid})
            delete=set(rg.choice(visible,int(np.floor(len(visible)*drop)),replace=False)) if drop else set()
            kept.append(set(visible)-delete)
        scores={}
        for mode in ['Genres','CoreContent','FullTags']:
            if mode not in wanted:continue
            cv,tf=original[mode];texts=[]
            for i,hid,keep in zip(qi,hidden,kept):
                tokens=['g:'+v for v in genre[i] if v!=hid]
                if mode!='Genres':tokens+=['c:'+v for v in cat[i] if v!=hid]
                if mode=='CoreContent':tokens+=['t:'+v for v in core[i] if v in keep]
                if mode=='FullTags':tokens+=['t:'+v for v in full[i] if v in keep]
                texts.append(' '.join(tokens))
            qx=tf.transform(cv.transform(texts)).astype('float32')
            scores[mode]=(qx@originalmat[mode][tr].T).toarray()
        tagcv,tagtf=vectorizers['tags']
        qtag=tagtf.transform(tagcv.transform([' '.join(sorted(a)) for a in kept])).astype('float32')
        ts=(qtag@block['tags'][tr].T).toarray()
        if 'TagsOnly' in wanted:scores['TagsOnly']=ts
        cache_key=tuple(qi)
        if cache_key not in cached:
            gs=(block['genre'][qi]@block['genre'][tr].T).toarray()
            cs=(block['category'][qi]@block['category'][tr].T).toarray()
            share=(genre_binary[qi]@genre_binary[tr].T).toarray()
            if len(cached)>=2:cached.pop(next(iter(cached)))
            cached[cache_key]=(gs,cs,share)
        gs,cs,share=cached[cache_key]
        # Per query lexicographic ordering: shared genre count, then popularity.
        # Identical catalog and hidden target removal as the content methods.
        if 'GenrePopularity' in wanted:scores['GenrePopularity']=share+popscore[None,:]*.5
        if 'Popularity' in wanted:scores['Popularity']=np.tile(popscore,(len(q),1))
        for w in [.25,.5,.75,1.]:
            key=f'BlockTags{int(100*w)}'
            if key in wanted:scores[key]=ts if w==1 else (1-w)*(.5*gs+.5*cs)+w*ts
        return scores,kept
    metrics=[];perqueries=[];rankrecords=[];ranks_cache={}
    for fold in ['validation','test']:
        q=queries[queries.split.eq(fold)].reset_index(drop=True)
        ranked_total={mode:[] for mode in methods}
        for b in range(0,len(q),100):
            score,_=qscores(q.iloc[b:b+100])
            for mode in methods:ranked_total[mode].append(np.argsort(-score[mode],axis=1,kind='stable')[:,:10])
            del score
        hidden=q.tag_index.to_numpy();expected=freq[hidden]
        for mode in methods:
            ranked=np.concatenate(ranked_total[mode])
            relevance=presence[tr[ranked],hidden[:,None]]
            precis=relevance.mean(axis=1);hit=relevance.any(axis=1).astype(float)
            ranks_cache[(fold,mode)]=ranked
            pq=pd.DataFrame({'split':fold,'method':mode,'appid':q.appid,'row_index':q.row_index,
                'developer':groups[q.row_index],'hidden_tag':q.hidden_tag,'tag_prevalence':expected,
                'precision':precis,'hit':hit,'lift':precis/expected,
                'query_reviews':d.reviews.to_numpy()[q.row_index],
                'query_tags':np.array([len(full[i]) for i in q.row_index])})
            perqueries.append(pq)
            row={'split':fold,'method':mode,'p10':float(precis.mean()),'hit10':float(hit.mean()),
                'coverage':float(np.unique(ranked).size/len(tr)),
                'macro_tag_p10':float(pq.groupby('hidden_tag').precision.mean().mean()),
                'mean_lift':float(pq.lift.mean()),'chance_p10':float(expected.mean())}
            metrics.append(row)
            if fold=='test':
                for j,ids in enumerate(ranked):
                    for rank,k in enumerate(ids):rankrecords.append({'method':mode,'query_appid':int(q.appid.iloc[j]),
                        'hidden_tag':q.hidden_tag.iloc[j],'rank':rank+1,'candidate_appid':int(d.appid.iloc[tr[k]]),
                        'match':int(relevance[j,rank])})
            print('retrieval',row,flush=True)
        if fold=='validation':
            validation=[a for a in metrics if a['split']=='validation']
            chosen=max(validation,key=lambda a:a['p10'])['method']
            weights=[a for a in validation if a['method'].startswith('BlockTags')]
            chosen_weight=max(weights,key=lambda a:a['p10'])['method']
            dump('selected_before_extended_test.json',{'method':chosen,'weighted_method':chosen_weight,
                  'basis':'validation P10 only; extension itself designed after original test inspection'})
    pq=pd.concat(perqueries,ignore_index=True)
    pd.DataFrame(metrics).to_csv(OUT/'retrieval_metrics.csv',index=False,encoding='utf-8-sig')
    pq.to_csv(OUT/'per_query_metrics.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(rankrecords).to_csv(OUT/'test_ranks.csv',index=False,encoding='utf-8-sig')
    selected=['Popularity','GenrePopularity','CoreContent','FullTags','TagsOnly',chosen_weight]
    selected=list(dict.fromkeys(selected))
    test=pq[pq.split.eq('test')].copy()
    test['frequency_group']=pd.cut(test.tag_prevalence,[0,.01,.05,1],labels=['<=1%','1–5%','>5%'])
    test['review_group']=pd.cut(test.query_reviews,[0,19,99,np.inf],labels=['<20','20–99','>=100'])
    test['tag_count_group']=pd.cut(test.query_tags,[0,9,19,np.inf],labels=['5–9','10–19','>=20'])
    strat=[]
    for col in ['frequency_group','review_group','tag_count_group']:
        for (mode,g),sub in test[test.method.isin(selected)].groupby(['method',col],observed=True):
            strat.append({'dimension':col,'group':str(g),'method':mode,'queries':len(sub),
               'p10':float(sub.precision.mean()),'hit10':float(sub.hit.mean()),'lift':float(sub.lift.mean()),
               'chance_p10':float(sub.tag_prevalence.mean())})
    pd.DataFrame(strat).to_csv(OUT/'retrieval_strata.csv',index=False,encoding='utf-8-sig')
    # Cluster bootstrap preserves all games of a developer together.
    uncertainty=[]
    for j,(a,b) in enumerate([('FullTags','Genres'),('CoreContent','Popularity'),(chosen_weight,'GenrePopularity')]):
        left=test[test.method.eq(a)].set_index('appid');right=test[test.method.eq(b)].set_index('appid').loc[left.index]
        delta=left.precision.to_numpy()-right.precision.to_numpy();dev=left.developer.to_numpy()
        gg=pd.DataFrame({'developer':dev,'delta':delta}).groupby('developer').delta.agg(['sum','count'])
        rg=np.random.default_rng(820+j);ids=rg.integers(0,len(gg),(2000,len(gg)))
        boot=gg['sum'].to_numpy()[ids].sum(axis=1)/gg['count'].to_numpy()[ids].sum(axis=1)
        uncertainty.append({'a':a,'b':b,'delta':float(delta.mean()),'ci95':np.quantile(boot,[.025,.975]).tolist(),
            'developer_groups':len(gg),'wins':int((delta>1e-8).sum()),'ties':int((np.abs(delta)<=1e-8).sum()),
            'losses':int((delta< -1e-8).sum())})
    dump('paired_developer_bootstrap.json',uncertainty)
    # Candidate representations stay fixed. Only observable query tags are removed.
    q=queries[queries.split.eq('test')].reset_index(drop=True);hidden=q.tag_index.to_numpy()
    sparse_rows=[]
    for missing in [0.,.25,.5,.75]:
        for rep in range(5):
            if missing==0:
                for mode in ['CoreContent','FullTags','TagsOnly',chosen_weight]:
                    base=next(a for a in metrics if a['split']=='test' and a['method']==mode)
                    sparse_rows.append({'missing':missing,'rep':rep,'method':mode,'p10':base['p10'],'hit10':base['hit10']})
                continue
            wanted=['CoreContent','FullTags','TagsOnly',chosen_weight]
            # A single seeded mask sequence is applied to all methods. Chunk
            # seeds are deterministic and shared; no 500x16040x10 dense cache.
            ranked_chunks={mode:[] for mode in wanted}
            mask_rng=np.random.default_rng(910+rep)
            for b in range(0,len(q),100):
                scores,_=qscores(q.iloc[b:b+100],missing,910+rep,wanted=wanted,mask_rng=mask_rng)
                for mode in wanted:ranked_chunks[mode].append(np.argsort(-scores[mode],axis=1,kind='stable')[:,:10])
                del scores
            for mode in ['CoreContent','FullTags','TagsOnly',chosen_weight]:
                ix=np.concatenate(ranked_chunks[mode])
                rel=presence[tr[ix],hidden[:,None]]
                sparse_rows.append({'missing':missing,'rep':rep,'method':mode,
                   'p10':float(rel.mean()),'hit10':float(rel.any(axis=1).mean())})
        print('sparse',missing,'done',flush=True)
    pd.DataFrame(sparse_rows).to_csv(OUT/'tag_missingness.csv',index=False,encoding='utf-8-sig')
    # Save a reusable all-catalog extension model without changing original default.
    joblib.dump(vectorizers,OUT/'block_vectorizers.joblib')
    for key,value in block.items():sparse.save_npz(OUT/f'{key}_features.npz',value)
    # Failure cases selected by largest per-query differences, not manual appeal.
    fulltest=test[test.method.eq('FullTags')].set_index('appid')
    coretest=test[test.method.eq('CoreContent')].set_index('appid').loc[fulltest.index]
    delta=fulltest.precision-coretest.precision
    failure=[]
    for direction,apps in [('full_gain',delta.sort_values(ascending=False,kind='stable').index[:3]),
                            ('full_loss',delta.sort_values(kind='stable').index[:3])]:
        for app in apps:
            row=fulltest.loc[app];idx=int(row.row_index)
            failure.append({'direction':direction,'appid':int(app),'name':d.name.iloc[idx],
                'hidden_tag':row.hidden_tag,'tag_prevalence':float(row.tag_prevalence),
                'full_p10':float(row.precision),'core_p10':float(coretest.loc[app,'precision']),
                'visible_tags':sorted(set(full[idx])-{row.hidden_tag})})
    dump('retrieval_failure_cases.json',failure)
    summary={'protocol':protocol,'surrogates':surrogates,'training_rho_reviews_wilson':float(rho),
       'core_not_subset_full_games':int(sum(bool(set(a)-set(b)) for a,b in zip(core,full))),
       'cluster_conditional':conditional,'cluster_group_bootstrap_ari':interval([a['ari'] for a in bootstrap]),
       'cluster_sensitivity':sensitivity,'retrieval':metrics,'selected':chosen,'selected_weight':chosen_weight,
       'strata':strat,'uncertainty':uncertainty,'sparsity_summary':pd.DataFrame(sparse_rows).groupby(['method','missing']).agg(
          p10=('p10','mean'),p10_sd=('p10','std'),hit10=('hit10','mean')).reset_index().to_dict('records'),
       'failure_cases':failure,'seconds':time.perf_counter()-start}
    dump('summary.json',summary)
    # Evidence plots.
    fig1,ax=plt.subplots(1,2,figsize=(7.2,2.8))
    for c,color in [(0,'#777777'),(1,'#202020')]:
        ss=d[(d.split=='test')&(d.cluster==c)]
        ax[0].hist(ss.log_reviews,bins=np.linspace(0,15,36),histtype='step',density=True,color=color,label=f'组{c}')
    ax[0].set_xlabel('ln 评价数加一');ax[0].set_ylabel('密度');ax[0].legend()
    names=['评价量单门槛','评价量深度二','四特征深度二']
    vals=[a['test_balanced_agreement'] for a in surrogates]
    ax[1].bar(range(3),vals,color=['#aaa','#777','#333']);ax[1].set_xticks(range(3),names,rotation=12)
    ax[1].set_ylabel('测试平衡一致率');ax[1].set_ylim(0,1)
    for i,v in enumerate(vals):ax[1].text(i,v+.015,f'{v:.3f}',ha='center')
    fig('07_cluster_mechanism.png')
    fig1,ax=plt.subplots(figsize=(7.2,2.9))
    ss=pd.DataFrame(sensitivity)
    nn=['Wilson90','Wilson95','Wilson99','Laplace','RawApproval','RobustScaler','ReviewsGE20']
    lab=['Wilson 90%','Wilson 95%','Wilson 99%','拉普拉斯','原始好评率','稳健缩放','评价数≥20']
    ax.bar(range(len(nn)),ss.ari_to_original,color='#666');ax.set_xticks(range(len(nn)),lab,rotation=18)
    ax.set_ylim(0,1.12);ax.set_ylabel('与原分群的测试 ARI')
    for i,v in enumerate(ss.ari_to_original):ax.text(i,v+.025,f'{v:.3f}',ha='center',fontsize=10)
    fig('08_cluster_sensitivity.png')
    st=pd.DataFrame(strat);fig1,ax=plt.subplots(1,2,figsize=(7.2,2.9))
    for mode,display in [('Popularity','热门'),('GenrePopularity','类型内热门'),('CoreContent','核心内容'),('FullTags','完整标签'),(chosen_weight,'分块加权')]:
        for axis,col,cats in [(ax[0],'frequency_group',['<=1%','1–5%','>5%']),(ax[1],'review_group',['<20','20–99','>=100'])]:
            sub=st[(st.dimension==col)&(st.method==mode)].set_index('group').reindex(cats)
            axis.plot(range(3),sub.p10,marker='o',label=display)
            axis.set_xticks(range(3),cats);axis.set_ylabel('隐藏标签 P@10')
    ax[0].set_xlabel('目标标签训练占比');ax[1].set_xlabel('查询游戏评价数')
    ax[0].legend(fontsize=9,ncol=2);fig('09_retrieval_strata.png')
    fig1,ax=plt.subplots(figsize=(7.2,2.9));sp=pd.DataFrame(summary['sparsity_summary'])
    for mode,display in [('CoreContent','核心内容'),('FullTags','完整标签'),('TagsOnly','仅完整标签'),(chosen_weight,'分块加权')]:
        sub=sp[sp.method==mode]
        ax.errorbar(sub.missing*100,sub.p10,yerr=sub.p10_sd,marker='o',capsize=3,label=display)
    ax.set_xlabel('额外删除的可见玩家标签比例 %');ax.set_ylabel('测试隐藏标签 P@10');ax.legend(fontsize=10)
    fig('10_retrieval_missingness.png')
    print('DONE extension seconds',summary['seconds'],flush=True)

if __name__=='__main__':main()
