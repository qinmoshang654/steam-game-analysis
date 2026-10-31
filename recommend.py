"""Query the frozen local content recommender. No internet access is needed."""
from pathlib import Path
import argparse, json
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

ROOT=Path(__file__).resolve().parent

def recommend(appid, topn=5, method=None):
    d=pd.read_csv(ROOT/'results/clean_games.csv')
    metrics=json.loads((ROOT/'results/metrics.json').read_text(encoding='utf8'))
    method=method or metrics['retrieval_selected_on_validation']
    if method not in ['Genres','CoreContent','FullTags','GenrePopularity','TagsOnly','BlockTags75']:
        raise ValueError('Unsupported content method')
    found=d.index[d.appid.eq(appid)]
    if len(found)!=1: raise ValueError('appid not found in the frozen 2019 catalog')
    i=int(found[0])
    if method=='GenrePopularity':
        query_genres=set(str(d.genres.iloc[i]).lower().split(';'))
        shared=np.array([len(query_genres&set(str(s).lower().split(';'))) for s in d.genres])
        order=np.lexsort((d.appid.to_numpy(),-d.reviews.to_numpy(),-shared))
        order=order[order!=i]
        scores=np.zeros(len(d));scores[order]=np.arange(len(order),0,-1)
        score_type='lexicographic genre overlap then review count; not cosine'
    elif method in ['TagsOnly','BlockTags75']:
        blocks={key:sparse.load_npz(ROOT/f'results/deep/{key}_features.npz')
                for key in (['tags'] if method=='TagsOnly' else ['tags','genre','category'])}
        scores=cosine_similarity(blocks['tags'][i],blocks['tags']).ravel()
        if method=='BlockTags75':
            scores=.75*scores+.125*cosine_similarity(blocks['genre'][i],blocks['genre']).ravel()+.125*cosine_similarity(blocks['category'][i],blocks['category']).ravel()
        score_type='tag cosine' if method=='TagsOnly' else 'weighted block cosine'
    else:
        x=sparse.load_npz(ROOT/f'results/{method}_features.npz')
        scores=cosine_similarity(x[i],x).ravel();score_type='cosine'
    scores[i]=-1
    ids=np.argsort(-scores,kind='stable')[:topn]
    return {'query':d.name.iloc[i],'query_appid':appid,'method':method,
      'catalog':'all frozen sample games; demonstration only; evaluation uses training catalog',
      'score_type':score_type,
      'recommendations':[{'rank':rank+1,'appid':int(d.appid.iloc[j]),'name':d.name.iloc[j],
       'score':float(scores[j]),'price_gbp':float(d.price.iloc[j]),'reviews':int(d.reviews.iloc[j]),
       'common_core_tags':sorted(set(str(d.steamspy_tags.iloc[i]).split(';')) & set(str(d.steamspy_tags.iloc[j]).split(';')))}
       for rank,j in enumerate(ids)]}

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--appid',type=int,required=True); p.add_argument('--topn',type=int,default=5)
    p.add_argument('--method',choices=['Genres','CoreContent','FullTags','GenrePopularity','TagsOnly','BlockTags75']); a=p.parse_args()
    print(json.dumps(recommend(a.appid,a.topn,a.method),ensure_ascii=False,indent=2))
