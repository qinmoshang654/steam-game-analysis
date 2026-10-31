from pathlib import Path
import json
import pandas as pd
import joblib
from run_analysis import make_figures
root=Path(__file__).resolve().parent
m=json.loads((root/'results/metrics.json').read_text(encoding='utf8'))
d=pd.read_csv(root/'results/clean_games.csv')
b=joblib.load(root/'results/market_cluster.joblib')
x=b['scaler'].transform(d[b['features']])
genres=pd.Series([v for cell in d.genres for v in cell.lower().replace(' ','_').split(';')]).value_counts()
make_figures(d,x,pd.read_csv(root/'results/cluster_validation.csv'),m['cluster_ablation'],m['retrieval'],genres)
print('Six figures regenerated from saved results')
