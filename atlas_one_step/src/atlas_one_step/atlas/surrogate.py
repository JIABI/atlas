
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

def fit_surrogate(df: pd.DataFrame, target: str='quality.fid'):
    X=df.select_dtypes(include='number').fillna(0)
    y=X.iloc[:,0]
    m=RandomForestRegressor(n_estimators=10, random_state=0).fit(X,y)
    return m, float(m.score(X,y))
