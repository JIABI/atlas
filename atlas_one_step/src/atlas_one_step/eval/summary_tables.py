import pandas as pd
def make_table(df:pd.DataFrame,out): df.to_csv(out,index=False)
