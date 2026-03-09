import pandas as pd
from atlas_one_step.atlas.surrogate import fit_surrogate
def test_surrogate():
 m,s=fit_surrogate(pd.DataFrame({"a":[1,2],"b":[3,4]})); assert s<=1
