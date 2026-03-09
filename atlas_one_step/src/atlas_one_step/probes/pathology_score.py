def aggregate_pathology(d): return sum(v for v in d.values() if isinstance(v,(int,float)))
