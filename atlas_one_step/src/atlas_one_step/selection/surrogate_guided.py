def surrogate_guided(candidates): return max(candidates,key=lambda c:c.get("score",0.0))
