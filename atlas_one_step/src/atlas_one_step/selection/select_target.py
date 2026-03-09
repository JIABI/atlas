def select_target(candidates):
    return sorted(candidates,key=lambda c:c.get("pathology_score",1e9))[0]
