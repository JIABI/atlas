def label_phase(pathology):
    s=pathology.get("pathology_score",0); return "trainable" if s<10 else "rescue" if s<20 else "failure"
