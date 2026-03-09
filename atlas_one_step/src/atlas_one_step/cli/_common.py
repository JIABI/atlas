from types import SimpleNamespace

def cfg():
    return SimpleNamespace(atlas=SimpleNamespace(name="sweep_line"), target=SimpleNamespace(name="line_x0_u"), train=SimpleNamespace(name="coupled",epochs=1,lr=1e-3,mu=0.1,tau=0.1), dataset=SimpleNamespace(batch_size=4,resolution=32), model={"in_channels":3,"out_channels":3,"base_channels":16})
