from .unet import SimpleUNet
def build_model(cfg):
    return SimpleUNet(in_channels=cfg.get("in_channels",3), out_channels=cfg.get("out_channels",3), base_channels=cfg.get("base_channels",32))
