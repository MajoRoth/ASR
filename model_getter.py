from models.linear_acoustic_model import LinearAcoustic

def get_model(args, cfg):
    if cfg.model == "LinearAcoustic":
        return LinearAcoustic(cfg)
    else:
        raise NotImplementedError