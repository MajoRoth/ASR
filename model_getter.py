from models.linear_acoustic_model import LinearAcoustic

def get_model(args, cfg):
    return LinearAcoustic(cfg)