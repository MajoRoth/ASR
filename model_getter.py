from models.linear_acoustic_model import LinearAcoustic
from models.DeepSpeech2 import DS2LargeModel, DS2SmallModel, DS2ToyModel
from models.mini_lstm_model import MiniLSTM

def get_model(args, cfg):
    if cfg.model == "LinearAcoustic":
        return LinearAcoustic(cfg)
    elif cfg.model == 'DeepSpeech2_Large':
        return DS2LargeModel(cfg)
    elif cfg.model == 'DeepSpeech2_Small':
        return DS2SmallModel(cfg)
    elif cfg.model == 'DeepSpeech2_Toy':
        return DS2ToyModel(cfg)
    elif cfg.model == 'MiniLSTM':
        return MiniLSTM(cfg)
    else:
        raise NotImplementedError

