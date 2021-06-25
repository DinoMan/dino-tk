import torch
from torchaudio import functional as F
import math


class AmplitudeToDB(torch.nn.Module):
    def __init__(self, stype='power', top_db=None, normalize=True):
        super(AmplitudeToDB, self).__init__()
        if top_db is not None and top_db < 0:
            raise ValueError('top_db must be positive value')
        self.top_db = top_db
        self.multiplier = 10.0 if stype == 'power' else 20.0
        self.amin = 1e-10
        self.normalize = normalize

    def forward(self, x):
        ref_value = torch.max(x)
        db_multiplier = math.log10(max(self.amin, ref_value))
        x = F.amplitude_to_DB(x, self.multiplier, self.amin, db_multiplier, self.top_db)
        if self.normalize:
            x /= 80
            x = (x + 0.5) / 0.5

        return x
