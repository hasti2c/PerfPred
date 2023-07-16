import modeling.functions as F
import util as U
from slicing.variable import Variable as V


if U.EXPERIMENT_TYPE == "one stage":
    VARS = {
        "A": [[V.TRAIN_SIZE]],
        "B": [[V.TRAIN_JSD]],
        "C": [[V.FEA_DIST], [V.INV_DIST], [V.PHO_DIST], [V.SYN_DIST], [V.GEN_DIST], [V.GEO_DIST],
            [V.INV_DIST, V.PHO_DIST], [V.INV_DIST, V.SYN_DIST], [V.PHO_DIST, V.SYN_DIST], [V.GEN_DIST, V.GEO_DIST], 
            [V.INV_DIST, V.PHO_DIST, V.SYN_DIST], [V.FEA_DIST, V.GEN_DIST, V.GEO_DIST], 
            [V.FEA_DIST, V.INV_DIST, V.PHO_DIST, V.SYN_DIST], 
            [V.INV_DIST, V.PHO_DIST, V.SYN_DIST, V.GEN_DIST, V.GEO_DIST],
            [V.FEA_DIST, V.INV_DIST, V.PHO_DIST, V.SYN_DIST, V.GEN_DIST, V.GEO_DIST]]
    }
else:
    VARS = {
        "A": [[V.TRAIN1_SIZE], [V.TRAIN2_SIZE], [V.TRAIN1_SIZE, V.TRAIN2_SIZE]],
        "B": [[V.TRAIN1_JSD], [V.TRAIN2_JSD], [V.TRAIN1_JSD, V.TRAIN2_JSD]],
        "C": [[V.FEA_DIST], [V.INV_DIST], [V.PHO_DIST], [V.SYN_DIST], [V.GEN_DIST], [V.GEO_DIST],
            [V.INV_DIST, V.PHO_DIST], [V.INV_DIST, V.SYN_DIST], [V.PHO_DIST, V.SYN_DIST], [V.GEN_DIST, V.GEO_DIST], 
            [V.INV_DIST, V.PHO_DIST, V.SYN_DIST], [V.FEA_DIST, V.GEN_DIST, V.GEO_DIST], 
            [V.FEA_DIST, V.INV_DIST, V.PHO_DIST, V.SYN_DIST], 
            [V.INV_DIST, V.PHO_DIST, V.SYN_DIST, V.GEN_DIST, V.GEO_DIST],
            [V.FEA_DIST, V.INV_DIST, V.PHO_DIST, V.SYN_DIST, V.GEN_DIST, V.GEO_DIST]]
    }

SPLITS = {
    "1": [None],
    "2": [[V.LANG], [V.TEST]] # TODO splits=[]
}

MODELS = {
    "linear": {'f': F.linear},
    "poly2": {'f': F.polynomial, 'k': 2},
    "poly3": {'f': F.polynomial, 'k': 3},
    "exp": {'f': F.exponential},
    "log": {'f': F.logarithmic},
    "power": {'f': F.power},
    "mult": {'f': F.multiplicative},
    "hybrid_mult": {'f': F.hybrid_multiplicative, 'bounds': (-1000, 1000)},
    "am": {'f': F.arithmetic_mean_linear},
    "gm": {'f': F.geometric_mean_linear},
    "hm": {'f': F.harmonic_mean_linear},
}
# TODO dont use mean models for single var