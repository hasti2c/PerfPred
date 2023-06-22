from expr import func
from expr.train_size import *
from expr.domain_div import *
from expr.language import *

size1 = [
  SingleSizeTrial(1, Model.linear(1), trial="trial1"),
  SingleSizeTrial(1, Model(func.log_single, np.array([0.1, 0.1, 0.1]),
                           bounds=([-np.inf, 0, -np.inf], [np.inf, np.inf, np.inf]),
                           pars=["C", "alpha", "beta"]), trial="trial2"),
  SingleSizeTrial(1, Model(func.recip_single, np.array([0, 0, -1]),
                           bounds=([-np.inf, 0, -np.inf], [np.inf, np.inf, np.inf]),
                           pars=["alpha", "C", "p"]), trial="trial3")
]

size2 = [
  SingleSizeTrial(2, Model.linear(1), trial="trial1"),
  SingleSizeTrial(2, Model(func.log_single, np.array([0.1, 0.1, 0.1]),
                           bounds=([-np.inf, 0, -np.inf], [np.inf, np.inf, np.inf]),
                           pars=["C", "alpha", "beta"]), trial="trial2"),
  SingleSizeTrial(2, Model(func.recip_single, np.array([0, 0, -1]),
                           bounds=([-np.inf, 0, -np.inf], [np.inf, np.inf, np.inf]),
                           pars=["alpha", "C", "p"]), trial="trial3")
]

sizes = [
  DoubleSizeTrial(Model.linear(2), trial="trial1"),
  DoubleSizeTrial(Model(func.product_double, np.zeros(4),
                        bounds=([-np.inf, 0, 0, 0], [0, np.inf, np.inf, np.inf]),
                        pars=["alpha", "p1", "p2", "C"]), trial="trial2"),
  DoubleSizeTrial(Model(func.depend_double, np.zeros(5),
                        bounds=([-np.inf, -np.inf, 0, 0, 0], [0, 0, np.inf, np.inf, np.inf]),
                        pars=["alpha1", "alpha2", "p1", "p2", "C"]), trial="trial3")
]

A_trials = size1 + size2 + sizes

jsd1 = [
  SingleDomainTrial(1, Model.linear(1), trial="trial1"),
]

jsd2 = [
  SingleDomainTrial(2, Model.linear(1), trial="trial1"),
]

jsds = [
  DoubleDomainTrial(Model.linear(2), trial="trial1")
]

B_trials = jsd1 + jsd2 + jsds

lang_1var = [
  LanguageTrial([Var.FEA_DIST], Model.linear(1), trial="trial1"),
  LanguageTrial([Var.INV_DIST], Model.linear(1), trial="trial1"),
  LanguageTrial([Var.PHO_DIST], Model.linear(1), trial="trial1"),
  LanguageTrial([Var.SYN_DIST], Model.linear(1), trial="trial1"),
  LanguageTrial([Var.GEN_DIST], Model.linear(1), trial="trial1"),
  LanguageTrial([Var.GEO_DIST], Model.linear(1), trial="trial1")
]

lang_2var = [
  LanguageTrial([Var.INV_DIST, Var.PHO_DIST], Model.linear(2), trial="trial1"),
  LanguageTrial([Var.INV_DIST, Var.SYN_DIST], Model.linear(2), trial="trial1"),
  LanguageTrial([Var.PHO_DIST, Var.SYN_DIST], Model.linear(2), trial="trial1"),
  LanguageTrial([Var.GEN_DIST, Var.GEO_DIST], Model.linear(2), trial="trial1"),
  LanguageTrial([Var.FEA_DIST, Var.INV_DIST], Model.linear(2), trial="trial1"),
  LanguageTrial([Var.FEA_DIST, Var.PHO_DIST], Model.linear(2), trial="trial1"),
  LanguageTrial([Var.FEA_DIST, Var.SYN_DIST], Model.linear(2), trial="trial1"),
  LanguageTrial([Var.FEA_DIST, Var.GEN_DIST], Model.linear(2), trial="trial1"),
  LanguageTrial([Var.FEA_DIST, Var.GEO_DIST], Model.linear(2), trial="trial1")
]

lang_3var = [
  LanguageTrial([Var.INV_DIST, Var.PHO_DIST, Var.SYN_DIST], Model.linear(3), trial="trial1"),  
  LanguageTrial([Var.FEA_DIST, Var.INV_DIST, Var.PHO_DIST], Model.linear(3), trial="trial1"),
  LanguageTrial([Var.FEA_DIST, Var.INV_DIST, Var.SYN_DIST], Model.linear(3), trial="trial1"),
  LanguageTrial([Var.FEA_DIST, Var.PHO_DIST, Var.SYN_DIST], Model.linear(3), trial="trial1"),
  LanguageTrial([Var.FEA_DIST, Var.GEN_DIST, Var.GEO_DIST], Model.linear(3), trial="trial1"),
]

lang_many_var = [
  LanguageTrial([Var.FEA_DIST, Var.INV_DIST, Var.PHO_DIST, Var.SYN_DIST], Model.linear(4), trial="trial1"),
  LanguageTrial([Var.INV_DIST, Var.PHO_DIST, Var.SYN_DIST, Var.GEN_DIST, Var.GEO_DIST], Model.linear(5), trial="trial1"),
  LanguageTrial([Var.FEA_DIST, Var.INV_DIST, Var.PHO_DIST, Var.SYN_DIST, Var.GEN_DIST, Var.GEO_DIST], Model.linear(6), trial="trial1")
]

C_trials = lang_1var + lang_2var + lang_3var + lang_many_var

all_trials = A_trials + B_trials + C_trials

linear_trials = [size1[0], size2[0], sizes[0]] + B_trials + C_trials

for trial in all_trials:
    trial.read_all_fits()