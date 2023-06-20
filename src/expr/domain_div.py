from trial import *

path_B = os.path.join("results", "1B")

class SingleDomainTrial(SingleVarTrial):
  """ Trial with factor j1 or j2.

  == Attributes ==
    alt_var: The other factor among j1, j2 not used as slice var.
  Otherwise same as SingleVarTrial, but with pre-set values:
    slice_vars = ["train set n", "test set"]
    x_vars = ["train set n jsd"]
  """
  def __init__(self, n: int,
               f: T.Callable[[FloatArray, FloatArray], FloatArray],
               init: T.Union[FloatArray, list[FloatArray]],
               fixed_init: bool=True,
               bounds: T.Optional[tuple[list[float]]]=None,
               loss: str='soft_l1',
               pars: list[str]=[],
               trial: T.Optional[str]=None,
               verbose: int=1) -> None:
    """ Initializes a SingleDomainTrial trial.
    == Arguments ==
      n: Whether to use j1 or j2.
         If n == 1, uses train set 1. If n == 2, uses train set 2.
      trial: Name of trial. Used as subdirectory name.
    """
    super().__init__([Var.TRAIN1 if n == 1 else Var.TRAIN2, "test set"], f, init,
                     fixed_init=fixed_init, bounds=bounds, loss=loss,
                     pars=pars, path=os.path.join(path_B, "jsd" + str(n), trial),
                     xvars=[Var.TRAIN1_JSD if n == 1 else Var.TRAIN2_JSD, Var.TEST], 
                     verbose=verbose)
    self.alt_var = Var.TRAIN2 if n == 1 else Var.TRAIN1

  def init_analyzer(self):
    """ Initalizes self.analyzer with attributes:
      plot_horiz = ["train set 1 size", "train set 2 size"]
      bar_horiz = ["train set 2", "language to"]
      scatter_horiz = ["train set 1 size", "train set 2 size"]
      scatter_seper = [[], ["language to"]]
    """
    numer = [Var.TRAIN1_SIZE, Var.TRAIN2_SIZE] # TODO can have alt jsd
    super().init_analyzer(plot_horiz=numer, scatter_horiz=numer,
                          bar_horiz=[self.alt_var, Var.LANG],
                          scatter_seper=[[], [Var.LANG]])

  def analyze_all(self, run_plots=True):
    """ Calls init_analyzer and super().analyze_all(). """
    self.init_analyzer()
    super().analyze_all(run_plots=run_plots)


class DoubleDomainTrial(DoubleVarTrial):
  """ Trial with factors j1 + j2.

  == Attributes ==
  Same as DoubleVarTrial, but with pre-set values:
    slice_vars = ["train set 1", "train set 2", "test set"]
    x_vars = ["train set 1 jsd", "train set 2 jsd"]
  """
  def __init__(self, f: T.Callable[[FloatArray, FloatArray], FloatArray],
               init: T.Union[FloatArray, list[FloatArray]],
               fixed_init: bool=True,
               bounds: T.Optional[tuple[list[float]]]=None,
               loss: str='soft_l1',
               pars: list[str]=[],
               trial: T.Optional[str]=None,
               verbose: int=1) -> None:
    super().__init__([Var.TRAIN1, Var.TRAIN2, Var.TEST], f, init,
                     fixed_init=fixed_init, bounds=bounds, loss=loss,
                     pars=pars, path=os.path.join(path_B, "jsds", trial),
                     xvars=[Var.TRAIN1_JSD, Var.TRAIN2_JSD],
                     label_func=lambda i: Var.TRAIN1 if i == 1 else Var.TRAIN2,
                     verbose=verbose)

  def init_analyzer(self): # TODO should have plot and scatter
    """ Initalizes self.analyzer with attributes:
      bar_horiz = [[], ["language to"]]
    """
    super().init_analyzer(bar_horiz=[Var.LANG])

  def analyze_all(self, run_plots=True):
    """ Calls init_analyzer and super().analyze_all(). """
    self.init_analyzer()
    super().analyze_all(run_plots=run_plots)