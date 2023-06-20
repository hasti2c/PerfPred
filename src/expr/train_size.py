from trial import *

path_A = os.path.join("results", "1A")

class SingleSizeTrial(SingleVarTrial):
  """ Trial with factor D1 or D2.

  == Attributes ==
    alt_var: The other factor among D1, D2 not used as slice var.
  Otherwise same as SingleVarTrial, but with pre-set values:
    slice_vars = ["train set n size"]
    x_vars = ["train set n size"]
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
    """ Initializes a SingleTrain trial.
    == Arguments ==
      n: Whether to use D1 or D2.
         If n == 1, uses train set 1. If n == 2, uses train set 2.
      trial: Name of trial. Used as subdirectory name.
    """
    super().__init__([Var.TRAIN1_SIZE if n == 1 else Var.TRAIN2_SIZE], f, init,
                     fixed_init=fixed_init, bounds=bounds, loss=loss, pars=pars,
                     path=os.path.join(path_A, "size" + str(n), trial),
                     xvars=[Var.TRAIN1_SIZE if n == 1 else Var.TRAIN2_SIZE], 
                     verbose=verbose)
    self.alt_var = Var.TRAIN2_SIZE if n == 1 else Var.TRAIN1_SIZE

  def init_analyzer(self):
    """ Initalizes self.analyzer with attributes:
      plot_horiz = ["train set (1 - n) size"]
      bar_horiz = ["train set 1", "train set 2", "test set", "language to"]
      scatter_horiz = ["train set (1 - n) size"]
      scatter_seper = [[], ["language to"]]
    """
    super().init_analyzer([self.alt_var], [self.alt_var],
                          [Var.TRAIN1, Var.TRAIN2, Var.TEST, Var.LANG],
                          [[], [Var.LANG]])

  def analyze_all(self, run_plots=True):
    """ Calls init_analyzer and super().analyze_all(). """
    self.init_analyzer()
    super().analyze_all(run_plots=run_plots)


class DoubleSizeTrial(DoubleVarTrial):
  """ Trial with factors D1+D2.

  == Attributes ==
  Same as DoubleVarTrial, but with pre-set values:
    slice_vars = ["train set 1 size", "train set 2 size"]
    x_vars = ["train set 1 size", "train set 2 size"]
  """
  def __init__(self, f: T.Callable[[FloatArray, FloatArray], FloatArray],
               init: T.Union[FloatArray, list[FloatArray]],
               fixed_init: bool=True,
               bounds: T.Optional[tuple[list[float]]]=None,
               loss: str='soft_l1',
               pars: list[str]=[],
               trial: T.Optional[str]=None,
               verbose: int=1) -> None:
    super().__init__([Var.TRAIN1_SIZE, Var.TRAIN2_SIZE], f, init,
                     fixed_init=fixed_init, bounds=bounds, loss=loss,
                     pars=pars, path=os.path.join(path_A, "sizes", trial),
                     xvars=[Var.TRAIN1_SIZE, Var.TRAIN2_SIZE],
                     verbose=verbose)

  def init_analyzer(self):
    """ Initalizes self.analyzer with attributes:
      plot_horiz = []
      bar_horiz = ["train set 1", "train set 2", "test set", "language to"]
      scatter_horiz = []
      scatter_seper = []
    """
    super().init_analyzer(bar_horiz=[Var.TRAIN1, Var.TRAIN2, Var.TEST, Var.LANG])

  def analyze_all(self, run_plots=True):
    """ Calls init_analyzer and super().analyze_all(). """
    self.init_analyzer()
    super().analyze_all(run_plots=run_plots)