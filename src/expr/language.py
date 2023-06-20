from trial import *

path_C = os.path.join("results", "1C")

class LanguageTrial(Trial):
  """ Trial with factor geographical, genetic, syntactic, phonological,
      inventory, or featural.

#   == Attributes ==
#     alt_var: The other factor among j1, j2 not used as slice var.
#   Otherwise same as SingleVarTrial, but with pre-set values:
#     slice_vars = ["train set n", "test set"]
#     x_vars = ["train set n jsd"]
  """
  def __init__(self, dists: Var,
               f: T.Callable[[FloatArray, FloatArray], FloatArray],
               init: T.Union[FloatArray, list[FloatArray]],
               fixed_init: bool=True,
               bounds: T.Optional[tuple[list[float]]]=None,
               loss: str='soft_l1',
               pars: list[str]=[],
               subpath: T.Optional[str]=None,
               plot_f: T.Callable[[Slice, FloatArray], None]=None,
               verbose: int=1) -> None:
    """ Initializes a Language trial.
    == Arguments ==
      dists: List of distances to use
      trial: Name of trial. Used as subdirectory name.
    """
    super().__init__([Var.LANG], f, init, fixed_init=fixed_init, 
                     bounds=bounds, loss=loss, pars=pars, 
                     path=os.path.join(path_C, subpath), plot_f=plot_f,
                     xvars=dists, label_func=lambda i: Var.LANG, 
                     verbose=verbose)
        
  def init_analyzer(self):
    """ Initalizes self.analyzer with attributes:
      plot_horiz = ["train set 1 size", "train set 2 size"]
      bar_horiz = ["train set 1", "train set 2", "test set"]
      scatter_horiz = ["train set 1 size", "train set 2 size"]
      scatter_seper = [[]]
    """
    numer = [Var.TRAIN1_SIZE, Var.TRAIN2_SIZE] # TODO can have jsd
    super().init_analyzer(plot_horiz=numer, scatter_horiz=numer,
                          bar_horiz=[Var.TRAIN1, Var.TRAIN2, Var.TEST],
                          scatter_seper=[[]])

  def analyze_all(self, run_plots=True):
    """ Calls init_analyzer and super().analyze_all(). """
    self.init_analyzer()
    super().analyze_all(run_plots=run_plots)



class SingleLanguageTrial(LanguageTrial, SingleVarTrial):
  def __init__(self, dist: Var,
               f: T.Callable[[FloatArray, FloatArray], FloatArray],
               init: T.Union[FloatArray, list[FloatArray]],
               fixed_init: bool=True,
               bounds: T.Optional[tuple[list[float]]]=None,
               loss: str='soft_l1',
               pars: list[str]=[],
               trial: T.Optional[str]=None,
               verbose: int=1) -> None:
    """ Initializes a Language trial.
    == Arguments ==
      dist: Distances to use
      trial: Name of trial. Used as subdirectory name.
    """
    super().__init__([dist], f, init, fixed_init=fixed_init, 
                     bounds=bounds, loss=loss, pars=pars, 
                     subpath=os.path.join("1var", dist.short, trial), verbose=verbose)
    

class DoubleLanguageTrial(LanguageTrial, DoubleVarTrial):
  def __init__(self, dists: Var,
               f: T.Callable[[FloatArray, FloatArray], FloatArray],
               init: T.Union[FloatArray, list[FloatArray]],
               fixed_init: bool=True,
               bounds: T.Optional[tuple[list[float]]]=None,
               loss: str='soft_l1',
               pars: list[str]=[],
               trial: T.Optional[str]=None,
               verbose: int=1) -> None:
    """ Initializes a Language trial.
    == Arguments ==
      dist: Distances to use
      trial: Name of trial. Used as subdirectory name.
    """
    super().__init__(dists, f, init, fixed_init=fixed_init, 
                     bounds=bounds, loss=loss, pars=pars, 
                     subpath=os.path.join("2var", "+".join([var.short for var in dists]), trial),
                     verbose=verbose)