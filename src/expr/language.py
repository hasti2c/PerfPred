from trial import *

path_C = os.path.join("results", "1C")

class LanguageTrial(SingleVar):
  """ Trial with factor geographical, genetic, syntactic, phonological,
      inventory, or featural.

#   == Attributes ==
#     alt_var: The other factor among j1, j2 not used as slice var.
#   Otherwise same as SingleVar, but with pre-set values:
#     slice_vars = ["train set n", "test set"]
#     x_vars = ["train set n jsd"]
  """
  def __init__(self, type: str,
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
      type: Which l2v distance to use.
      trial: Name of trial. Used as subdirectory name.
    """
    super().__init__(["language to"], f, init, fixed_init=fixed_init, 
                     bounds=bounds, loss=loss, pars=pars, 
                     path=os.path.join(path_C, var_names[type], trial),
                     xvars=[type], verbose=verbose)

#   def init_analyzer(self):
#     """ Initalizes self.analyzer with attributes:
#       plot_horiz = ["train set 1 size", "train set 2 size"]
#       bar_horiz = ["train set 2", "language to"]
#       scatter_horiz = ["train set 1 size", "train set 2 size"]
#       scatter_seper = [[], ["language to"]]
#     """
#     numer = ["train set 1 size", "train set 2 size"] # TODO can have alt jsd
#     super().init_analyzer(plot_horiz=numer, scatter_horiz=numer,
#                           bar_horiz=[self.alt_var, "language to"],
#                           scatter_seper=[[], ["language to"]])

#   def analyze_all(self, run_plots=True):
#     """ Calls init_analyzer and super().analyze_all(). """
#     self.init_analyzer()
#     super().analyze_all(run_plots=run_plots)
