from .evaluator import Evaluator
from .dataset import Dataset
import pandas as pd
dataset = Dataset()
evaluator = Evaluator(dataset, N_experiments=2)
print(pd.DataFrame.from_dict(evaluator.get_evaluation(model_list=['rfc'])))