from .dataset import Dataset
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.linear_model import LogisticRegression as lr
from sklearn.metrics import classification_report
import numpy as np

class Evaluator():
    
    def __init__(self, dataset, N_experiments=2):
        self.dataset = dataset
        self.models = {'rfc':rfc,
                       'lr':lr}
        self.dataset.prepare_dataset(N=N_experiments)
        
    def get_evaluation(self, model_list=[], N_experiments = 2):
        evaluation = {'model':[],
                      'accuracy':[]}
        X_train , y_train, X_test, y_test = self.dataset.get_dataset()
        
        if model_list == []:
            model_list = self.models.keys()
            
        for model_name in model_list:
            model = self.models[model_name]()
            model.fit(X_train, y_train)
            evaluation['model'].append(model_name)
            preds = model.predict(X_test)
            accuracy = np.mean(y_test==preds)
            report = classification_report(y_test, preds, output_dict=True)
            print(report)
            evaluation['accuracy'].append(accuracy)
        return evaluation

        
    
