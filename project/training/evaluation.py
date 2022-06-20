import sklearn
from abc import ABC, abstractmethod


class Metrics(ABC):

    @abstractmethod
    def calculate(self, model_output, y):
        pass

    def create_logs(self, model_output, y ,_string):
        metrics_dict = {}
        metrics_lst = self.classification(model_output, y)

        for metric ,name  in zip(metrics_lst, self.metrics_names):
            metrics_dict[name] = metric

        return metrics_dict



class ClassificationMetrics:
    metrics_names = ['accuracy', 'f1_score', 'precision_score', 'recall_score']

    def calculate(self, model_output, y):
        predictions = model_output.data.cpu().argmax(dim=1)
        y = y.cpu()
        accuracy = sklearn.metrics.accuracy_score(y ,predictions,  average = 'macro')
        f1_score = sklearn.metrics.f1_score(y ,predictions, average = 'macro')
        precision_score = sklearn.metrics.precision_score(y ,predictions, average = 'macro')
        recall_score = sklearn.metrics.recall_score(y ,predictions, average = 'macro')

        return [accuracy, f1_score, precision_score, recall_score ]