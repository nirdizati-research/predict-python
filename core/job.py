"""
Type Classification
{
      "clustering": "None",
      "status": "completed",
      "run": "KNN_simpleIndex_None",
      "log": "Production.xes",
      "classification": "KNN",
      "encoding": "simpleIndex",
      "timestamp": "Oct 03 2017 13:26:53",
      "rule": "remaining_time",
      "prefix": 0,
      "threshold": "default",
      "type": "Classification",
      "uuid": "f1725991-9673-4ead-b964-bd1f43596f69"
    }

Type Regression
{
      "clustering": "kmeans",
      "status": "completed",
      "run": "linear_simpleIndex_kmeans",
      "log": "Production.xes",
      "encoding": "simpleIndex",
      "timestamp": "Oct 03 2017 13:50:52",
      "prefix": 0,
      "type": "Regression",
      "regression": "linear",
      "uuid": "803be08e-f807-459a-a025-4f33a669bf68"
    }
Type NextActivity
{
      "clustering": "None",
      "status": "queued",
      "run": "KNN_simpleIndex_None",
      "log": "Production.xes",
      "classification": "KNN",
      "encoding": "simpleIndex",
      "timestamp": "Oct 09 2017 10:57:45",
      "prefix": 0,
      "type": "NextActivity",
      "uuid": "1830e0ff-ebac-4396-8f54-5f7ad9247132"
    }
"""
import time
import uuid
import copy

TIME_FORMAT = "%b %d %Y %H:%M:%S"


class Job(object):
    """Dict containing training job options

    NextActivity is Classification without threshold and rule.
    """

    def __init__(self, d):
        self.__dict__ = d

    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError("No such attribute: " + name)

    # Public methods
    def get_results_path(self):
        """Path to csv"""
        if self.type == "Classification":
            return self.__get_results_class_path()
        elif self.type == "NextActivity":
            return self.__get_results_next_activity_path()
        elif self.type == "Regression":
            return self.__get_results_regg_path()

    def get_results_general(self):
        """Path to General.csv"""
        if self.type == "Classification":
            return self.__get_results_class_general()
        elif self.type == "NextActivity":
            return self.__get_results_next_activity_general()
        elif self.type == "Regression":
            return self.__get_results_regg_general()

    def method_val(self):
        """For results"""
        return self.get_run() + "_clustering"

    def prediction_model_dir(self):
        """Only for regression"""
        return 'core_predictionmodels/' + self.log + '/' + str(self.prefix)

    def prediction_model_path(self):
        """Only for regression"""
        return 'core_predictionmodels/' + self.log + '/' + str(
            self.prefix) + '/' + self.regression + '_' + self.clustering + '.pkl'

    def get_results_dir(self):
        """Directory containing results"""
        if self.type == "Classification":
            return self.__get_results_class_dir()
        elif self.type == "NextActivity":
            return self.__get_results_next_activity_dir()
        elif self.type == "Regression":
            return self.__get_results_regg_dir()

    def get_encoded_file_path(self):
        path = self.encoding + "_" + self.log + "_" + str(self.prefix) + ".csv"
        if self.type == "NextActivity":
            return "core_encodedFiles/next_activity/" + path
        else:
            return "core_encodedFiles/" + path

    def get_run(self):
        """Defines job identity"""
        if self.type == "Classification":
            return run_classification(self.classification, self.encoding, self.clustering)
        elif self.type == "nextActivity":
            return run_classification(self.classification, self.encoding, self.clustering)
        elif self.type == "Regression":
            return run_regression(self.regression, self.encoding, self.clustering)

    # Private methods
    def __get_results_class_path(self):
        """Path to csv"""
        return 'core_results_class/' + self.log + '/' + str(self.prefix) + '/' + self.rule + '/' + str(
            self.threshold) + '/' + \
               self.classification + '_' + self.encoding + '_' + self.clustering + '_clustering.csv'

    def __get_results_regg_path(self):
        return 'core_results_regg/' + self.log + '/' + str(
            self.prefix) + '/' + self.regression + '_' + self.encoding + '_' + self.clustering + '_clustering.csv'

    def __get_results_next_activity_path(self):
        return 'core_results_class/' + self.log + '/' + str(
            self.prefix) + '/' + self.classification + '_' + self.encoding + '_' + self.clustering + '_clustering.csv'

    def __get_results_class_dir(self):
        return 'core_results_class/' + self.log + '/' + str(self.prefix) + '/' + self.rule + '/' + str(self.threshold)

    def __get_results_next_activity_dir(self):
        return 'core_results_class/' + self.log + '/' + str(self.prefix)

    def __get_results_regg_dir(self):
        return 'core_results_regg/' + self.log + '/' + str(self.prefix)

    def __get_results_class_general(self):
        """Path to General.csv"""
        return 'core_results_class/' + self.log + '/' + str(self.prefix) + '/' + self.rule + '/' + str(
            self.threshold) + '/General.csv'

    def __get_results_regg_general(self):
        """Path to General.csv"""
        return 'core_results_regg/' + self.log + '/' + str(self.prefix) + '/General.csv'

    def __get_results_next_activity_general(self):
        """Path to General.csv"""
        return 'core_results_class/' + self.log + '/' + str(self.prefix) + '/General.csv'


def build_job_dict(config, run, clustering, encoding, classification='', regression=''):
    """Create a dict from existing config"""
    json = copy.deepcopy(config)
    json['uuid'] = str(uuid.uuid4())
    json['timestamp'] = time.strftime(TIME_FORMAT, time.localtime())
    json['status'] = 'queued'
    json['run'] = run
    if classification != '':
        json['classification'] = classification
    if regression != '':
        json['regression'] = regression
    # rewrite list as 1
    json['clustering'] = clustering
    json['encoding'] = encoding
    return json


def run_classification(classification, encoding, clustering):
    return classification + '_' + encoding + '_' + clustering


def run_regression(regression, encoding, clustering):
    return regression + '_' + encoding + '_' + clustering
