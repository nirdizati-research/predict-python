from enum import Enum

from django.db import models
from jsonfield.fields import JSONField

from src.common.models import CommonModel
from src.jobs.models import Job
from src.predictive_model.models import PredictiveModel
from src.split.models import Split


class ExplanationTypes(Enum):
    SHAP = 'shap'
    LIME = 'lime'
    ANCHOR = 'anchor'


EXPLANATION_TYPE_MAPPINGS = (
    (ExplanationTypes.SHAP.value, 'shap'),
    (ExplanationTypes.LIME.value, 'lime'),
    (ExplanationTypes.ANCHOR.value, 'anchor')
)


class Explanation(CommonModel):
    type = models.CharField(choices=EXPLANATION_TYPE_MAPPINGS, default='shap',
                            max_length=max(len(el[1]) for el in EXPLANATION_TYPE_MAPPINGS) + 1, null=True, blank=True)
    split = models.ForeignKey(Split, on_delete=models.DO_NOTHING, null=True)
    predictive_model = models.ForeignKey(PredictiveModel, on_delete=models.DO_NOTHING, null=True)
    job = models.ForeignKey(Job, on_delete=models.DO_NOTHING, null=True, default=None)
    results = JSONField(default=dict)

    def to_dict(self):
        return {
            'type': self.type,
            'split': self.split,
            'predictive_model': self.predictive_model,
            'results': self.results
        }
