from django.db import models


class PredictiveModel(models.Model):
    """Container of Classification to be shown in frontend"""

    def to_dict(self):
        return {}
