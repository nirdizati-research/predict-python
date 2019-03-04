import abc

from django.db import models


class CommonModel(models.Model):
    @abc.abstractmethod
    def to_dict(self) -> dict:
        return {}

    def __str__(self) -> dict:
        return self.to_dict()
