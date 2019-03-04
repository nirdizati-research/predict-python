import abc

from django.db import models


class CommonModel(models.Model):
    @abc.abstractmethod
    def to_dict(self) -> dict:
        return {}

    def get_full_dict(self):
        if self.__class__.__name__ != 'CommonModel':  # TODO: improve on this
            return {**super(self.__class__, self).to_dict(), **self.to_dict()}
        return {}

    def __str__(self):
        full_dict = self.get_full_dict()
        return '{' + ', '.join(['{key}: {value}'.format(key=key, value=full_dict.get(key)) for key in full_dict]) + '}'

    class Meta:
        abstract = True
