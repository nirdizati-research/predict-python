import copy

from django.db import models


def duplicate_orm_row(obj: models.Model):
    """Returns a copy of the given object

    :param obj:
    :return:
    """
    cloned = copy.deepcopy(obj)
    cloned.id = None
    cloned.pk = None
    try:
        delattr(cloned, '_prefetched_objects_cache')
    except AttributeError:
        pass
    cloned.save()
    return cloned
