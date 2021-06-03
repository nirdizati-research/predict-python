from ws4redis.publisher import RedisPublisher
from ws4redis.redis_store import RedisMessage

from src.jobs.serializers import JobSerializer
from src.logs.serializers import LogSerializer

redis_publisher = RedisPublisher(facility='default', broadcast=True)
from rest_framework.renderers import JSONRenderer  # TODO: check module import not at top level


def publish(object):
    """
    Publish an object to websocket listeners

    :param object: A Django predictive_model
    :return: {type: object class name, data: OBJECT}
    """
    message = RedisMessage(_serializer(object))
    redis_publisher.publish_message(message)


def _serializer(object):
    """Assumed to be Django models

    :param object:
    :return:
    """
    name = object.__class__.__name__
    if name == 'Log':
        data = LogSerializer(object).data
    elif name == 'Job':
        data = JobSerializer(object).data
    #elif name == 'XTrace':
     #   data = TraceSerializer(object).data
    else:
        raise NotImplementedError("Websocket not implemented for class ".format(name))
    return JSONRenderer().render({'type': name, 'data': data})
