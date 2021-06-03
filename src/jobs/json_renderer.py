from json.encoder import encode_basestring_ascii, encode_basestring, INFINITY, _make_iterencode

from rest_framework.renderers import JSONRenderer
from rest_framework.utils.encoders import JSONEncoder


# default renderer cannot handle JSON inf values
# https://stackoverflow.com/questions/35939464/django-rest-framework-json-data-monkey-patching
class CustomJSONEncoder(JSONEncoder):

    def iterencode(self, o, _one_shot=False):
        """Encode the given object and yield each string
        representation as available.

        :param o:
        :param _one_shot:
        :return:

        For example::

            for chunk in JSONEncoder().iterencode(bigobject):
                mysocket.write(chunk)

        """
        # Hack to enforce
        c_make_encoder = None
        if self.check_circular:
            markers = {}
        else:
            markers = None
        if self.ensure_ascii:
            _encoder = encode_basestring_ascii
        else:
            _encoder = encode_basestring

        def floatstr(o, allow_nan=True, _repr=lambda o: format(o, '.4f'), _inf=INFINITY, _neginf=-INFINITY):
            """Convert float number into a string

            :param o:
            :param allow_nan:
            :param _repr:
            :param _inf:
            :param _neginf:
            :return:
            """
            # Check for specials.  Note that this type of test is processor
            # and/or platform-specific, so do tests which don't depend on the
            # internals.

            if o != o:
                text = '0'
            elif o == _inf:
                text = '1000000'  # infinity is 1000000
            elif o == _neginf:
                text = '-1000000'
            else:
                return _repr(o)

            if not allow_nan:
                raise ValueError(
                    "Out of range float values are not JSON compliant: " +
                    repr(o))

            return text

        if _one_shot and c_make_encoder is not None and self.indent is None:
            _iterencode = c_make_encoder(
                markers, self.default, _encoder, self.indent,
                self.key_separator, self.item_separator, self.sort_keys,
                self.skipkeys, self.allow_nan)  # TODO: fix call to non-callable object
        else:
            _iterencode = _make_iterencode(
                markers, self.default, _encoder, self.indent, floatstr,
                self.key_separator, self.item_separator, self.sort_keys,
                self.skipkeys, _one_shot)
        return _iterencode(o, 0)


class PalJSONRenderer(JSONRenderer):
    encoder_class = CustomJSONEncoder
