from rest_framework.parsers import BaseParser


class CustomXMLParser(BaseParser):
    """
    Plain text parser.
    """
    media_type = 'text/plain'

    def parse(self, stream, media_type=None, parser_context=None):
        """
        Simply return a string representing the body of the request.

        :param stream:
        :param media_type:
        :param parser_context:
        :return:
        """
        return stream.read()
