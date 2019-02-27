from rest_framework import mixins, generics, status
from rest_framework.response import Response

from pred_models.serializers import SplitSerializer
from src.split.models import Split
from src.split.serializers import CreateSplitSerializer


class SplitList(mixins.ListModelMixin, generics.GenericAPIView):
    queryset = Split.objects.all()
    serializer_class = SplitSerializer

    def get(self, request, *args, **kwargs):
        return self.list(request, *args, **kwargs)

    @staticmethod
    def post(request):
        serializer = CreateSplitSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=400)

        # Other serializer for data
        item = serializer.save()
        result = SplitSerializer(item)
        return Response(result.data, status=status.HTTP_201_CREATED)
