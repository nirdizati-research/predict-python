from rest_framework import mixins, generics, status
from rest_framework.response import Response

from src.split.models import Split
from src.split.serializers import CreateSplitSerializer, SplitSerializer


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

        try:
            if Split.objects.filter(
                original_log=serializer.validated_data['original_log'],
                test_size=serializer.validated_data['test_size'],
                splitting_method=serializer.validated_data['splitting_method']
            ).exists():
                return Response(Split.objects.filter(
                    original_log=serializer.validated_data['original_log'],
                    test_size=serializer.validated_data['test_size'],
                    splitting_method=serializer.validated_data['splitting_method']
                )[0].to_dict(), status=status.HTTP_200_OK)  # TODO: Use better code, like 304
            else:
                # Other serializer for data
                split = serializer.save()
        except:
            # Other serializer for data
            split = serializer.save()

        result = SplitSerializer(split)
        return Response(result.data, status=status.HTTP_201_CREATED)
