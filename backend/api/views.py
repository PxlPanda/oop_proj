from django.shortcuts import render

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .serializers import TemplateSessionSerializer
from .mongo_models import TemplateSession

class TemplateUploadView(APIView):
    def post(self, request):
        serializer = TemplateSessionSerializer()
        if serializer.is_valid():
            session = serializer.save()
            return Response({"id": str(session.id)}, status=status.HTTP_201_CREATED)
        return Response({"id": str(session.id)}, status=status.HTTP_400_BAD_REQUEST)