# views.py
from django.shortcuts import render

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .serializers import TemplateSessionSerializer
from .mongo_models import TemplateSession
from rest_framework.parsers import MultiPartParser, FormParser

class TemplateUploadView(APIView):
    parser_classes = [MultiPartParser, FormParser]

    def post(self, request):
        serializer = TemplateSessionSerializer(data=request.data)
        if serializer.is_valid():
            session = serializer.save()
            return Response({"sessionId": str(session.id)}, status=status.HTTP_201_CREATED)
        else:
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
