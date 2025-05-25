from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.decorators import api_view

from .serializers import TemplateSessionSerializer
from .mongo_models import TemplateSession, Symbol

from recognition.inference import recognize_template

class TemplateUploadView(APIView):
    parser_classes = [MultiPartParser, FormParser]

    def post(self, request):
        print("🔍 DEBUG: request.data =", request.data)
        print("🔍 DEBUG: request.FILES =", request.FILES)

        serializer = TemplateSessionSerializer(data=request.data)
        if serializer.is_valid():
            session = serializer.save()

            # распознавание
            try:
                recognize_template(session)
            except Exception as e:
                print("❌ Ошибка при вызове recognize_template:", e)

            return Response({"sessionId": str(session.id)}, status=status.HTTP_201_CREATED)
        else:
            print("❌ serializer.errors =", serializer.errors)
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

@api_view(['GET'])
def get_symbols(request, session_id):
    try:
        session = TemplateSession.objects.get(id=session_id)
    except TemplateSession.DoesNotExist:
        return Response({'error': 'Сессия не найдена'}, status=404)

    symbols = Symbol.objects.filter(session=session)
    symbol_data = [
        {
            "id": str(sym.id),
            "x": sym.x,
            "y": sym.y,
            "width": sym.width,
            "height": sym.height,
            "char": sym.label,  # если label = распознанный символ
        }
        for sym in symbols
    ]

    return Response({
        "image_url": "/media/" + session.image_path,  # путь к загруженному шаблону
        "symbols": symbol_data
    })
