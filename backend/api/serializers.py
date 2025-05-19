# serializers.py
from rest_framework import serializers
from .mongo_models import TemplateSession, Symbol, WordSample
from django.contrib.auth.models import User

class SymbolSerializer(serializers.Serializer):
    label = serializers.CharField()
    predicted = serializers.CharField(allow_null=True, required=False)
    image = serializers.ImageField()
    x = serializers.IntegerField()
    y = serializers.IntegerField()
    width = serializers.IntegerField()
    height = serializers.IntegerField()
    is_corrected = serializers.BooleanField(default=False)

class WordSampleSerializer(serializers.Serializer):
    text = serializers.CharField()
    predicted = serializers.CharField(allow_null=True, required=False)
    image = serializers.ImageField()
    x = serializers.IntegerField()
    y = serializers.IntegerField()
    width = serializers.IntegerField()
    height = serializers.IntegerField()
    is_corrected = serializers.BooleanField(default=False)

from django.core.files.storage import default_storage

class TemplateSessionSerializer(serializers.Serializer):
    id = serializers.CharField(read_only=True)
    template_type = serializers.CharField(required=False, allow_blank=True)
    image = serializers.ImageField()
    symbols = SymbolSerializer(many=True, read_only=True)
    words = WordSampleSerializer(many=True, read_only=True)

    def create(self, validated_data):
        uploaded_image = validated_data.pop("image")
        image_path = default_storage.save(f"uploads/{uploaded_image.name}", uploaded_image)

        validated_data["image_path"] = image_path
        validated_data["template_type"] = validated_data.get("template_type", "letters")
        validated_data["user_id"] = "demo"  # временно, пока нет авторизации

        return TemplateSession(**validated_data).save()

