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

class TemplateSessionSerializer(serializers.Serializer):
    id = serializers.CharField(read_only=True)
    template_type = serializers.CharField()
    image = serializers.ImageField()
    user = serializers.PrimaryKeyRelatedField(queryset=User.objects.all())
    symbols = SymbolSerializer(many=True, read_only=True)
    words = WordSampleSerializer(many=True, read_only=True)

    def create(self, validated_data):
        return TemplateSession(**validated_data).save()
