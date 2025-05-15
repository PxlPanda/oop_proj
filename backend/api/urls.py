from django.urls import path
from . import views
from .views import TemplateUploadView

urlpatterns = [
    path('upload-template/', TemplateUploadView.as_view(), name = 'upload-template'),
]

