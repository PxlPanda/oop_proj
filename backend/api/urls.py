from django.urls import path
from . import views
from .views import TemplateUploadView, get_symbols

urlpatterns = [
    path('upload-template/', TemplateUploadView.as_view(), name = 'upload-template'),
    path('session/<str:session_id>/symbols/', get_symbols),
]

