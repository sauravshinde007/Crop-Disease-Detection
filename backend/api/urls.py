from django.urls import path
from .views import predict
from .views import get_weather
urlpatterns = [
    path('predict/', predict, name='predict'),
    path('weather/', get_weather, name='weather')
]
