# filepath: c:\Users\ricom\OneDrive\Desktop\Proyecto_Analisis\Capitulo1\urls.py
from django.urls import path
from . import views

app_name = 'capitulo1'

urlpatterns = [
    path('', views.capitulo1_index, name='index'),
    path('biseccion/', views.biseccion, name='biseccion'),
    path('reglafalsa/', views.reglafalsa, name='reglafalsa'),
    path('puntofijo/', views.puntofijo, name='puntofijo'),
    path('newton/', views.newton, name='newton'),
    path('secante/', views.secante, name='secante'),
    path('raicesmultiples/', views.raicesmultiples, name='raicesmultiples'),
]