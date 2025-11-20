# filepath: c:\Users\ricom\OneDrive\Desktop\Proyecto_Analisis\Capitulo1\urls.py
from django.urls import path
from . import views

app_name = 'capitulo1'

urlpatterns = [
    path('', views.capitulo1_index, name='index'),
    path('biseccion/', views.biseccion_view, name='biseccion'),
    path('reglafalsa/', views.regla_falsa_view, name='reglafalsa'),
    path('puntofijo/', views.punto_fijo_view, name='puntofijo'),
    path('newton/', views.newton_view, name='newton'),
    path('secante/', views.secante_view, name='secante'),
    path('raicesmultiples/', views.raices_multiples_view, name='raicesmultiples'),
]