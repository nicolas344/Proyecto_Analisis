from django.urls import path
from . import views

app_name = "capitulo2"

urlpatterns = [
    path('options/', views.optionsch2, name ='optionsch2'),
    path('jacobi/', views.jacobi_view, name='jacobi'),
    path('gauss_seidel/', views.gauss_seidel_view, name='gauss_seidel'),
    path('sor/', views.sor_view, name='sor'),
]