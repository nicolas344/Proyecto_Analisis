from django.urls import path
from . import views

app_name = "capitulo3"

urlpatterns = [
    path('options/', views.optionsch3, name='optionsch3'),
    path('vandermonde/', views.vandermonde_view, name='vandermonde'),
    path('newton_int/', views.newton_int_view, name='newton_int'),
    path('lagrange/', views.lagrange_view, name='lagrange'),
    path('spline/', views.spline_view, name='spline'),
]