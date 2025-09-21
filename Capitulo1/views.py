from django.shortcuts import render
from django.http import JsonResponse
import json

def capitulo1_index(request):
    """Vista principal del Capítulo 1"""
    return render(request, 'capitulo1_index.html')

def biseccion(request):
    """Vista para el método de bisección"""
    context = {
        'metodo': 'Método de Bisección',
        'descripcion': 'Método que divide repetidamente un intervalo por la mitad para encontrar la raíz.'
    }
    return render(request, 'biseccion.html', context)

def reglafalsa(request):
    """Vista para el método de regla falsa"""
    context = {
        'metodo': 'Método de Regla Falsa',
        'descripcion': 'Mejora del método de bisección que usa interpolación lineal.'
    }
    return render(request, 'reglafalsa.html', context)

def puntofijo(request):
    """Vista para el método de punto fijo"""
    context = {
        'metodo': 'Método de Punto Fijo',
        'descripcion': 'Transforma f(x) = 0 en x = g(x) y encuentra el punto donde x es igual a g(x).'
    }
    return render(request, 'puntofijo.html', context)

def newton(request):
    """Vista para el método de Newton-Raphson"""
    context = {
        'metodo': 'Método de Newton-Raphson',
        'descripcion': 'Usa la derivada de la función para convergir rápidamente hacia la raíz.'
    }
    return render(request, 'newton.html', context)

def secante(request):
    """Vista para el método de la secante"""
    context = {
        'metodo': 'Método de la Secante',
        'descripcion': 'Aproxima la derivada usando dos puntos anteriores.'
    }
    return render(request, 'secante.html', context)

def raicesmultiples(request):
    """Vista para el método de raíces múltiples"""
    context = {
        'metodo': 'Método de Raíces Múltiples',
        'descripcion': 'Método modificado de Newton-Raphson para manejar raíces con multiplicidad mayor a 1.'
    }
    return render(request, 'raicesmultiples.html', context)
