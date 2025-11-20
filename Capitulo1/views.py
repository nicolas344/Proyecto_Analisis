from django.shortcuts import render
from .methods import biseccion, regla_falsa, punto_fijo, newton_raphson, secante, raices_multiples, ejecutar_todos
from .graph import graficar_funcion
from django.http import JsonResponse
import json
import sympy as sp

def capitulo1_index(request):
    """Vista principal del Capítulo 1"""
    return render(request, 'capitulo1_index.html')

def biseccion_view(request):
    """Vista para el método de bisección"""
    context = {
        'metodo': 'Método de Bisección',
        'descripcion': 'Método que divide repetidamente un intervalo por la mitad para encontrar la raíz.'
    }
    
    if request.method == 'POST':
        funcion = request.POST.get('funcion')
        xi = float(request.POST.get('xi'))
        xs = float(request.POST.get('xs'))
        tol = float(request.POST.get('tol'))
        niter = int(request.POST.get('niter'))

        tipo_error = request.POST.get('tipo_error', 'absoluto')

        tabla, resultado, mensaje = biseccion(funcion, xi, xs, tol, niter, tipo_error)
        grafico = graficar_funcion(funcion, float(xi), float(xs), resultado) if resultado else None

        comparar = request.POST.get('comparar') == 'on'

        if comparar:
            resultados_comparativos = ejecutar_todos(f_str=funcion, g_str=None, xi=xi, xs=xs, tol=tol, niter=niter, x1=None, tipo_error=tipo_error)
        else:
            resultados_comparativos = None

        context.update({
            'tabla': tabla,
            'resultado': resultado,
            'mensaje': mensaje,
            'funcion': funcion,
            'xi': xi,
            'xs': xs,
            'tol': tol,
            'niter': niter,
            'grafico': grafico,
            'tipo_error': tipo_error,
            'resultados_comparativos': resultados_comparativos
        })

    return render(request, 'biseccion.html', context)

def regla_falsa_view(request):
    """Vista para el método de regla falsa"""
    context = {
        'metodo': 'Método de Regla Falsa',
        'descripcion': 'Mejora del método de bisección que usa interpolación lineal.'
    }
    
    if request.method == 'POST':
        funcion = request.POST.get('funcion')
        xi = float(request.POST.get('xi'))
        xs = float(request.POST.get('xs'))
        tol = float(request.POST.get('tol'))
        niter = int(request.POST.get('niter'))

        tipo_error = request.POST.get('tipo_error', 'absoluto')

        tabla, resultado, mensaje = regla_falsa(funcion, xi, xs, tol, niter, tipo_error)
        grafico = graficar_funcion(funcion, float(xi), float(xs), resultado) if resultado else None

        comparar = request.POST.get('comparar') == 'on'

        if comparar:
            resultados_comparativos = ejecutar_todos(f_str=funcion, g_str=None, xi=xi, xs=xs, tol=tol, niter=niter, x1=None, tipo_error=tipo_error)
        else:
            resultados_comparativos = None

        context.update({
            'tabla': tabla,
            'resultado': resultado,
            'mensaje': mensaje,
            'funcion': funcion,
            'xi': xi,
            'xs': xs,
            'tol': tol,
            'niter': niter,
            'grafico': grafico,
            'tipo_error': tipo_error,
            'resultados_comparativos': resultados_comparativos
        })

    return render(request, 'reglafalsa.html', context)

def punto_fijo_view(request):
    """Vista para el método de punto fijo"""
    context = {
        'metodo': 'Método de Punto Fijo',
        'descripcion': 'Transforma f(x) = 0 en x = g(x) y encuentra el punto donde x es igual a g(x).'
    }
    
    if request.method == 'POST':
        f_str = request.POST['f']
        g_str = request.POST['g']
        x0 = float(request.POST.get('x0'))
        tol = float(request.POST.get('tol'))
        niter = int(request.POST.get('niter'))

        tipo_error = request.POST.get('tipo_error', 'absoluto')

        resultado, tabla, mensaje = punto_fijo(x0, tol, niter, f_str, g_str, tipo_error)
        grafico = graficar_funcion(f_str, x0 - 2, resultado + 5, resultado) if resultado else None

        comparar = request.POST.get('comparar') == 'on'

        if comparar:
            resultados_comparativos = ejecutar_todos(f_str=f_str, g_str=g_str, xi=x0, xs=None, tol=tol, niter=niter, x1=None, tipo_error=tipo_error)
        else:
            resultados_comparativos = None

        context.update({
            'funcion': f_str,
            'g': g_str,
            'x0': x0,
            'tol': tol,
            'niter': niter,
            'resultado': resultado,
            'tabla': tabla,
            'mensaje': mensaje,
            'grafico': grafico,
            'tipo_error': tipo_error,
            'resultados_comparativos': resultados_comparativos
        })

    return render(request, 'puntofijo.html', context)

def newton_view(request):
    """Vista para el método de Newton-Raphson"""
    context = {
        'metodo': 'Método de Newton-Raphson',
        'descripcion': 'Usa la derivada de la función para convergir rápidamente hacia la raíz.'
    }
    
    if request.method == 'POST':
        f_str = request.POST.get('f')
        x0 = float(request.POST.get('x0'))
        tol = float(request.POST.get('tol'))
        niter = int(request.POST.get('niter'))

        tipo_error = request.POST.get('tipo_error', 'absoluto')

        resultado, tabla, mensaje = newton_raphson(x0, tol, niter, f_str, tipo_error)
        grafico = graficar_funcion(f_str, x0 - 2, resultado + 5, resultado) if resultado else None

        comparar = request.POST.get('comparar') == 'on'

        if comparar:
            resultados_comparativos = ejecutar_todos(f_str=f_str, g_str=None, xi=x0, xs=None, tol=tol, niter=niter, x1=None, tipo_error=tipo_error)
        else:
            resultados_comparativos = None

        context.update({
            'funcion': f_str,
            'x0': x0,
            'tol': tol,
            'niter': niter,
            'resultado': resultado,
            'tabla': tabla,
            'mensaje': mensaje,
            'grafico': grafico,
            'tipo_error': tipo_error,
            'resultados_comparativos': resultados_comparativos
        })

    return render(request, 'newton.html', context)

def secante_view(request):
    """Vista para el método de la secante"""
    context = {
        'metodo': 'Método de la Secante',
        'descripcion': 'Aproxima la derivada usando dos puntos anteriores.'
    }
    
    if request.method == 'POST':
        f_str = request.POST['f']
        x0 = float(request.POST['x0'])
        x1 = float(request.POST['x1'])
        tol = float(request.POST['tol'])
        niter = int(request.POST['niter'])

        tipo_error = request.POST.get('tipo_error', 'absoluto')

        resultado, tabla, mensaje = secante(x0, x1, tol, niter, f_str, tipo_error)
        grafico = graficar_funcion(f_str, x0 - 5, x1 + 5, resultado) if resultado else None

        comparar = request.POST.get('comparar') == 'on'

        if comparar:
            resultados_comparativos = ejecutar_todos(f_str=f_str, g_str=None, xi=x0, xs=None, tol=tol, niter=niter, x1=x1, tipo_error=tipo_error)
        else:
            resultados_comparativos = None

        context.update({
            'funcion': f_str,
            'x0': x0,
            'x1': x1,
            'tol': tol,
            'niter': niter,
            'resultado': resultado,
            'tabla': tabla,
            'mensaje': mensaje,
            'grafico': grafico,
            'tipo_error': tipo_error,
            'resultados_comparativos': resultados_comparativos
        })

    return render(request, 'secante.html', context)

def raices_multiples_view(request):
    """Vista para el método de raíces múltiples"""
    context = {
        'metodo': 'Método de Raíces Múltiples',
        'descripcion': 'Método modificado de Newton-Raphson para manejar raíces con multiplicidad mayor a 1.'
    }
    
    if request.method == 'POST':
        f_str = request.POST['f']
        x0 = float(request.POST['x0'])
        tol = float(request.POST['tol'])
        niter = int(request.POST['niter'])

        tipo_error = request.POST.get('tipo_error', 'absoluto')
        
        resultado, tabla, mensaje = raices_multiples(x0, tol, niter, f_str, tipo_error)
        grafico = graficar_funcion(f_str, x0 - 5, x0 + 5, resultado) if resultado else None

        comparar = request.POST.get('comparar') == 'on'

        if comparar:
            resultados_comparativos = ejecutar_todos(f_str=f_str, g_str=None, xi=x0, xs=None, tol=tol, niter=niter, x1=None, tipo_error=tipo_error)
        else:
            resultados_comparativos = None

        context.update({
            'f': f_str,
            'x0': x0,
            'tol': tol,
            'niter': niter,
            'resultado': resultado,
            'tabla': tabla,
            'mensaje': mensaje,
            'grafico': grafico,
            'tipo_error': tipo_error,
            'resultados_comparativos': resultados_comparativos
        })

    return render(request, 'raicesmultiples.html', context)

