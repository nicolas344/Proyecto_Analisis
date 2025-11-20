from django.shortcuts import render
from .methods import biseccion, regla_falsa, punto_fijo, newton_raphson, secante, raices_multiples, ejecutar_todos, comparar_errores_metodo
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
        try:
            funcion = request.POST.get('funcion', '').strip()
            
            # Validar función no vacía
            if not funcion:
                context['mensaje'] = "Error: Debe proporcionar una función f(x)"
                return render(request, 'biseccion.html', context)
            
            # Validar parámetros numéricos
            try:
                xi = float(request.POST.get('xi'))
                xs = float(request.POST.get('xs'))
                tol = float(request.POST.get('tol'))
                niter = int(request.POST.get('niter'))
            except (ValueError, TypeError):
                context['mensaje'] = "Error: Los valores numéricos deben ser válidos"
                return render(request, 'biseccion.html', context)

            tipo_error = request.POST.get('tipo_error', 'absoluto')

            tabla, resultado, mensaje = biseccion(funcion, xi, xs, tol, niter, tipo_error)
            grafico = graficar_funcion(funcion, float(xi), float(xs), resultado) if resultado else None

            comparar = request.POST.get('comparar') == 'on'
            comparar_errores = request.POST.get('comparar_errores') == 'on'

            if comparar:
                resultados_comparativos = ejecutar_todos(f_str=funcion, g_str=None, xi=xi, xs=xs, tol=tol, niter=niter, x1=None, tipo_error=tipo_error)
            else:
                resultados_comparativos = None
            
            if comparar_errores:
                resultados_errores = comparar_errores_metodo('biseccion', funcion, None, xi, xs, tol, niter, None)
            else:
                resultados_errores = None

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
                'resultados_comparativos': resultados_comparativos,
                'resultados_errores': resultados_errores
            })
        
        except Exception as e:
            context['mensaje'] = f"Error inesperado: {str(e)}"

    return render(request, 'biseccion.html', context)

def regla_falsa_view(request):
    """Vista para el método de regla falsa"""
    context = {
        'metodo': 'Método de Regla Falsa',
        'descripcion': 'Mejora del método de bisección que usa interpolación lineal.'
    }
    
    if request.method == 'POST':
        try:
            funcion = request.POST.get('funcion', '').strip()
            
            if not funcion:
                context['mensaje'] = "Error: Debe proporcionar una función f(x)"
                return render(request, 'reglafalsa.html', context)
            
            try:
                xi = float(request.POST.get('xi'))
                xs = float(request.POST.get('xs'))
                tol = float(request.POST.get('tol'))
                niter = int(request.POST.get('niter'))
            except (ValueError, TypeError):
                context['mensaje'] = "Error: Los valores numéricos deben ser válidos"
                return render(request, 'reglafalsa.html', context)

            tipo_error = request.POST.get('tipo_error', 'absoluto')

            tabla, resultado, mensaje = regla_falsa(funcion, xi, xs, tol, niter, tipo_error)
            grafico = graficar_funcion(funcion, float(xi), float(xs), resultado) if resultado else None

            comparar = request.POST.get('comparar') == 'on'
            comparar_errores = request.POST.get('comparar_errores') == 'on'

            if comparar:
                resultados_comparativos = ejecutar_todos(f_str=funcion, g_str=None, xi=xi, xs=xs, tol=tol, niter=niter, x1=None, tipo_error=tipo_error)
            else:
                resultados_comparativos = None
            
            if comparar_errores:
                resultados_errores = comparar_errores_metodo('regla_falsa', funcion, None, xi, xs, tol, niter, None)
            else:
                resultados_errores = None

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
                'resultados_comparativos': resultados_comparativos,
                'resultados_errores': resultados_errores
            })
        
        except Exception as e:
            context['mensaje'] = f"Error inesperado: {str(e)}"

    return render(request, 'reglafalsa.html', context)

def punto_fijo_view(request):
    """Vista para el método de punto fijo"""
    context = {
        'metodo': 'Método de Punto Fijo',
        'descripcion': 'Transforma f(x) = 0 en x = g(x) y encuentra el punto donde x es igual a g(x).'
    }
    
    if request.method == 'POST':
        try:
            f_str = request.POST['f']
            g_str = request.POST.get('g', '').strip()  # Obtener g_str y eliminar espacios
            
            # Validar que f_str no esté vacío
            if not f_str or f_str.strip() == '':
                context['mensaje'] = "Error: Debe proporcionar una función f(x)"
                return render(request, 'puntofijo.html', context)
            
            # Validar que los parámetros numéricos sean válidos
            try:
                x0 = float(request.POST.get('x0'))
                tol = float(request.POST.get('tol'))
                niter = int(request.POST.get('niter'))
            except (ValueError, TypeError):
                context['mensaje'] = "Error: Los valores de x0, tolerancia e iteraciones deben ser numéricos"
                return render(request, 'puntofijo.html', context)

            # Si g_str está vacío, dejarlo como vacío para cálculo automático
            if not g_str:
                g_str = ''

            tipo_error = request.POST.get('tipo_error', 'absoluto')

            resultado, tabla, mensaje = punto_fijo(x0, tol, niter, f_str, g_str, tipo_error)
            grafico = graficar_funcion(f_str, x0 - 2, resultado + 5, resultado) if resultado else None

            comparar = request.POST.get('comparar') == 'on'
            comparar_errores = request.POST.get('comparar_errores') == 'on'

            if comparar:
                resultados_comparativos = ejecutar_todos(f_str=f_str, g_str=g_str if g_str else None, xi=x0, xs=None, tol=tol, niter=niter, x1=None, tipo_error=tipo_error)
            else:
                resultados_comparativos = None
            
            if comparar_errores:
                resultados_errores = comparar_errores_metodo('punto_fijo', f_str, g_str if g_str else None, x0, None, tol, niter, None)
            else:
                resultados_errores = None

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
                'resultados_comparativos': resultados_comparativos,
                'resultados_errores': resultados_errores
            })
        
        except Exception as e:
            context['mensaje'] = f"Error inesperado: {str(e)}"

    return render(request, 'puntofijo.html', context)

def newton_view(request):
    """Vista para el método de Newton-Raphson"""
    context = {
        'metodo': 'Método de Newton-Raphson',
        'descripcion': 'Usa la derivada de la función para convergir rápidamente hacia la raíz.'
    }
    
    if request.method == 'POST':
        try:
            f_str = request.POST.get('f', '').strip()
            
            if not f_str:
                context['mensaje'] = "Error: Debe proporcionar una función f(x)"
                return render(request, 'newton.html', context)
            
            try:
                x0 = float(request.POST.get('x0'))
                tol = float(request.POST.get('tol'))
                niter = int(request.POST.get('niter'))
            except (ValueError, TypeError):
                context['mensaje'] = "Error: Los valores numéricos deben ser válidos"
                return render(request, 'newton.html', context)

            tipo_error = request.POST.get('tipo_error', 'absoluto')

            resultado, tabla, mensaje = newton_raphson(x0, tol, niter, f_str, tipo_error)
            grafico = graficar_funcion(f_str, x0 - 2, resultado + 5, resultado) if resultado else None

            comparar = request.POST.get('comparar') == 'on'
            comparar_errores = request.POST.get('comparar_errores') == 'on'

            if comparar:
                resultados_comparativos = ejecutar_todos(f_str=f_str, g_str=None, xi=x0, xs=None, tol=tol, niter=niter, x1=None, tipo_error=tipo_error)
            else:
                resultados_comparativos = None
            
            if comparar_errores:
                resultados_errores = comparar_errores_metodo('newton', f_str, None, x0, None, tol, niter, None)
            else:
                resultados_errores = None

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
                'resultados_comparativos': resultados_comparativos,
                'resultados_errores': resultados_errores
            })
        
        except Exception as e:
            context['mensaje'] = f"Error inesperado: {str(e)}"

    return render(request, 'newton.html', context)

def secante_view(request):
    """Vista para el método de la secante"""
    context = {
        'metodo': 'Método de la Secante',
        'descripcion': 'Aproxima la derivada usando dos puntos anteriores.'
    }
    
    if request.method == 'POST':
        try:
            f_str = request.POST.get('f', '').strip()
            
            if not f_str:
                context['mensaje'] = "Error: Debe proporcionar una función f(x)"
                return render(request, 'secante.html', context)
            
            try:
                x0 = float(request.POST['x0'])
                x1 = float(request.POST['x1'])
                tol = float(request.POST['tol'])
                niter = int(request.POST['niter'])
            except (ValueError, TypeError):
                context['mensaje'] = "Error: Los valores numéricos deben ser válidos"
                return render(request, 'secante.html', context)

            tipo_error = request.POST.get('tipo_error', 'absoluto')

            resultado, tabla, mensaje = secante(x0, x1, tol, niter, f_str, tipo_error)
            grafico = graficar_funcion(f_str, x0 - 5, x1 + 5, resultado) if resultado else None

            comparar = request.POST.get('comparar') == 'on'
            comparar_errores = request.POST.get('comparar_errores') == 'on'

            if comparar:
                resultados_comparativos = ejecutar_todos(f_str=f_str, g_str=None, xi=x0, xs=None, tol=tol, niter=niter, x1=x1, tipo_error=tipo_error)
            else:
                resultados_comparativos = None
            
            if comparar_errores:
                resultados_errores = comparar_errores_metodo('secante', f_str, None, x0, None, tol, niter, x1)
            else:
                resultados_errores = None

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
                'resultados_comparativos': resultados_comparativos,
                'resultados_errores': resultados_errores
            })
        
        except Exception as e:
            context['mensaje'] = f"Error inesperado: {str(e)}"

    return render(request, 'secante.html', context)

def raices_multiples_view(request):
    """Vista para el método de raíces múltiples"""
    context = {
        'metodo': 'Método de Raíces Múltiples',
        'descripcion': 'Método modificado de Newton-Raphson para manejar raíces con multiplicidad mayor a 1.'
    }
    
    if request.method == 'POST':
        try:
            f_str = request.POST.get('f', '').strip()
            
            if not f_str:
                context['mensaje'] = "Error: Debe proporcionar una función f(x)"
                return render(request, 'raicesmultiples.html', context)
            
            try:
                x0 = float(request.POST['x0'])
                tol = float(request.POST['tol'])
                niter = int(request.POST['niter'])
            except (ValueError, TypeError):
                context['mensaje'] = "Error: Los valores numéricos deben ser válidos"
                return render(request, 'raicesmultiples.html', context)

            tipo_error = request.POST.get('tipo_error', 'absoluto')
            
            resultado, tabla, mensaje = raices_multiples(x0, tol, niter, f_str, tipo_error)
            grafico = graficar_funcion(f_str, x0 - 5, x0 + 5, resultado) if resultado else None

            comparar = request.POST.get('comparar') == 'on'
            comparar_errores = request.POST.get('comparar_errores') == 'on'

            if comparar:
                resultados_comparativos = ejecutar_todos(f_str=f_str, g_str=None, xi=x0, xs=None, tol=tol, niter=niter, x1=None, tipo_error=tipo_error)
            else:
                resultados_comparativos = None
            
            if comparar_errores:
                resultados_errores = comparar_errores_metodo('raices_multiples', f_str, None, x0, None, tol, niter, None)
            else:
                resultados_errores = None

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
                'resultados_comparativos': resultados_comparativos,
                'resultados_errores': resultados_errores
            })
        
        except Exception as e:
            context['mensaje'] = f"Error inesperado: {str(e)}"

    return render(request, 'raicesmultiples.html', context)

