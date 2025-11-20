import numpy as np
from django.shortcuts import render
from django.contrib import messages
from .methods import vandermonde_interpolation, newton_interpolation, lagrange_interpolation, spline_interpolation, comparar_metodos_interpolacion, comparar_errores_metodo
from .graph import graficar

# Create your views here.
def optionsch3(request):
    return render(request, 'capitulo3_index.html')

def vandermonde_view(request):
    if request.method == 'POST':
        try:
            num_puntos = int(request.POST.get('num_puntos', 0))
            
            # Validar rango de puntos
            if num_puntos < 2 or num_puntos > 8:
                messages.error(request, "El número de puntos debe estar entre 2 y 8.")
                return render(request, 'vandermonde.html', {'num_puntos': 2})
            
            x_vals = []
            y_vals = []
            for i in range(num_puntos):
                x_i = request.POST.get(f'x_{i}')
                y_i = request.POST.get(f'y_{i}')
                if x_i is None or y_i is None or x_i.strip() == '' or y_i.strip() == '':
                    messages.error(request, "Todos los puntos deben estar completos.")
                    return render(request, 'vandermonde.html')
                x_vals.append(float(x_i))
                y_vals.append(float(y_i))

            x = np.array(x_vals)
            y = np.array(y_vals)

            if len(x) != len(y):
                messages.error(request, "Los vectores x e y deben tener la misma cantidad de datos.")
                return render(request, 'vandermonde.html', {'x_vals': x_vals, 'y_vals': y_vals, 'num_puntos': num_puntos})

            if len(x) < 2:
                messages.error(request, "Debes ingresar al menos 2 puntos.")
                return render(request, 'vandermonde.html', {'x_vals': x_vals, 'y_vals': y_vals, 'num_puntos': num_puntos})

            a, poly_str, xpol, p = vandermonde_interpolation(x, y)
            image_base64 = graficar(x, y, xpol, p)

            comparar = request.POST.get('comparar') == 'on'

            if comparar:
                resultados_comparacion = comparar_metodos_interpolacion(x, y)
            else:
                resultados_comparacion = None
            
            if request.POST.get('comparar_errores') == 'on':
                resultados_errores = comparar_errores_metodo('vandermonde', x, y)
            else:
                resultados_errores = None

            return render(request, 'vandermonde.html', {
                'poly_str': poly_str,
                'image_base64': image_base64,
                'x_vals': x_vals,
                'y_vals': y_vals,
                'num_puntos': num_puntos,
                'resultados_comparacion': resultados_comparacion,
                'resultados_errores': resultados_errores
            })

        except ValueError:
            messages.error(request, "Por favor, ingresa solo valores numéricos.")
        except Exception as e:
            messages.error(request, f"Ocurrió un error: {str(e)}")

        return render(request, 'vandermonde.html')

    else:
        # GET request: puedes enviar un valor por defecto para num_puntos
        return render(request, 'vandermonde.html', {'num_puntos': 2})


def newton_int_view(request):
    if request.method == 'POST':
        try:
            num_puntos = int(request.POST.get('num_puntos', 0))
            
            # Validar rango de puntos
            if num_puntos < 2 or num_puntos > 8:
                messages.error(request, "El número de puntos debe estar entre 2 y 8.")
                return render(request, 'newton_int.html', {'num_puntos': 2})
            
            x_vals = []
            y_vals = []
            for i in range(num_puntos):
                x_i = request.POST.get(f'x_{i}')
                y_i = request.POST.get(f'y_{i}')
                if x_i is None or y_i is None or x_i.strip() == '' or y_i.strip() == '':
                    messages.error(request, "Todos los puntos deben estar completos.")
                    return render(request, 'newton_int.html')
                x_vals.append(float(x_i))
                y_vals.append(float(y_i))

            x = np.array(x_vals)
            y = np.array(y_vals)

            if len(x) != len(y):
                messages.error(request, "Los vectores x e y deben tener la misma cantidad de datos.")
                return render(request, 'newton_int.html', {'x_vals': x_vals, 'y_vals': y_vals, 'num_puntos': num_puntos})

            if len(x) < 2:
                messages.error(request, "Debes ingresar al menos 2 puntos.")
                return render(request, 'newton_int.html', {'x_vals': x_vals, 'y_vals': y_vals, 'num_puntos': num_puntos})

            a, poly_str, xpol, p = newton_interpolation(x, y)
            image_base64 = graficar(x, y, xpol, p)

            comparar = request.POST.get('comparar') == 'on'

            if comparar:
                resultados_comparacion = comparar_metodos_interpolacion(x, y)
            else:
                resultados_comparacion = None
            
            if request.POST.get('comparar_errores') == 'on':
                resultados_errores = comparar_errores_metodo('newton', x, y)
            else:
                resultados_errores = None

            return render(request, 'newton_int.html', {
                'poly_str': poly_str,
                'image_base64': image_base64,
                'x_vals': x_vals,
                'y_vals': y_vals,
                'num_puntos': num_puntos,
                'resultados_comparacion': resultados_comparacion,
                'resultados_errores': resultados_errores
            })

        except ValueError:
            messages.error(request, "Por favor, ingresa solo valores numéricos.")
        except Exception as e:
            messages.error(request, f"Ocurrió un error: {str(e)}")

        return render(request, 'newton_int.html')

    else:
        # GET request: puedes enviar un valor por defecto para num_puntos
        return render(request, 'newton_int.html', {'num_puntos': 2})

def lagrange_view(request):
    if request.method == 'POST':
        try:
            num_puntos = int(request.POST.get('num_puntos', 0))
            
            # Validar rango de puntos
            if num_puntos < 2 or num_puntos > 8:
                messages.error(request, "El número de puntos debe estar entre 2 y 8.")
                return render(request, 'lagrange.html', {'num_puntos': 2})
            
            x_vals = []
            y_vals = []
            for i in range(num_puntos):
                x_i = request.POST.get(f'x_{i}')
                y_i = request.POST.get(f'y_{i}')
                if x_i is None or y_i is None or x_i.strip() == '' or y_i.strip() == '':
                    messages.error(request, "Todos los puntos deben estar completos.")
                    return render(request, 'lagrange.html')
                x_vals.append(float(x_i))
                y_vals.append(float(y_i))

            x = np.array(x_vals)
            y = np.array(y_vals)

            if len(x) != len(y):
                messages.error(request, "Los vectores x e y deben tener la misma cantidad de datos.")
                return render(request, 'lagrange.html', {'x_vals': x_vals, 'y_vals': y_vals, 'num_puntos': num_puntos})

            if len(x) < 2:
                messages.error(request, "Debes ingresar al menos 2 puntos.")
                return render(request, 'lagrange.html', {'x_vals': x_vals, 'y_vals': y_vals, 'num_puntos': num_puntos})

            poly_str, xpol, p = lagrange_interpolation(x, y)
            image_base64 = graficar(x, y, xpol, p)

            comparar = request.POST.get('comparar') == 'on'

            if comparar:
                resultados_comparacion = comparar_metodos_interpolacion(x, y)
            else:
                resultados_comparacion = None
            
            if request.POST.get('comparar_errores') == 'on':
                resultados_errores = comparar_errores_metodo('lagrange', x, y)
            else:
                resultados_errores = None

            return render(request, 'lagrange.html', {
                'poly_str': poly_str,
                'image_base64': image_base64,
                'x_vals': x_vals,
                'y_vals': y_vals,
                'num_puntos': num_puntos,
                'resultados_comparacion': resultados_comparacion,
                'resultados_errores': resultados_errores
            })

        except ValueError:
            messages.error(request, "Por favor, ingresa solo valores numéricos.")
        except Exception as e:
            messages.error(request, f"Ocurrió un error: {str(e)}")

        return render(request, 'lagrange.html')

    else:
        return render(request, 'lagrange.html', {'num_puntos': 2})

def spline_view(request):
    if request.method == 'POST':
        try:
            num_puntos = int(request.POST.get('num_puntos', 0))
            
            # Validar rango de puntos
            if num_puntos < 2 or num_puntos > 8:
                messages.error(request, "El número de puntos debe estar entre 2 y 8.")
                return render(request, 'spline.html', {'num_puntos': 2})
            
            x_vals = []
            y_vals = []
            for i in range(num_puntos):
                x_i = request.POST.get(f'x_{i}')
                y_i = request.POST.get(f'y_{i}')
                if x_i is None or y_i is None or x_i.strip() == '' or y_i.strip() == '':
                    messages.error(request, "Todos los puntos deben estar completos.")
                    return render(request, 'spline.html')
                x_vals.append(float(x_i))
                y_vals.append(float(y_i))

            # Extraer tipo de spline lineal o cúbico
            tipo_spline = request.POST.get('tipo_spline')
            if tipo_spline not in ['lineal', 'cubico']:
                messages.error(request, "Debes seleccionar spline lineal o cúbico.")
                return render(request, 'spline.html', {'x_vals': x_vals, 'y_vals': y_vals, 'num_puntos': num_puntos})

            x = np.array(x_vals)
            y = np.array(y_vals)

            if len(x) != len(y):
                messages.error(request, "Los vectores x e y deben tener la misma cantidad de datos.")
                return render(request, 'spline.html', {'x_vals': x_vals, 'y_vals': y_vals, 'num_puntos': num_puntos})

            if len(x) < 2:
                messages.error(request, "Debes ingresar al menos 2 puntos.")
                return render(request, 'spline.html', {'x_vals': x_vals, 'y_vals': y_vals, 'num_puntos': num_puntos})

            poly_str, xpol, p = spline_interpolation(x, y, tipo_spline)
            image_base64 = graficar(x, y, xpol, p)

            comparar = request.POST.get('comparar') == 'on'

            if comparar:
                resultados_comparacion = comparar_metodos_interpolacion(x, y)
            else:
                resultados_comparacion = None
            
            if request.POST.get('comparar_errores') == 'on':
                metodo_nombre = 'spline_lineal' if tipo_spline == 'lineal' else 'spline_cubico'
                resultados_errores = comparar_errores_metodo(metodo_nombre, x, y, tipo_spline)
            else:
                resultados_errores = None

            return render(request, 'spline.html', {
                'poly_str': poly_str,
                'image_base64': image_base64,
                'x_vals': x_vals,
                'y_vals': y_vals,
                'num_puntos': num_puntos,
                'tipo_spline': tipo_spline,
                'resultados_comparacion': resultados_comparacion,
                'resultados_errores': resultados_errores
            })

        except ValueError:
            messages.error(request, "Por favor, ingresa solo valores numéricos.")
        except Exception as e:
            messages.error(request, f"Ocurrió un error: {str(e)}")

        return render(request, 'spline.html')

    else:
        return render(request, 'spline.html', {'num_puntos': 2})