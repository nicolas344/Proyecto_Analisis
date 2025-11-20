import numpy as np
from django.shortcuts import render
from .methods import jacobi, gauss_seidel, sor, ejecutar_todos, comparar_errores_metodo


def capitulo2_index(request):
    """Vista principal del Capítulo 2"""
    return render(request, 'capitulo2_index.html')


def calcular_radio_espectral(A, metodo, w=None):
    A = np.array(A, float)

    D = np.diag(np.diag(A))
    L = -np.tril(A, -1)
    U = -np.triu(A, 1)

    if metodo == "jacobi":
        T = np.linalg.inv(D).dot(L + U)
    elif metodo == "gauss":
        T = np.linalg.inv(D - L).dot(U)
    elif metodo == "sor":
        T = np.linalg.inv(D - w * L).dot((1 - w) * D + w * U)
    else:
        return None

    eig = np.linalg.eigvals(T)
    return float(max(abs(eig)))


def optionsch2(request):
    return render(request, 'optionsch2.html')




def jacobi_view(request):

    size = 2
    A = None
    b = None
    x0 = None
    tabla = None
    mensaje = None
    resultados_todos = None
    resultados_errores = None
    rho = None
    converge = None
    error_type = "relative"
    tol = None
    niter = None

    if request.method == 'POST':
        try:
            size = int(request.POST.get('matrix_size', 2))
            
            # Validar tamaño de matriz
            if size < 2 or size > 7:
                mensaje = "Error: El tamaño de la matriz debe estar entre 2 y 7"
                return render(request, 'jacobi.html', {
                    'matrix_size': size, 'range': range(2),
                    'mensaje': mensaje, 'error_type': error_type
                })

            A = [[float(request.POST.get(f'a_{i}_{j}', 0)) for j in range(size)] for i in range(size)]
            b = [float(request.POST.get(f'b_{i}', 0)) for i in range(size)]
            x0 = [float(request.POST.get(f'x0_{i}', 0)) for i in range(size)]

            tol = float(request.POST.get('tol', '1e-7'))
            niter = int(request.POST.get('niter', 100))
            
            # Validar tolerancia y iteraciones
            if tol <= 0:
                mensaje = "Error: La tolerancia debe ser mayor que 0"
                return render(request, 'jacobi.html', {
                    'matrix_size': size, 'range': range(size),
                    'A': A, 'b': b, 'x0': x0, 'mensaje': mensaje,
                    'tol': tol, 'niter': niter, 'error_type': error_type
                })
            
            if niter <= 0:
                mensaje = "Error: El número de iteraciones debe ser mayor que 0"
                return render(request, 'jacobi.html', {
                    'matrix_size': size, 'range': range(size),
                    'A': A, 'b': b, 'x0': x0, 'mensaje': mensaje,
                    'tol': tol, 'niter': niter, 'error_type': error_type
                })

            # === Obtener tipo de error ===
            error_type = request.POST.get("error_type", "relative")

            # Traducir para methods.py
            cs = (error_type == "relative")

            # Ejecutar Jacobi
            tabla, mensaje = jacobi(x0, A, b, tol, niter, cs)

            # =============== Error condicional ===============
            if error_type == "condition":
                tabla_cond = []
                for idx, (k, x, err) in enumerate(tabla):
                    if idx == 0:
                        tabla_cond.append((k, x, None))
                    else:
                        Ax = np.dot(A, x)
                        new_err = float(np.linalg.norm(Ax - b))
                        tabla_cond.append((k, x, new_err))
                tabla = tabla_cond

            # Radio espectral
            rho = calcular_radio_espectral(A, "jacobi")
            converge = (rho < 1)

            # Comparación de métodos
            if request.POST.get('comparar') == 'on':
                resultados_todos = ejecutar_todos(x0, A, b, tol, niter, cs)
            
            # Comparación de errores
            resultados_errores = None
            if request.POST.get('comparar_errores') == 'on':
                resultados_errores = comparar_errores_metodo('jacobi', x0, A, b, tol, niter)

        except ValueError:
            mensaje = "Error: Verifique que todos los valores numéricos sean válidos"
        except Exception as e:
            mensaje = f"Error inesperado: {str(e)}"

    return render(request, 'jacobi.html', {
        'matrix_size': size,
        'range': range(size),
        'A': A, 'b': b, 'x0': x0,
        'tabla': tabla, 'mensaje': mensaje,
        'rho': rho, 'converge': converge,
        'error_type': error_type,
        'tol': tol, 'niter': niter,
        'resultados_todos': resultados_todos,
        'resultados_errores': resultados_errores
    })




def gauss_seidel_view(request):

    size = 2
    A = b = x0 = None
    tabla = mensaje = None
    resultados_todos = None
    resultados_errores = None
    rho = None
    converge = None
    error_type = "relative"
    tol = None
    niter = None

    if request.method == 'POST':
        try:
            size = int(request.POST.get('matrix_size', 2))
            
            # Validar tamaño de matriz
            if size < 2 or size > 7:
                mensaje = "Error: El tamaño de la matriz debe estar entre 2 y 7"
                return render(request, 'gauss_seidel.html', {
                    'matrix_size': size, 'range': range(2),
                    'mensaje': mensaje, 'error_type': error_type
                })

            A = [[float(request.POST.get(f'a_{i}_{j}', 0)) for j in range(size)] for i in range(size)]
            b = [float(request.POST.get(f'b_{i}', 0)) for i in range(size)]
            x0 = [float(request.POST.get(f'x0_{i}', 0)) for i in range(size)]

            tol = float(request.POST.get('tol', '1e-7'))
            niter = int(request.POST.get('niter', 100))
            
            # Validar tolerancia y iteraciones
            if tol <= 0:
                mensaje = "Error: La tolerancia debe ser mayor que 0"
                return render(request, 'gauss_seidel.html', {
                    'matrix_size': size, 'range': range(size),
                    'A': A, 'b': b, 'x0': x0, 'mensaje': mensaje,
                    'tol': tol, 'niter': niter, 'error_type': error_type
                })
            
            if niter <= 0:
                mensaje = "Error: El número de iteraciones debe ser mayor que 0"
                return render(request, 'gauss_seidel.html', {
                    'matrix_size': size, 'range': range(size),
                    'A': A, 'b': b, 'x0': x0, 'mensaje': mensaje,
                    'tol': tol, 'niter': niter, 'error_type': error_type
                })

            error_type = request.POST.get("error_type", "relative")
            cs = (error_type == "relative")

            tabla, mensaje = gauss_seidel(x0, A, b, tol, niter, cs)

            # Condicional
            if error_type == "condition":
                tabla_cond = []
                for idx, (k, x, err) in enumerate(tabla):
                    if idx == 0:
                        tabla_cond.append((k, x, None))
                    else:
                        Ax = np.dot(A, x)
                        new_err = float(np.linalg.norm(Ax - b))
                        tabla_cond.append((k, x, new_err))
                tabla = tabla_cond

            rho = calcular_radio_espectral(A, "gauss")
            converge = (rho < 1)

            if request.POST.get('comparar') == 'on':
                resultados_todos = ejecutar_todos(x0, A, b, tol, niter, cs)
            
            if request.POST.get('comparar_errores') == 'on':
                resultados_errores = comparar_errores_metodo('gauss_seidel', x0, A, b, tol, niter)

        except ValueError:
            mensaje = "Error: Verifique que todos los valores numéricos sean válidos"
        except Exception as e:
            mensaje = f"Error inesperado: {str(e)}"

    return render(request, 'gauss_seidel.html', {
        'matrix_size': size,
        'range': range(size),
        'A': A, 'b': b, 'x0': x0,
        'tabla': tabla, 'mensaje': mensaje,
        'rho': rho, 'converge': converge,
        'error_type': error_type,
        'tol': tol, 'niter': niter,
        'resultados_todos': resultados_todos,
        'resultados_errores': resultados_errores
    })




def sor_view(request):

    size = 2
    A = b = x0 = tabla = mensaje = None
    resultados_todos = None
    resultados_errores = None
    w = None
    rho = None
    converge = None
    error_type = "relative"
    tol = None
    niter = None

    if request.method == 'POST':
        try:
            size = int(request.POST.get('matrix_size', 2))
            
            # Validar tamaño de matriz
            if size < 2 or size > 7:
                mensaje = "Error: El tamaño de la matriz debe estar entre 2 y 7"
                return render(request, 'sor.html', {
                    'matrix_size': size, 'range': range(2),
                    'mensaje': mensaje, 'error_type': error_type
                })

            A = [[float(request.POST.get(f'a_{i}_{j}', 0)) for j in range(size)] for i in range(size)]
            b = [float(request.POST.get(f'b_{i}', 0)) for i in range(size)]
            x0 = [float(request.POST.get(f'x0_{i}', 0)) for i in range(size)]

            tol = float(request.POST.get('tol', '1e-7'))
            niter = int(request.POST.get('niter', 100))
            w = float(request.POST.get('w', 1.0))
            
            # Validar tolerancia y iteraciones
            if tol <= 0:
                mensaje = "Error: La tolerancia debe ser mayor que 0"
                return render(request, 'sor.html', {
                    'matrix_size': size, 'range': range(size),
                    'A': A, 'b': b, 'x0': x0, 'mensaje': mensaje,
                    'tol': tol, 'niter': niter, 'w': w, 'error_type': error_type
                })
            
            if niter <= 0:
                mensaje = "Error: El número de iteraciones debe ser mayor que 0"
                return render(request, 'sor.html', {
                    'matrix_size': size, 'range': range(size),
                    'A': A, 'b': b, 'x0': x0, 'mensaje': mensaje,
                    'tol': tol, 'niter': niter, 'w': w, 'error_type': error_type
                })
            
            # Validar factor de relajación
            if w <= 0 or w >= 2:
                mensaje = "Error: El factor de relajación ω debe estar entre 0 y 2 (no inclusivo)"
                return render(request, 'sor.html', {
                    'matrix_size': size, 'range': range(size),
                    'A': A, 'b': b, 'x0': x0, 'mensaje': mensaje,
                    'tol': tol, 'niter': niter, 'w': w, 'error_type': error_type
                })

            error_type = request.POST.get("error_type", "relative")
            cs = (error_type == "relative")

            tabla, mensaje = sor(x0, A, b, tol, niter, w, cs)

            # Condicional
            if error_type == "condition":
                tabla_cond = []
                for idx, (k, x, err) in enumerate(tabla):
                    if idx == 0:
                        tabla_cond.append((k, x, None))
                    else:
                        Ax = np.dot(A, x)
                        new_err = float(np.linalg.norm(Ax - b))
                        tabla_cond.append((k, x, new_err))
                tabla = tabla_cond

            rho = calcular_radio_espectral(A, "sor", w=w)
            converge = (rho < 1)

            if request.POST.get('comparar') == 'on':
                resultados_todos = ejecutar_todos(x0, A, b, tol, niter, cs)
            
            if request.POST.get('comparar_errores') == 'on':
                resultados_errores = comparar_errores_metodo('sor', x0, A, b, tol, niter, w=w)

        except ValueError:
            mensaje = "Error: Verifique que todos los valores numéricos sean válidos"
        except Exception as e:
            mensaje = f"Error inesperado: {str(e)}"

    return render(request, 'sor.html', {
        'matrix_size': size,
        'range': range(size),
        'A': A, 'b': b, 'x0': x0,
        'tabla': tabla, 'mensaje': mensaje,
        'w': w,
        'rho': rho, 'converge': converge,
        'error_type': error_type,
        'tol': tol, 'niter': niter,
        'resultados_todos': resultados_todos,
        'resultados_errores': resultados_errores
    })