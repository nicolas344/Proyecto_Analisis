import numpy as np
from django.shortcuts import render
from .methods import jacobi, gauss_seidel, sor, ejecutar_todos



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
    rho = None
    converge = None
    error_type = "relative"

    if request.method == 'POST':
        try:
            size = int(request.POST.get('matrix_size', 2))

            A = [[float(request.POST.get(f'a_{i}_{j}', 0)) for j in range(size)] for i in range(size)]
            b = [float(request.POST.get(f'b_{i}', 0)) for i in range(size)]
            x0 = [float(request.POST.get(f'x0_{i}', 0)) for i in range(size)]

            tol = float(request.POST.get('tol', '1e-7'))
            niter = int(request.POST.get('niter', 100))

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

            # Comparación
            if request.POST.get('comparar') == 'on':
                resultados_todos = ejecutar_todos(x0, A, b, tol, niter, cs)

        except Exception as e:
            mensaje = f"Error: {str(e)}"

    return render(request, 'jacobi.html', {
        'matrix_size': size,
        'range': range(size),
        'A': A, 'b': b, 'x0': x0,
        'tabla': tabla, 'mensaje': mensaje,
        'rho': rho, 'converge': converge,
        'error_type': error_type,
        'resultados_todos': resultados_todos
    })




def gauss_seidel_view(request):

    size = 2
    A = b = x0 = None
    tabla = mensaje = None
    resultados_todos = None
    rho = None
    converge = None
    error_type = "relative"

    if request.method == 'POST':
        try:
            size = int(request.POST.get('matrix_size', 2))

            A = [[float(request.POST.get(f'a_{i}_{j}', 0)) for j in range(size)] for i in range(size)]
            b = [float(request.POST.get(f'b_{i}', 0)) for i in range(size)]
            x0 = [float(request.POST.get(f'x0_{i}', 0)) for i in range(size)]

            tol = float(request.POST.get('tol', '1e-7'))
            niter = int(request.POST.get('niter', 100))

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

        except Exception as e:
            mensaje = f"Error: {str(e)}"

    return render(request, 'gauss_seidel.html', {
        'matrix_size': size,
        'range': range(size),
        'A': A, 'b': b, 'x0': x0,
        'tabla': tabla, 'mensaje': mensaje,
        'rho': rho, 'converge': converge,
        'error_type': error_type,
        'resultados_todos': resultados_todos
    })




def sor_view(request):

    size = 2
    A = b = x0 = tabla = mensaje = None
    resultados_todos = None
    w = None
    rho = None
    converge = None
    error_type = "relative"

    if request.method == 'POST':
        try:
            size = int(request.POST.get('matrix_size', 2))

            A = [[float(request.POST.get(f'a_{i}_{j}', 0)) for j in range(size)] for i in range(size)]
            b = [float(request.POST.get(f'b_{i}', 0)) for i in range(size)]
            x0 = [float(request.POST.get(f'x0_{i}', 0)) for i in range(size)]

            tol = float(request.POST.get('tol', '1e-7'))
            niter = int(request.POST.get('niter', 100))
            w = float(request.POST.get('w', 1.0))

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

        except Exception as e:
            mensaje = f"Error: {str(e)}"

    return render(request, 'sor.html', {
        'matrix_size': size,
        'range': range(size),
        'A': A, 'b': b, 'x0': x0,
        'tabla': tabla, 'mensaje': mensaje,
        'w': w,
        'rho': rho, 'converge': converge,
        'error_type': error_type,
        'resultados_todos': resultados_todos
    })
