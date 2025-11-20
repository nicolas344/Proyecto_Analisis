import numpy as np
from django.shortcuts import render
from .methods import jacobi, gauss_seidel, sor, ejecutar_todos


def optionsch2(request):
    return render(request, 'optionsch2.html')



def jacobi_view(request):

    
    size = 2
    A = None
    b = None
    x0 = None
    tabla = None
    mensaje = None
    usar_cifras = False
    resultados_todos = None

    if request.method == 'POST':
        try:
            size = int(request.POST.get('matrix_size', 2))

            
            A = []
            for i in range(size):
                fila = []
                for j in range(size):
                    val = float(request.POST.get(f'a_{i}_{j}', 0))
                    fila.append(val)
                A.append(fila)

           
            b = [float(request.POST.get(f'b_{i}', 0)) for i in range(size)]

            
            x0 = [float(request.POST.get(f'x0_{i}', 0)) for i in range(size)]

            tol = float(request.POST.get('tol', '1e-7'))
            niter = int(request.POST.get('niter', 100))
            usar_cifras = request.POST.get('usar_cifras') == 'on'

            tabla, mensaje = jacobi(x0, A, b, tol, niter, usar_cifras)

            if request.POST.get('comparar') == 'on':
                resultados_todos = ejecutar_todos(x0, A, b, tol, niter, usar_cifras)

        except Exception as e:
            mensaje = f"Error: {str(e)}"

    return render(request, 'jacobi.html', {
        'matrix_size': size,
        'range': range(size),   
        'A': A,
        'b': b,
        'x0': x0,
        'tabla': tabla,
        'mensaje': mensaje,
        'usar_cifras': usar_cifras,
        'resultados_todos': resultados_todos
    })



def gauss_seidel_view(request):

    size = 2
    A = None
    b = None
    x0 = None
    tabla = None
    mensaje = None
    usar_cifras = False
    resultados_todos = None

    if request.method == 'POST':
        try:
            size = int(request.POST.get('matrix_size', 2))

            A = []
            for i in range(size):
                fila = []
                for j in range(size):
                    val = float(request.POST.get(f'a_{i}_{j}', 0))
                    fila.append(val)
                A.append(fila)

            b = [float(request.POST.get(f'b_{i}', 0)) for i in range(size)]
            x0 = [float(request.POST.get(f'x0_{i}', 0)) for i in range(size)]

            tol = float(request.POST.get('tol', '1e-7'))
            niter = int(request.POST.get('niter', 100))
            usar_cifras = request.POST.get('usar_cifras') == 'on'

            tabla, mensaje = gauss_seidel(x0, A, b, tol, niter, usar_cifras)

            if request.POST.get('comparar') == 'on':
                resultados_todos = ejecutar_todos(x0, A, b, tol, niter, usar_cifras)

        except Exception as e:
            mensaje = f"Error: {str(e)}"

    return render(request, 'gauss_seidel.html', {
        'matrix_size': size,
        'range': range(size),  
        'A': A,
        'b': b,
        'x0': x0,
        'tabla': tabla,
        'mensaje': mensaje,
        'usar_cifras': usar_cifras,
        'resultados_todos': resultados_todos
    })



def sor_view(request):

    size = 2
    A = None
    b = None
    x0 = None
    tabla = None
    mensaje = None
    usar_cifras = False
    resultados_todos = None
    w = None

    if request.method == 'POST':
        try:
            size = int(request.POST.get('matrix_size', 2))

            A = []
            for i in range(size):
                fila = []
                for j in range(size):
                    val = float(request.POST.get(f'a_{i}_{j}', 0))
                    fila.append(val)
                A.append(fila)

            b = [float(request.POST.get(f'b_{i}', 0)) for i in range(size)]
            x0 = [float(request.POST.get(f'x0_{i}', 0)) for i in range(size)]

            tol = float(request.POST.get('tol', '1e-7'))
            niter = int(request.POST.get('niter', 100))
            w = float(request.POST.get('w', 1.0))
            usar_cifras = request.POST.get('usar_cifras') == 'on'

            tabla, mensaje = sor(x0, A, b, tol, niter, w, usar_cifras)

            if request.POST.get('comparar') == 'on':
                resultados_todos = ejecutar_todos(x0, A, b, tol, niter, usar_cifras)

        except Exception as e:
            mensaje = f"Error: {str(e)}"

    return render(request, 'sor.html', {
        'matrix_size': size,
        'range': range(size),   
        'A': A,
        'b': b,
        'x0': x0,
        'w': w,
        'tabla': tabla,
        'mensaje': mensaje,
        'usar_cifras': usar_cifras,
        'resultados_todos': resultados_todos
    })
