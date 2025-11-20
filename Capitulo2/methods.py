import numpy as np
import random

def jacobi(x0, A, b, tol, niter, cs):
    x0 = np.array(x0, dtype=float)
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)

    D = np.diag(np.diag(A))
    L = -np.tril(A, -1)
    U = -np.triu(A, 1)
    T = np.linalg.inv(D).dot(L + U)
    C = np.linalg.inv(D).dot(b)
    error = tol + 1
    c = 0
    
    # Store x0 as a list, not unpacked
    tabla = [(c, x0.tolist(), None)]

    while error > tol and c < niter:
        x1 = T.dot(x0) + C
        if cs:
            denominador = np.linalg.norm(x1, ord=np.inf)
            if denominador == 0:
                error = float('inf')
            else:
                error = np.linalg.norm(x1 - x0, ord=np.inf) / denominador
        else:
            error = np.linalg.norm(x1 - x0, ord=np.inf)
        c += 1
        
        # Store x1 as a list, not unpacked
        tabla.append((c, x1.tolist(), error))
        x0 = x1

    if error < tol:
        mensaje = f"La solución se encontró con tolerancia {tol} en {c} iteraciones"
    else:
        mensaje = f"Fracasó en {niter} iteraciones"

    return tabla, mensaje


def gauss_seidel(x0, A, b, tol, niter, cs):
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    x0 = np.array(x0, dtype=float)
    
    D = np.diag(np.diag(A))
    L = -np.tril(A, -1)
    U = -np.triu(A, 1)

    try:
        inv_DL = np.linalg.inv(D - L)
    except np.linalg.LinAlgError:
        return [], "Error: No se puede invertir D - L (puede no ser invertible)."

    T = np.matmul(inv_DL, U)
    C = np.matmul(inv_DL, b)

    tabla = []
    error = tol + 1
    contador = 0
    x_act = x0

    while error > tol and contador < niter:
        x_sig = np.matmul(T, x_act) + C

        if cs:
            denominador = np.linalg.norm(x_sig, ord=np.inf)
            if denominador == 0:
                error = float('inf')
            else:
                error = np.linalg.norm(x_sig - x_act, ord=np.inf) / denominador
        else:
            error = np.linalg.norm(x_sig - x_act, ord=np.inf)

        tabla.append([contador + 1, x_sig.tolist(), round(error, 10) if contador > 0 else None])
        x_act = x_sig
        contador += 1

    if error <= tol:
        mensaje = f"Aproximación encontrada: {x_act.tolist()} con tolerancia {tol}"
    else:
        mensaje = f"Fracasó en {niter} iteraciones. Última aproximación: {x_act.tolist()}"

    return tabla, mensaje

def sor(x0, A, b, tol, niter, w, cs):
    """
    Método SOR para resolver Ax = b.
    
    Parámetros:
    x0: vector inicial
    A: matriz de coeficientes
    b: vector de términos independientes
    tol: tolerancia para el error
    niter: número máximo de iteraciones
    w: factor de relajación (0 < w < 2)
    cs: si True calcula error relativo, si False absoluto
    
    Retorna:
    tabla: lista con filas [iteración, vector x, error]
    mensaje: string con resultado de la convergencia
    """
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    x0 = np.array(x0, dtype=float)

    D = np.diag(np.diag(A))
    L = -np.tril(A, -1)
    U = -np.triu(A, 1)

    try:
        inv_term = np.linalg.inv(D - w * L)
    except np.linalg.LinAlgError:
        return [], "Error: No se puede invertir (D - w*L), matriz singular o no invertible."

    T = inv_term @ ((1 - w) * D + w * U)
    C = w * inv_term @ b

    tabla = []
    error = tol + 1
    contador = 0
    x_act = x0

    while error > tol and contador < niter:
        x_sig = T @ x_act + C

        if cs:
            denom = np.linalg.norm(x_sig, ord=np.inf)
            if denom == 0:
                error = float('inf')
            else:
                error = np.linalg.norm(x_sig - x_act, ord=np.inf) / denom
        else:
            error = np.linalg.norm(x_sig - x_act, ord=np.inf)

        tabla.append([contador + 1, x_sig.tolist(), round(error, 10) if contador > 0 else None])
        x_act = x_sig
        contador += 1

    if error <= tol:
        mensaje = f"Aproximación encontrada en {contador} iteraciones con tolerancia {tol}."
    else:
        mensaje = f"Fracasó en {niter} iteraciones. Última aproximación: {x_act.tolist()}"

    return tabla, mensaje


def calcular_radio_espectral_metodo(A, metodo, w=None):
    """Calcula el radio espectral para un método específico"""
    A = np.array(A, dtype=float)
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


def ejecutar_todos(x0, A, b, tol, niter, cs):
    """
    Ejecuta los métodos Jacobi, Gauss-Seidel y SOR con diferentes valores aleatorios de w.
    
    Parámetros:
    x0: vector inicial
    A: matriz de coeficientes
    b: vector de términos independientes
    tol: tolerancia para el error
    niter: número máximo de iteraciones
    cs: si True calcula error relativo, si False absoluto
    
    Retorna:
    resultados: diccionario con resultados de cada método
    """
    resultados = {}
    A_array = np.array(A, dtype=float)
    b_array = np.array(b, dtype=float)
    
    # Jacobi
    tabla_jacobi, mensaje_jacobi = jacobi(x0, A, b, tol, niter, cs)
    n_alcanzado = len(tabla_jacobi) - 1
    x_sol = tabla_jacobi[-1][1] if tabla_jacobi else []
    error_final = tabla_jacobi[-1][2] if tabla_jacobi and len(tabla_jacobi) > 0 and tabla_jacobi[-1][2] is not None else None
    
    # Calcular residual: ||Ax - b||
    residual = None
    convergio = False
    if x_sol:
        x_sol_array = np.array(x_sol, dtype=float)
        residual = float(np.linalg.norm(A_array.dot(x_sol_array) - b_array))
        convergio = error_final is not None and error_final < tol
    
    # Calcular radio espectral
    rho = calcular_radio_espectral_metodo(A, "jacobi")
    
    resultados['Jacobi'] = {
        'tabla': tabla_jacobi,
        'mensaje': mensaje_jacobi,
        'n_alcanzado': n_alcanzado,
        'x_sol': x_sol,
        'error_final': error_final,
        'residual': residual,
        'rho': rho,
        'convergio': convergio,
        'w': None  # No aplica para Jacobi
    }
    
    # Gauss-Seidel
    tabla_gauss_seidel, mensaje_gauss_seidel = gauss_seidel(x0, A, b, tol, niter, cs)
    n_alcanzado = len(tabla_gauss_seidel)
    x_sol = tabla_gauss_seidel[-1][1] if tabla_gauss_seidel else []
    error_final = tabla_gauss_seidel[-1][2] if tabla_gauss_seidel and len(tabla_gauss_seidel) > 0 and tabla_gauss_seidel[-1][2] is not None else None
    
    # Calcular residual
    residual = None
    convergio = False
    if x_sol:
        x_sol_array = np.array(x_sol, dtype=float)
        residual = float(np.linalg.norm(A_array.dot(x_sol_array) - b_array))
        convergio = error_final is not None and error_final < tol
    
    # Calcular radio espectral
    rho = calcular_radio_espectral_metodo(A, "gauss")
    
    resultados['Gauss-Seidel'] = {
        'tabla': tabla_gauss_seidel,
        'mensaje': mensaje_gauss_seidel,
        'n_alcanzado': n_alcanzado,
        'x_sol': x_sol,
        'error_final': error_final,
        'residual': residual,
        'rho': rho,
        'convergio': convergio,
        'w': None  # No aplica para Gauss-Seidel
    }
    
    # SOR con diferentes valores fijos de w
    valores_w = [0.5, 1.0, 1.5]
    
    for i, w in enumerate(valores_w):
        tabla_sor, mensaje_sor = sor(x0, A, b, tol, niter, w, cs)
        n_alcanzado = len(tabla_sor)
        x_sol = tabla_sor[-1][1] if tabla_sor else []
        error_final = tabla_sor[-1][2] if tabla_sor and len(tabla_sor) > 0 and tabla_sor[-1][2] is not None else None
        
        # Calcular residual
        residual = None
        convergio = False
        if x_sol:
            x_sol_array = np.array(x_sol, dtype=float)
            residual = float(np.linalg.norm(A_array.dot(x_sol_array) - b_array))
            convergio = error_final is not None and error_final < tol
        
        # Calcular radio espectral
        rho = calcular_radio_espectral_metodo(A, "sor", w=w)
        
        resultados[f'SOR_w{w}'] = {
            'tabla': tabla_sor,
            'mensaje': mensaje_sor,
            'n_alcanzado': n_alcanzado,
            'x_sol': x_sol,
            'error_final': error_final,
            'residual': residual,
            'rho': rho,
            'convergio': convergio,
            'w': w
        }
    
    # Identificar el mejor método
    mejor_metodo = None
    menor_iteraciones = float('inf')
    menor_error = float('inf')
    
    for nombre, datos in resultados.items():
        if datos['error_final'] is not None and datos['n_alcanzado'] != "N/A":
            n_iter = datos['n_alcanzado']
            error_final = datos['error_final']
            
            # Comparar primero por número de iteraciones, luego por error
            if n_iter < menor_iteraciones or (n_iter == menor_iteraciones and error_final < menor_error):
                if mejor_metodo:
                    resultados[mejor_metodo]['mejor'] = False
                mejor_metodo = nombre
                menor_iteraciones = n_iter
                menor_error = error_final
                datos['mejor'] = True
            else:
                datos['mejor'] = False
        else:
            datos['mejor'] = False
    
    return resultados


def comparar_errores_metodo(metodo_nombre, x0, A, b, tol, niter, w=None):
    """
    Ejecuta un método específico con error absoluto y error relativo (condición) para comparar.
    
    Args:
        metodo_nombre: Nombre del método ('jacobi', 'gauss_seidel', 'sor')
        x0: Vector inicial
        A: Matriz de coeficientes
        b: Vector de términos independientes
        tol: Tolerancia
        niter: Número máximo de iteraciones
        w: Factor de relajación (solo para SOR)
    
    Returns:
        Lista con resultados para cada tipo de error
    """
    tipos_error = [
        {'nombre': 'Error Absoluto', 'cs': False},
        {'nombre': 'Error Relativo (Condición)', 'cs': True}
    ]
    
    resultados_comparativos = []
    A_array = np.array(A, dtype=float)
    b_array = np.array(b, dtype=float)
    
    for tipo in tipos_error:
        tabla_metodo = []
        mensaje_metodo = ""
        
        # Ejecutar el método según el tipo
        if metodo_nombre == 'jacobi':
            tabla_metodo, mensaje_metodo = jacobi(x0, A, b, tol, niter, tipo['cs'])
        elif metodo_nombre == 'gauss_seidel':
            tabla_metodo, mensaje_metodo = gauss_seidel(x0, A, b, tol, niter, tipo['cs'])
        elif metodo_nombre == 'sor':
            if w is None:
                w = 1.0  # Valor por defecto
            tabla_metodo, mensaje_metodo = sor(x0, A, b, tol, niter, w, tipo['cs'])
        
        # Procesar resultados
        if tabla_metodo and len(tabla_metodo) > 0:
            n_iteraciones = len(tabla_metodo) - 1
            x_sol = tabla_metodo[-1][1] if tabla_metodo else []
            error_final = tabla_metodo[-1][2] if n_iteraciones > 0 and tabla_metodo[-1][2] is not None else 0
            
            # Calcular residual
            residual = None
            convergio = False
            if x_sol:
                x_sol_array = np.array(x_sol, dtype=float)
                residual = float(np.linalg.norm(A_array.dot(x_sol_array) - b_array))
                convergio = error_final is not None and error_final < tol
            
            # Calcular radio espectral
            rho = None
            if metodo_nombre == 'jacobi':
                rho = calcular_radio_espectral_metodo(A, "jacobi")
            elif metodo_nombre == 'gauss_seidel':
                rho = calcular_radio_espectral_metodo(A, "gauss")
            elif metodo_nombre == 'sor' and w is not None:
                rho = calcular_radio_espectral_metodo(A, "sor", w=w)
            
            resultados_comparativos.append({
                'tipo_error': tipo['nombre'],
                'tipo_error_key': 'relativo' if tipo['cs'] else 'absoluto',
                'x_sol': x_sol,
                'n': n_iteraciones,
                'error': error_final,
                'residual': residual,
                'rho': rho,
                'convergio': convergio,
                'mensaje': mensaje_metodo
            })
        else:
            resultados_comparativos.append({
                'tipo_error': tipo['nombre'],
                'tipo_error_key': 'relativo' if tipo['cs'] else 'absoluto',
                'x_sol': "N/A",
                'n': "N/A",
                'error': "N/A",
                'residual': "N/A",
                'rho': None,
                'convergio': False,
                'mensaje': mensaje_metodo
            })
    
    # Identificar el mejor tipo de error
    mejor_resultado = None
    menor_iteraciones = float('inf')
    menor_error = float('inf')
    
    for resultado in resultados_comparativos:
        if resultado['convergio'] and resultado['n'] != "N/A":
            n_iter = resultado['n']
            error_final = resultado['error']
            
            if n_iter < menor_iteraciones or (n_iter == menor_iteraciones and error_final < menor_error):
                if mejor_resultado:
                    mejor_resultado['mejor'] = False
                mejor_resultado = resultado
                menor_iteraciones = n_iter
                menor_error = error_final
                resultado['mejor'] = True
            else:
                resultado['mejor'] = False
        else:
            resultado['mejor'] = False
    
    return resultados_comparativos