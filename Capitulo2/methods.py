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
    
    # Jacobi
    tabla_jacobi, mensaje_jacobi = jacobi(x0, A, b, tol, niter, cs)
    n_alcanzado = len(tabla_jacobi) - 1
    x_sol = tabla_jacobi[-1][1] if tabla_jacobi else []
    error_final = tabla_jacobi[-1][2] if tabla_jacobi and len(tabla_jacobi) > 0 and tabla_jacobi[-1][2] is not None else "N/A"
    
    resultados['Jacobi'] = {
        'tabla': tabla_jacobi,
        'mensaje': mensaje_jacobi,
        'n_alcanzado': n_alcanzado,
        'x_sol': x_sol,
        'error_final': error_final,
        'w': None  # No aplica para Jacobi
    }
    
    # Gauss-Seidel
    tabla_gauss_seidel, mensaje_gauss_seidel = gauss_seidel(x0, A, b, tol, niter, cs)
    n_alcanzado = len(tabla_gauss_seidel)
    x_sol = tabla_gauss_seidel[-1][1] if tabla_gauss_seidel else []
    error_final = tabla_gauss_seidel[-1][2] if tabla_gauss_seidel and len(tabla_gauss_seidel) > 0 and tabla_gauss_seidel[-1][2] is not None else "N/A"
    
    resultados['Gauss-Seidel'] = {
        'tabla': tabla_gauss_seidel,
        'mensaje': mensaje_gauss_seidel,
        'n_alcanzado': n_alcanzado,
        'x_sol': x_sol,
        'error_final': error_final,
        'w': None  # No aplica para Gauss-Seidel
    }
    
    # SOR con diferentes valores aleatorios de w
    # Generar valores aleatorios de w cercanos a 1 (entre 0.5 y 1.5)
    valores_w = [
        round(random.uniform(0.5, 0.9), 2),  # Valor entre 0.5 y 0.9
        round(random.uniform(1.0, 1.2), 2),  # Valor entre 1.0 y 1.2
        round(random.uniform(1.3, 1.8), 2)   # Valor entre 1.3 y 1.8
    ]
    
    for i, w in enumerate(valores_w):
        tabla_sor, mensaje_sor = sor(x0, A, b, tol, niter, w, cs)
        n_alcanzado = len(tabla_sor)
        x_sol = tabla_sor[-1][1] if tabla_sor else []
        error_final = tabla_sor[-1][2] if tabla_sor and len(tabla_sor) > 0 and tabla_sor[-1][2] is not None else "N/A"
        
        resultados[f'SOR{i+1}'] = {
            'tabla': tabla_sor,
            'mensaje': mensaje_sor,
            'n_alcanzado': n_alcanzado,
            'x_sol': x_sol,
            'error_final': error_final,
            'w': w  # Guardar el valor de w usado
        }
    
    return resultados