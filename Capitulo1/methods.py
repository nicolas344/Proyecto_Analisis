from sympy import symbols, sympify, Symbol, diff
from sympy.utilities.lambdify import lambdify
import numpy as np

def calcular_error(x_nuevo, x_anterior, tipo_error):
    """
    Calcula el error según el tipo especificado.
    
    Args:
        x_nuevo: Valor actual de la aproximación
        x_anterior: Valor anterior de la aproximación
        tipo_error: 'absoluto', 'relativo' o 'condicion'
    
    Returns:
        Error calculado según el tipo
    """
    if tipo_error == 'absoluto':
        return abs(x_nuevo - x_anterior)
    elif tipo_error == 'relativo':
        if x_nuevo == 0:
            return float('inf')
        return abs((x_nuevo - x_anterior) / x_nuevo)
    elif tipo_error == 'condicion':
        # Error de condición: |f(x_nuevo)| / |x_nuevo|
        # Para métodos que no tienen f(x) disponible, usar error relativo
        return abs((x_nuevo - x_anterior) / x_nuevo) if x_nuevo != 0 else float('inf')
    else:
        # Por defecto, usar error absoluto
        return abs(x_nuevo - x_anterior)

def biseccion(f_expr_str, xi, xs, Tol, niter, tipo_error='absoluto'):
    """
    Método de Bisección para encontrar raíces.
    
    Args:
        tipo_error: 'absoluto', 'relativo' o 'condicion'
    """
    x = symbols('x')
    f_expr = sympify(f_expr_str)
    f = lambdify(x, f_expr, 'math')

    fi = f(xi)
    fs = f(xs)

    resultados = []

    if fi == 0:
        resultados.append((0, xi, fi, 0))
        return resultados, xi, "es raíz exacta"
    elif fs == 0:
        resultados.append((0, xs, fs, 0))
        return resultados, xs, "es raíz exacta"
    elif np.iscomplex(fi) or np.iscomplex(fs):
        return [], None, "La función toma valores complejos en el intervalo dado"
    elif fi * fs < 0:
        c = 0
        xm = (xi + xs) / 2
        fe = f(xm)
        error = Tol + 1
        resultados.append((c, xm, fe, None))

        while error > Tol and fe != 0 and c < niter:
            if fi * fe < 0:
                xs = xm
                fs = f(xs)
            else:
                xi = xm
                fi = f(xi)
            xa = xm
            xm = (xi + xs) / 2
            fe = f(xm)
            
            # Calcular error según el tipo especificado
            if tipo_error == 'condicion':
                # Error de condición: |f(xm)| (cuánto se aleja de cero)
                error = abs(fe)
            else:
                error = calcular_error(xm, xa, tipo_error)
            
            c += 1
            resultados.append((c, xm, fe, error))

        if fe == 0:
            mensaje = "es raíz exacta"
        elif error < Tol:
            mensaje = f"es una aproximación con tolerancia {Tol}"
        else:
            mensaje = f"falló en {niter} iteraciones"
        return resultados, xm, mensaje
    else:
        return [], None, "Intervalo inadecuado"

def regla_falsa(f_str, xi, xs, tol, niter, tipo_error='absoluto'):
    """
    Método de Regla Falsa para encontrar raíces.
    
    Args:
        tipo_error: 'absoluto', 'relativo' o 'condicion'
    """
    x = Symbol('x')
    f_expr = sympify(f_str)
    f = lambdify(x, f_expr, 'math')  # 'math' para coherencia con bisección

    fxi = f(xi)
    fxs = f(xs)

    if fxi == 0:
        return [(0, xi, fxi, 0)], xi, "es raíz exacta"
    elif fxs == 0:
        return [(0, xs, fxs, 0)], xs, "es raíz exacta"
    elif np.iscomplex(fxi) or np.iscomplex(fxs):
        return [], None, "La función toma valores complejos en el intervalo dado"
    elif fxi * fxs < 0:
        tabla = []
        error = tol + 1
        xm_old = xi
        c = 0
        xm = xs - fxs * (xi - xs) / (fxi - fxs)
        fxm = f(xm)
        tabla.append((c, xm, fxm, None))

        while error > tol and fxm != 0 and c < niter:
            if fxi * fxm < 0:
                xs = xm
                fxs = fxm
            else:
                xi = xm
                fxi = fxm

            xm_old = xm
            xm = xs - fxs * (xi - xs) / (fxi - fxs)
            fxm = f(xm)
            
            # Calcular error según el tipo especificado
            if tipo_error == 'condicion':
                error = abs(fxm)
            else:
                error = calcular_error(xm, xm_old, tipo_error)
            
            c += 1
            tabla.append((c, xm, fxm, error))

        if fxm == 0:
            mensaje = "es raíz exacta"
        elif error < tol:
            mensaje = f"es una aproximación con tolerancia {tol}"
        else:
            mensaje = f"falló en {niter} iteraciones"
        return tabla, xm, mensaje
    else:
        return [], None, "Intervalo inadecuado"

def punto_fijo(x0, tol, niter, f_str, g_str, tipo_error='absoluto'):
    """
    Método de Punto Fijo para encontrar raíces.
    
    Args:
        tipo_error: 'absoluto', 'relativo' o 'condicion'
    """
    x = symbols('x')
    f_expr = sympify(f_str)
    g_expr = sympify(g_str)

    f = lambdify(x, f_expr, modules=["numpy"])
    g = lambdify(x, g_expr, modules=["numpy"])

    tabla = []
    c = 0
    xn = x0
    fxn = f(xn)
    error = tol + 1

    tabla.append((c, xn, fxn, None))

    while error > tol and fxn != 0 and c < niter:
        x_next = g(xn)
        fxn = f(x_next)
        
        # Calcular error según el tipo especificado
        if tipo_error == 'condicion':
            error = abs(fxn)
        else:
            error = calcular_error(x_next, xn, tipo_error)
        
        xn = x_next
        c += 1
        tabla.append((c, xn, fxn, error))

    if fxn == 0:
        mensaje = f"{xn} es raíz de f(x)"
    elif error < tol:
        mensaje = f"{xn} es una aproximación de una raíz con tolerancia = {tol}"
    else:
        mensaje = f"Fracasó en {niter} iteraciones"

    return xn, tabla, mensaje

def newton_raphson(x0, tol, niter, f_str, tipo_error='absoluto'):
    """
    Método de Newton-Raphson para encontrar raíces.
    
    Args:
        tipo_error: 'absoluto', 'relativo' o 'condicion'
    """
    x = symbols('x')
    f_expr = sympify(f_str)
    df_expr = diff(f_expr, x)

    f = lambdify(x, f_expr, modules=["numpy"])
    df = lambdify(x, df_expr, modules=["numpy"])

    tabla = []
    c = 0
    xn = x0
    fxn = f(xn)
    dfxn = df(xn)
    error = tol + 1

    tabla.append((c, xn, fxn, None))

    while error > tol and fxn != 0 and dfxn != 0 and c < niter:
        x_next = xn - fxn / dfxn
        fxn = f(x_next)
        dfxn = df(x_next)
        
        # Calcular error según el tipo especificado
        if tipo_error == 'condicion':
            error = abs(fxn)
        else:
            error = calcular_error(x_next, xn, tipo_error)
        
        xn = x_next
        c += 1
        tabla.append((c, xn, fxn, error))

    # Generar mensaje de resultado
    if fxn == 0:
        mensaje = f"{xn} es raíz de f(x)"
    elif error < tol:
        mensaje = f"{xn} es una aproximación de una raíz con tolerancia = {tol}"
    elif dfxn == 0:
        mensaje = f"{xn} es una posible raíz múltiple (f'(x) = 0)"
    else:
        mensaje = f"Fracasó en {niter} iteraciones"

    return xn, tabla, mensaje

def secante(x0, x1, tol, niter, f_str, tipo_error='absoluto'):
    """
    Método de la Secante para encontrar raíces.
    
    Args:
        tipo_error: 'absoluto', 'relativo' o 'condicion'
    """
    x = symbols('x')
    f_expr = sympify(f_str)
    f = lambdify(x, f_expr, modules=["numpy"])

    xn = [x0, x1]
    fm = [f(x0), f(x1)]
    E = [tol + 1]
    tabla = []

    # Primeras dos filas
    tabla.append((0, xn[0], fm[0], None))
    tabla.append((1, xn[1], fm[1], E[0]))

    c = 1
    while E[-1] > tol and fm[-1] != 0 and c < niter:
        try:
            x_next = xn[-1] - (fm[-1] * (xn[-1] - xn[-2])) / (fm[-1] - fm[-2])
        except ZeroDivisionError:
            break

        fx_next = f(x_next)
        
        # Calcular error según el tipo especificado
        if tipo_error == 'condicion':
            error = abs(fx_next)
        else:
            error = calcular_error(x_next, xn[-1], tipo_error)

        xn.append(x_next)
        fm.append(fx_next)
        E.append(error)

        c += 1
        tabla.append((c, x_next, fx_next, error))

    # Mensaje final
    if fm[-1] == 0:
        mensaje = f"{xn[-1]} es raíz de f(x)"
    elif E[-1] < tol:
        mensaje = f"{xn[-1]} es una aproximación de una raíz con tolerancia = {tol}"
    else:
        mensaje = f"Fracasó en {niter} iteraciones"

    return xn[-1], tabla, mensaje

def raices_multiples(x0, tol, niter, f_str, tipo_error='absoluto'):
    """
    Método de Raíces Múltiples para encontrar raíces con multiplicidad > 1.
    
    Args:
        tipo_error: 'absoluto', 'relativo' o 'condicion'
    """
    x = symbols('x')
    f_expr = sympify(f_str)
    df_expr = diff(f_expr, x)
    ddf_expr = diff(df_expr, x)

    f = lambdify(x, f_expr, modules=['numpy'])
    df = lambdify(x, df_expr, modules=['numpy'])
    ddf = lambdify(x, ddf_expr, modules=['numpy'])

    tabla = []
    c = 0
    fx = f(x0)
    dfx = df(x0)
    ddfx = ddf(x0)
    error = tol + 1

    tabla.append((c, x0, fx, error if c > 0 else None))

    while error > tol and fx != 0 and dfx != 0 and c < niter:
        denom = dfx**2 - fx * ddfx
        if denom == 0:
            break
        x1 = x0 - (fx * dfx) / denom
        
        # Calcular error según el tipo especificado
        if tipo_error == 'condicion':
            fx_temp = f(x1)
            error = abs(fx_temp)
        else:
            error = calcular_error(x1, x0, tipo_error)
        
        x0 = x1
        fx = f(x0)
        dfx = df(x0)
        ddfx = ddf(x0)
        c += 1
        tabla.append((c, x0, fx, error))

    if fx == 0:
        mensaje = f"{x0} es raíz de f(x)"
    elif error < tol:
        mensaje = f"{x0} es una aproximación de una raíz con tolerancia = {tol}"
    elif dfx == 0:
        mensaje = f"{x0} es una posible raíz múltiple (f'(x) = 0)"
    else:
        mensaje = f"Fracasó en {niter} iteraciones"

    return x0, tabla, mensaje


def calcular_funcion_g_optima(f_str, x0):
    """
    Calcula automáticamente una buena función g(x) para el método de punto fijo.
    Busca un valor de k para g(x) = x - f(x)/k que optimice la convergencia.
    """
    x = symbols('x')
    f_expr = sympify(f_str)
    f_prime = diff(f_expr, x)
    
    # Evaluar f'(x0)
    f_prime_func = lambdify(x, f_prime, modules=["numpy"])
    f_prime_x0 = f_prime_func(x0)
    
    # Si f'(x0) es cercano a cero, usamos un valor predeterminado
    if abs(f_prime_x0) < 1e-10:
        k = 1.0
        convergencia_info = "f'(x0) ≈ 0, usando k=1"
    else:
        # Probar diferentes valores de k
        best_k = 1.0
        best_derivative = float('inf')
        
        # Probar valores de k en un rango
        for k in np.linspace(0.5, 10, 20):
            # g(x) = x - f(x)/k
            g_expr = x - f_expr/k
            g_prime = diff(g_expr, x)
            g_prime_func = lambdify(x, g_prime, modules=["numpy"])
            
            try:
                g_prime_x0 = abs(g_prime_func(x0))
                if g_prime_x0 < best_derivative and g_prime_x0 < 1:
                    best_derivative = g_prime_x0
                    best_k = k
            except:
                continue
        
        k = best_k
        convergencia_info = f"|g'(x0)| = {best_derivative:.4f} con k={k:.4f}"
    
    # Construir la función g(x)
    g_str = f"x - ({f_str})/{k}"
    
    return g_str, convergencia_info

def calcular_estadisticas_error(tabla):
    """
    Calcula estadísticas de error de una tabla de iteraciones.
    
    Returns:
        dict con error_abs_promedio, error_rel_promedio, error_abs_max, error_rel_max
    """
    if not tabla or len(tabla) <= 1:
        return {
            'error_abs_promedio': 0.0,
            'error_rel_promedio': 0.0,
            'error_abs_max': 0.0,
            'error_rel_max': 0.0
        }
    
    errores = []
    valores_x = []
    
    # Recopilar errores y valores de x
    for fila in tabla:
        if len(fila) >= 4 and fila[3] is not None and fila[3] != "N/A":
            try:
                error = float(fila[3])
                errores.append(error)
            except:
                pass
        # Obtener el valor de x (índice 1 para la mayoría de métodos)
        if len(fila) >= 2:
            try:
                x_val = float(fila[1])
                valores_x.append(x_val)
            except:
                pass
    
    if not errores:
        return {
            'error_abs_promedio': 0.0,
            'error_rel_promedio': 0.0,
            'error_abs_max': 0.0,
            'error_rel_max': 0.0
        }
    
    # Calcular errores absolutos
    errores_abs = [abs(e) for e in errores]
    error_abs_promedio = np.mean(errores_abs)
    error_abs_max = np.max(errores_abs)
    
    # Calcular errores relativos
    errores_rel = []
    for i, x_val in enumerate(valores_x[1:], 1):  # Empezar desde segundo valor
        if x_val != 0 and i < len(errores):
            err_rel = abs(errores[i] / x_val)
            errores_rel.append(err_rel)
    
    error_rel_promedio = np.mean(errores_rel) if errores_rel else 0.0
    error_rel_max = np.max(errores_rel) if errores_rel else 0.0
    
    return {
        'error_abs_promedio': error_abs_promedio,
        'error_rel_promedio': error_rel_promedio,
        'error_abs_max': error_abs_max,
        'error_rel_max': error_rel_max
    }

def ejecutar_todos(f_str, g_str, xi, xs, tol, niter, x1, tipo_error='absoluto'):
    """
    Ejecuta todos los métodos y devuelve los resultados comparativos con estadísticas detalladas.
    Identifica automáticamente el mejor método.
    
    Args:
        tipo_error: 'absoluto', 'relativo' o 'condicion'
    
    Returns:
        Lista de resultados con el mejor método marcado y estadísticas de error completas
    """
    resultados_comparativos = []

    x0 = xi

    if xs is None:
        xi = x0 - 2
        xs = x0 + 2

    resultados = {}
    
    # Bisección
    tabla_biseccion, resultado_biseccion, mensaje_biseccion = biseccion(f_str, xi, xs, tol, niter, tipo_error)
    if resultado_biseccion is not None:
        n_biseccion = len(tabla_biseccion) - 1
        error_biseccion = tabla_biseccion[-1][3] if n_biseccion > 0 and tabla_biseccion[-1][3] is not None else 0
        estadisticas = calcular_estadisticas_error(tabla_biseccion)
        resultados_comparativos.append({
            'metodo': 'Bisección',
            'xs': resultado_biseccion,
            'n': n_biseccion,
            'error': error_biseccion,
            **estadisticas
        })
    else:
        resultados_comparativos.append({
            'metodo': 'Bisección',
            'xs': "N/A",
            'n': "N/A",
            'error': "N/A",
            'info': mensaje_biseccion,
            'error_abs_promedio': float('inf'),
            'error_rel_promedio': float('inf'),
            'error_abs_max': float('inf'),
            'error_rel_max': float('inf')
        })
    
    # Regla Falsa
    tabla_regla_falsa, resultado_regla_falsa, mensaje_regla_falsa = regla_falsa(f_str, xi, xs, tol, niter, tipo_error)
    if resultado_regla_falsa is not None:
        n_regla_falsa = len(tabla_regla_falsa) - 1
        error_regla_falsa = tabla_regla_falsa[-1][3] if n_regla_falsa > 0 and tabla_regla_falsa[-1][3] is not None else 0
        estadisticas = calcular_estadisticas_error(tabla_regla_falsa)
        resultados_comparativos.append({
            'metodo': 'Regla Falsa',
            'xs': resultado_regla_falsa,
            'n': n_regla_falsa,
            'error': error_regla_falsa,
            **estadisticas
        })
    else:
        resultados_comparativos.append({
            'metodo': 'Regla Falsa',
            'xs': "N/A",
            'n': "N/A",
            'error': "N/A",
            'info': mensaje_regla_falsa,
            'error_abs_promedio': float('inf'),
            'error_rel_promedio': float('inf'),
            'error_abs_max': float('inf'),
            'error_rel_max': float('inf')
        })
    
    # Punto Fijo
    if g_str is None:
        g_str, convergencia_info = calcular_funcion_g_optima(f_str, x0)
    else:
        convergencia_info = "Función g proporcionada por el usuario"
    resultado_punto_fijo, tabla_punto_fijo, mensaje_punto_fijo = punto_fijo(x0, tol, niter, f_str, g_str, tipo_error)
    if resultado_punto_fijo is not None:
        n_punto_fijo = len(tabla_punto_fijo) - 1
        error_punto_fijo = tabla_punto_fijo[-1][3] if n_punto_fijo > 0 and tabla_punto_fijo[-1][3] is not None else 0
        estadisticas = calcular_estadisticas_error(tabla_punto_fijo)
        resultados_comparativos.append({
            'metodo': 'Punto Fijo',
            'xs': resultado_punto_fijo,
            'n': n_punto_fijo,
            'error': error_punto_fijo,
            'g_str': g_str,
            'convergencia_info': convergencia_info,
            **estadisticas
        })
    else:
        resultados_comparativos.append({
            'metodo': 'Punto Fijo',
            'xs': "N/A",
            'n': "N/A",
            'error': "N/A",
            'info': mensaje_punto_fijo,
            'error_abs_promedio': float('inf'),
            'error_rel_promedio': float('inf'),
            'error_abs_max': float('inf'),
            'error_rel_max': float('inf')
        })

    # Newton-Raphson
    resultado_newton, tabla_newton, mensaje_newton = newton_raphson(x0, tol, niter, f_str, tipo_error)
    if resultado_newton is not None:
        n_newton = len(tabla_newton) - 1
        error_newton = tabla_newton[-1][3] if n_newton > 0 and tabla_newton[-1][3] is not None else 0
        estadisticas = calcular_estadisticas_error(tabla_newton)
        resultados_comparativos.append({
            'metodo': 'Newton',
            'xs': resultado_newton,
            'n': n_newton,
            'error': error_newton,
            **estadisticas
        })
    else:
        resultados_comparativos.append({
            'metodo': 'Newton',
            'xs': "N/A",
            'n': "N/A",
            'error': "N/A",
            'info': mensaje_newton,
            'error_abs_promedio': float('inf'),
            'error_rel_promedio': float('inf'),
            'error_abs_max': float('inf'),
            'error_rel_max': float('inf')
        })
    
    # Secante
    if x1 is None:
        x1 = x0 + 1
    resultado_secante, tabla_secante, mensaje_secante = secante(x0, x1, tol, niter, f_str, tipo_error)
    if resultado_secante is not None:
        n_secante = len(tabla_secante) - 1
        error_secante = tabla_secante[-1][3] if n_secante > 0 and tabla_secante[-1][3] is not None else 0
        estadisticas = calcular_estadisticas_error(tabla_secante)
        resultados_comparativos.append({
            'metodo': 'Secante',
            'xs': resultado_secante,
            'n': n_secante,
            'error': error_secante,
            **estadisticas
        })
    else:
        resultados_comparativos.append({
            'metodo': 'Secante',
            'xs': "N/A",
            'n': "N/A",
            'error': "N/A",
            'info': mensaje_secante,
            'error_abs_promedio': float('inf'),
            'error_rel_promedio': float('inf'),
            'error_abs_max': float('inf'),
            'error_rel_max': float('inf')
        })
    
    # Raíces Múltiples
    resultado_rm, tabla_rm, mensaje_rm = raices_multiples(x0, tol, niter, f_str, tipo_error)
    if resultado_rm is not None:
        n_rm = len(tabla_rm) - 1
        error_rm = tabla_rm[-1][3] if n_rm > 0 and tabla_rm[-1][3] is not None else 0
        estadisticas = calcular_estadisticas_error(tabla_rm)
        resultados_comparativos.append({
            'metodo': 'Raíces Múltiples',
            'xs': resultado_rm,
            'n': n_rm,
            'error': error_rm,
            **estadisticas
        })
    else:
        resultados_comparativos.append({
            'metodo': 'Raíces Múltiples',
            'xs': "N/A",
            'n': "N/A",
            'error': "N/A",
            'info': mensaje_rm,
            'error_abs_promedio': float('inf'),
            'error_rel_promedio': float('inf'),
            'error_abs_max': float('inf'),
            'error_rel_max': float('inf')
        })
    
    # Identificar el mejor método
    # Criterio: menor error promedio absoluto, luego menor número de iteraciones
    mejor_metodo = None
    menor_error_promedio = float('inf')
    menor_iteraciones = float('inf')
    
    for resultado in resultados_comparativos:
        if resultado['n'] != "N/A" and resultado['error'] != "N/A":
            error_prom = resultado.get('error_abs_promedio', float('inf'))
            n_iter = resultado['n']
            
            # Comparar primero por error promedio, luego por iteraciones
            if error_prom < menor_error_promedio or (error_prom == menor_error_promedio and n_iter < menor_iteraciones):
                if mejor_metodo:
                    mejor_metodo['mejor'] = False
                mejor_metodo = resultado
                menor_error_promedio = error_prom
                menor_iteraciones = n_iter
                resultado['mejor'] = True
            else:
                resultado['mejor'] = False
    
    return resultados_comparativos