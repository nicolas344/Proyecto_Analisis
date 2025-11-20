import numpy as np

def vandermonde_interpolation(x, y):
    """Interpolación usando matriz de Vandermonde"""
    # Calcular matriz de Vandermonde y resolver
    A = np.vander(x, increasing=False)
    a = np.linalg.solve(A, y)

    # Construcción del polinomio como string
    poly_str = "P(x) = "
    degree = len(a) - 1
    terms = []
    for i, coef in enumerate(a):
        power = degree - i
        terms.append(f"{coef:.6f}x^{power}" if power > 1 else
                     f"{coef:.6f}x" if power == 1 else
                     f"{coef:.6f}")
    poly_str += " + ".join(terms)

    # Evaluar el polinomio en un intervalo para graficar
    xpol = np.linspace(min(x), max(x), 500)
    p = np.polyval(a, xpol)

    return a, poly_str, xpol, p


def newton_interpolation(x, y):
    """Interpolación usando diferencias divididas de Newton"""
    n = len(x)
    diff_table = np.zeros((n, n))
    diff_table[:, 0] = y

    for j in range(1, n):
        for i in range(n - j):
            diff_table[i][j] = (diff_table[i+1][j-1] - diff_table[i][j-1]) / (x[i+j] - x[i])

    coef = diff_table[0, :]

    poly_str = f"P(x) = {coef[0]:.6f}"
    for i in range(1, n):
        term = "".join([f"(x - {x[j]:.6f})" for j in range(i)])
        poly_str += f" + ({coef[i]:.6f})*{term}"

    def eval_newton_poly(val, coef, x_points):
        result = coef[0]
        for i in range(1, len(coef)):
            term = coef[i]
            for j in range(i):
                term *= (val - x_points[j])
            result += term
        return result

    xpol = np.linspace(min(x), max(x), 500)
    p = [eval_newton_poly(xi, coef, x) for xi in xpol]

    return coef, poly_str, xpol, p

def lagrange_interpolation(x, y):
    """Interpolación usando polinomios de Lagrange"""
    def L(k, x_val):
        terms = [(x_val - x[j])/(x[k] - x[j]) for j in range(len(x)) if j != k]
        return np.prod(terms)

    def P(x_val):
        return sum(y[k] * L(k, x_val) for k in range(len(x)))

    # Construir string del polinomio
    poly_terms = []
    n = len(x)
    for k in range(n):
        term = f"{y[k]:.6f} * "
        term += " * ".join([f"(x - {x[j]:.2f})/({x[k]:.2f} - {x[j]:.2f})" for j in range(n) if j != k])
        poly_terms.append(term)
    poly_str = "P(x) = " + " + ".join(poly_terms)

    # Evaluar el polinomio en un rango para graficar
    xpol = np.linspace(min(x), max(x), 500)
    p = np.array([P(val) for val in xpol])

    return poly_str, xpol, p

def spline_interpolation(x, y, tipo):
    """Interpolación usando splines (lineal o cúbico)"""
    n = len(x)
    
    if tipo == 'lineal':
        d = 1
    elif tipo == 'cubico':
        d = 3
    else:
        raise ValueError("Tipo de spline inválido")

    A = np.zeros(((d + 1) * (n - 1), (d + 1) * (n - 1)))
    b = np.zeros(((d + 1) * (n - 1),))

    # Variables auxiliares
    cua = x**2
    cub = x**3
    c = 0
    h = 0

    # Construcción del sistema
    if d == 1:  # Lineal
        for i in range(n - 1):
            A[h, c] = x[i]
            A[h, c + 1] = 1
            b[h] = y[i]
            c += 2
            h += 1
        c = 0
        for i in range(1, n):
            A[h, c] = x[i]
            A[h, c + 1] = 1
            b[h] = y[i]
            c += 2
            h += 1

    elif d == 3:  # Cúbico
        for i in range(n - 1):
            A[h, c] = cub[i]
            A[h, c + 1] = cua[i]
            A[h, c + 2] = x[i]
            A[h, c + 3] = 1
            b[h] = y[i]
            c += 4
            h += 1

        c = 0
        for i in range(1, n):
            A[h, c] = cub[i]
            A[h, c + 1] = cua[i]
            A[h, c + 2] = x[i]
            A[h, c + 3] = 1
            b[h] = y[i]
            c += 4
            h += 1

        c = 0
        for i in range(1, n - 1):
            A[h, c] = 3 * cua[i]
            A[h, c + 1] = 2 * x[i]
            A[h, c + 2] = 1
            A[h, c + 4] = -3 * cua[i]
            A[h, c + 5] = -2 * x[i]
            A[h, c + 6] = -1
            b[h] = 0
            c += 4
            h += 1

        c = 0
        for i in range(1, n - 1):
            A[h, c] = 6 * x[i]
            A[h, c + 1] = 2
            A[h, c + 4] = -6 * x[i]
            A[h, c + 5] = -2
            b[h] = 0
            c += 4
            h += 1

        A[h, 0] = 6 * x[0]
        A[h, 1] = 2
        b[h] = 0
        h += 1
        A[h, -4] = 6 * x[-1]
        A[h, -3] = 2
        b[h] = 0

    # Resolver sistema
    val = np.linalg.solve(A, b)
    coef = val.reshape(n - 1, d + 1)

    # Construir string de polinomios
    poly_str = ""
    for i in range(n - 1):
        coefs = coef[i]
        terms = []
        if d == 1:
            # a1 x + a0
            terms.append(f"({coefs[0]:.4f})x + ({coefs[1]:.4f})")
        else:
            # a3 x^3 + a2 x^2 + a1 x + a0
            terms.append(f"({coefs[0]:.4f})x³ + ({coefs[1]:.4f})x² + ({coefs[2]:.4f})x + ({coefs[3]:.4f})")
        poly_str += f"Intervalo [{x[i]:.4f}, {x[i+1]:.4f}]:  {terms[0]} |\n"

    # Evaluar spline en puntos para gráfica
    xpol = np.linspace(x[0], x[-1], 500)
    p = np.zeros_like(xpol)
    for i in range(n - 1):
        mask = (xpol >= x[i]) & (xpol <= x[i + 1])
        xi = xpol[mask]
        if d == 1:
            # a1 x + a0
            p[mask] = coef[i, 0] * xi + coef[i, 1]
        else:
            # a3 x^3 + a2 x^2 + a1 x + a0
            p[mask] = coef[i, 0] * xi**3 + coef[i, 1] * xi**2 + coef[i, 2] * xi + coef[i, 3]

    return poly_str, xpol, p


def comparar_metodos_interpolacion(x, y):
    """
    Compara todos los métodos de interpolación y calcula métricas de error.
    
    Returns:
        Lista de diccionarios con información de cada método incluyendo:
        - metodo: nombre del método
        - polinomio: representación del polinomio
        - error_absoluto: error absoluto promedio
        - error_relativo: error relativo promedio
        - error_abs_max: error absoluto máximo
        - error_rel_max: error relativo máximo
        - mejor: True si es el mejor método
    """
    resultados = []
    
    # Asegurar que hay al menos 2 puntos
    if len(x) < 2:
        return []

    metodos = [
        ("Vandermonde", lambda: vandermonde_interpolation(x, y)),
        ("Newton Interpolante", lambda: newton_interpolation(x, y)),
        ("Lagrange", lambda: lagrange_interpolation(x, y)),
        ("Spline Lineal", lambda: spline_interpolation(x, y, 'lineal')),
        ("Spline Cúbico", lambda: spline_interpolation(x, y, 'cubico'))
    ]
    
    for metodo_nombre, metodo_func in metodos:
        try:
            # Ejecutar el método
            resultado = metodo_func()
            
            # Extraer información según el método
            if metodo_nombre in ["Vandermonde", "Newton Interpolante"]:
                coef, polinomio, x_eval, p_eval = resultado
            else:  # Lagrange y Splines
                polinomio, x_eval, p_eval = resultado
            
            # Calcular errores en todos los puntos originales
            errores_abs = []
            errores_rel = []
            
            # Para cada punto original, calcular el error
            for i in range(len(x)):
                # Encontrar el valor interpolado en x[i]
                idx = np.argmin(np.abs(x_eval - x[i]))
                y_interp = p_eval[idx]
                
                # Calcular errores
                error_abs = abs(y[i] - y_interp)
                errores_abs.append(error_abs)
                
                if y[i] != 0:
                    error_rel = abs((y[i] - y_interp) / y[i])
                    errores_rel.append(error_rel)
            
            # Calcular métricas estadísticas
            error_abs_promedio = np.mean(errores_abs) if errores_abs else 0.0
            error_rel_promedio = np.mean(errores_rel) if errores_rel else 0.0
            error_abs_max = np.max(errores_abs) if errores_abs else 0.0
            error_rel_max = np.max(errores_rel) if errores_rel else 0.0
            
            resultados.append({
                'metodo': metodo_nombre,
                'polinomio': polinomio,
                'error_absoluto': error_abs_promedio,
                'error_relativo': error_rel_promedio,
                'error_abs_max': error_abs_max,
                'error_rel_max': error_rel_max,
                'mejor': False
            })
            
        except Exception as e:
            resultados.append({
                'metodo': metodo_nombre,
                'polinomio': "Error",
                'error_absoluto': float('inf'),
                'error_relativo': float('inf'),
                'error_abs_max': float('inf'),
                'error_rel_max': float('inf'),
                'mejor': False
            })
    
    # Determinar el mejor método (menor error absoluto promedio)
    if resultados:
        mejor_idx = min(range(len(resultados)), 
                       key=lambda i: resultados[i]['error_absoluto'])
        resultados[mejor_idx]['mejor'] = True
    
    return resultados


def comparar_errores_metodo(metodo_nombre, x, y, tipo_spline=None):
    """
    Compara un método de interpolación usando error absoluto vs error relativo.
    
    Args:
        metodo_nombre: nombre del método ('vandermonde', 'newton', 'lagrange', 'spline_lineal', 'spline_cubico')
        x: vector de puntos x
        y: vector de puntos y
        tipo_spline: tipo de spline si el método es spline ('lineal' o 'cubico')
    
    Returns:
        Lista de diccionarios con los resultados de cada tipo de error
    """
    resultados = []
    
    # Verificar que hay suficientes puntos
    if len(x) < 2:
        return []
    
    # Seleccionar el método a ejecutar
    if metodo_nombre == 'vandermonde':
        metodo_func = lambda: vandermonde_interpolation(x, y)
    elif metodo_nombre == 'newton':
        metodo_func = lambda: newton_interpolation(x, y)
    elif metodo_nombre == 'lagrange':
        metodo_func = lambda: lagrange_interpolation(x, y)
    elif metodo_nombre == 'spline_lineal':
        metodo_func = lambda: spline_interpolation(x, y, 'lineal')
    elif metodo_nombre == 'spline_cubico':
        metodo_func = lambda: spline_interpolation(x, y, 'cubico')
    else:
        return []
    
    # Ejecutar el método una vez para obtener los resultados base
    try:
        resultado = metodo_func()
        
        # Extraer información según el método
        if metodo_nombre in ['vandermonde', 'newton']:
            coef, polinomio, x_eval, p_eval = resultado
        else:  # lagrange y splines
            polinomio, x_eval, p_eval = resultado
        
        # Calcular errores ABSOLUTOS en todos los puntos originales
        errores_abs = []
        for i in range(len(x)):
            idx = np.argmin(np.abs(x_eval - x[i]))
            y_interp = p_eval[idx]
            error_abs = abs(y[i] - y_interp)
            errores_abs.append(error_abs)
        
        error_abs_promedio = np.mean(errores_abs) if errores_abs else 0.0
        error_abs_max = np.max(errores_abs) if errores_abs else 0.0
        
        resultados.append({
            'tipo_error': 'Absoluto',
            'polinomio': polinomio,
            'error_promedio': error_abs_promedio,
            'error_max': error_abs_max,
            'convergio': True,
            'mejor': False
        })
        
        # Calcular errores RELATIVOS en todos los puntos originales
        errores_rel = []
        for i in range(len(x)):
            idx = np.argmin(np.abs(x_eval - x[i]))
            y_interp = p_eval[idx]
            
            if y[i] != 0:
                error_rel = abs((y[i] - y_interp) / y[i])
                errores_rel.append(error_rel)
            else:
                # Si y[i] es 0, usar error absoluto
                errores_rel.append(abs(y[i] - y_interp))
        
        error_rel_promedio = np.mean(errores_rel) if errores_rel else 0.0
        error_rel_max = np.max(errores_rel) if errores_rel else 0.0
        
        resultados.append({
            'tipo_error': 'Relativo',
            'polinomio': polinomio,
            'error_promedio': error_rel_promedio,
            'error_max': error_rel_max,
            'convergio': True,
            'mejor': False
        })
        
        # Determinar cuál es mejor (menor error promedio)
        if resultados:
            mejor_idx = 0 if resultados[0]['error_promedio'] <= resultados[1]['error_promedio'] else 1
            resultados[mejor_idx]['mejor'] = True
            
    except Exception as e:
        # Si hay error, devolver resultados vacíos
        return []
    
    return resultados