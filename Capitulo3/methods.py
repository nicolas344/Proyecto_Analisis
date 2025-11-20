import numpy as np

def vandermonde_interpolation(x, y, x_eliminado=None):
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

    # Eliminar un punto aleatorio y recalcular
    if len(x) > 2:
        if x_eliminado is not None:
            index_to_remove = x_eliminado
        else:
            index_to_remove = np.random.randint(0, len(x))
        x_reduced = np.delete(x, index_to_remove)
        y_reduced = np.delete(y, index_to_remove)
        A_reduced = np.vander(x_reduced, increasing=False)
        a_reduced = np.linalg.solve(A_reduced, y_reduced)

        y_est = np.polyval(a_reduced, x[index_to_remove])
        error = abs(y[index_to_remove] - y_est)
        mensaje_error = (f"Se eliminó el punto x = {x[index_to_remove]:.2f}. "
                         f"El valor estimado con la nueva interpolación es y = {y_est:.4f}, "
                         f"el valor real era y = {y[index_to_remove]:.4f}, "
                         f"error = {error:.4f}")
    else:
        mensaje_error = "No se puede eliminar un punto si solo hay dos."

    return a, poly_str, xpol, p, mensaje_error


def newton_interpolation(x, y, x_eliminado=None):
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

    # Calcular error al eliminar un punto (sin recursividad)
    if n > 2:
        if x_eliminado is not None:
            index_to_remove = x_eliminado
        else:
            index_to_remove = np.random.randint(0, n)
        x_reduced = np.delete(x, index_to_remove)
        y_reduced = np.delete(y, index_to_remove)

        # Calcular coeficientes del polinomio reducido
        m = len(x_reduced)
        diff_table_reduced = np.zeros((m, m))
        diff_table_reduced[:, 0] = y_reduced
        for j in range(1, m):
            for i in range(m - j):
                diff_table_reduced[i][j] = (diff_table_reduced[i+1][j-1] - diff_table_reduced[i][j-1]) / (x_reduced[i+j] - x_reduced[i])
        coef_reduced = diff_table_reduced[0, :]

        # Evaluar polinomio reducido en el punto eliminado
        y_est = eval_newton_poly(x[index_to_remove], coef_reduced, x_reduced)
        error = abs(y[index_to_remove] - y_est)
        mensaje_error = (f"Se eliminó el punto x = {x[index_to_remove]:.2f}. "
                         f"El valor estimado con la nueva interpolación es y = {y_est:.4f}, "
                         f"el valor real era y = {y[index_to_remove]:.4f}, "
                         f"error = {error:.4f}")
    else:
        mensaje_error = "No se puede eliminar un punto si solo hay dos."

    return coef, poly_str, xpol, p, mensaje_error

def lagrange_interpolation(x, y, x_eliminado=None):
    def L(k, x_val):
        terms = [(x_val - x[j])/(x[k] - x[j]) for j in range(len(x)) if j != k]
        return np.prod(terms)

    def P(x_val):
        return sum(y[k] * L(k, x_val) for k in range(len(x)))

    # Construir string del polinomio simbólicamente
    # Esto es complejo para polinomio Lagrange general, aquí simplificamos imprimiendo términos base
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

    # Quitar un punto aleatorio y recalcular
    if len(x) > 2:
        if x_eliminado is not None:
            index_to_remove = x_eliminado
        else:
            index_to_remove = np.random.randint(0, len(x))
        x_reduced = np.delete(x, index_to_remove)
        y_reduced = np.delete(y, index_to_remove)

        def L_reduced(k, x_val):
            terms = [(x_val - x_reduced[j])/(x_reduced[k] - x_reduced[j]) for j in range(len(x_reduced)) if j != k]
            return np.prod(terms)

        def P_reduced(x_val):
            return sum(y_reduced[k] * L_reduced(k, x_val) for k in range(len(x_reduced)))

        y_est = P_reduced(x[index_to_remove])
        error = abs(y[index_to_remove] - y_est)
        mensaje_error = (f"Se eliminó el punto x = {x[index_to_remove]:.2f}. "
                         f"Valor estimado: y = {y_est:.4f}, "
                         f"valor real: y = {y[index_to_remove]:.4f}, "
                         f"error = {error:.4f}")
    else:
        mensaje_error = "No se puede eliminar un punto si solo hay dos."

    return poly_str, xpol, p, mensaje_error

def spline_interpolation(x, y, tipo):
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

    # Método de validación: quitar un dato y recalcular
    if n > 2:
        # quitar el segundo punto (índice 1)
        idx_quitar = 1
        x_red = np.delete(x, idx_quitar)
        y_red = np.delete(y, idx_quitar)

        coef_red, _, _, _ = spline_interpolation_aux(x_red, y_red, d)

        # Evaluar en el x eliminado
        xq = x[idx_quitar]

        # Evaluar spline reducido en xq
        y_est = evaluar_spline_punto(coef_red, x_red, xq, d)

        error = abs(y[idx_quitar] - y_est)
        mensaje_error = (f"Se eliminó el dato en x = {xq:.4f}. "
                         f"Valor estimado: {y_est:.4f}, "
                         f"Valor real: {y[idx_quitar]:.4f}, "
                         f"Error = {error:.4f}")
    else:
        mensaje_error = "No se puede eliminar un punto si solo hay dos."

    return poly_str, xpol, p, mensaje_error


def spline_interpolation_aux(x, y, d):
    # Igual que spline_interpolation pero sin la parte de validación y gráficos,
    # solo para resolver sistema y devolver coeficientes.
    n = len(x)
    A = np.zeros(((d + 1) * (n - 1), (d + 1) * (n - 1)))
    b = np.zeros(((d + 1) * (n - 1),))

    cua = x**2
    cub = x**3
    c = 0
    h = 0

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

    val = np.linalg.solve(A, b)
    coef = val.reshape(n - 1, d + 1)
    return coef, None, None, None


def evaluar_spline_punto(coef, x, xi, d):
    # Evalúa spline en xi usando coeficientes y nodos x
    n = len(x)
    # Buscar intervalo correcto
    for i in range(n - 1):
        if x[i] <= xi <= x[i + 1]:
            if d == 1:
                return coef[i, 0] * xi + coef[i, 1]
            else:
                return coef[i, 0] * xi**3 + coef[i, 1] * xi**2 + coef[i, 2] * xi + coef[i, 3]
    # Si no está en rango, devolver None o extrapolar (aquí None)
    return None



def comparar_metodos_interpolacion(x, y):
    """
    Compara todos los métodos de interpolación y calcula métricas de error detalladas.
    
    Returns:
        Lista de diccionarios con información de cada método incluyendo:
        - metodo: nombre del método
        - polinomio: representación del polinomio
        - error_puntual: error en el punto eliminado
        - error_absoluto: error absoluto promedio
        - error_relativo: error relativo promedio
        - error_abs_max: error absoluto máximo
        - error_rel_max: error relativo máximo
        - mejor: True si es el mejor método
    """
    resultados = []
    
    # Asegurar que hay al menos 3 puntos
    if len(x) < 3:
        return [], "Se necesitan al menos 3 puntos para comparar los métodos eliminando uno."

    # Puntos de prueba para calcular errores (usar todos los puntos originales)
    x_test = x.copy()
    y_test = y.copy()
    
    metodos = [
        ("Vandermonde", lambda: vandermonde_interpolation(x, y, 1)),
        ("Newton Interpolante", lambda: newton_interpolation(x, y, 1)),
        ("Lagrange", lambda: lagrange_interpolation(x, y, 1)),
        ("Spline Lineal", lambda: spline_interpolation(x, y, 'lineal')),
        ("Spline Cúbico", lambda: spline_interpolation(x, y, 'cubico'))
    ]
    
    for metodo_nombre, metodo_func in metodos:
        try:
            # Ejecutar el método
            resultado = metodo_func()
            
            # Extraer información según el método
            if metodo_nombre in ["Vandermonde", "Newton Interpolante"]:
                if metodo_nombre == "Vandermonde":
                    coef, polinomio, x_eval, p_eval, error_msg = resultado
                else:
                    coef, polinomio, x_eval, p_eval, error_msg = resultado
            else:  # Lagrange y Splines
                polinomio, x_eval, p_eval, error_msg = resultado
            
            # Extraer el error puntual del mensaje
            import re
            match = re.search(r'error\s*=\s*([\d.]+)', error_msg)
            error_puntual = float(match.group(1)) if match else 0.0
            
            # Calcular errores en todos los puntos originales
            # Evaluar el polinomio en los puntos originales
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
                'error_msg': error_msg,
                'error_puntual': error_puntual,
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
                'error_msg': str(e),
                'error_puntual': float('inf'),
                'error_absoluto': float('inf'),
                'error_relativo': float('inf'),
                'error_abs_max': float('inf'),
                'error_rel_max': float('inf'),
                'mejor': False
            })
    
    # Determinar el mejor método (menor error puntual)
    if resultados:
        mejor_idx = min(range(len(resultados)), 
                       key=lambda i: resultados[i]['error_puntual'])
        resultados[mejor_idx]['mejor'] = True
    
    return resultados