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
    resultados = []
    
    # Asegurar que hay al menos 3 puntos
    if len(x) < 3:
        return [], "Se necesitan al menos 3 puntos para comparar los métodos eliminando uno."

    # Método Vandermonde
    try:
        _, polinomio_v, x_eval, p_eval, error_v = vandermonde_interpolation(x, y, 1)
        resultados.append(("Vandermonde", polinomio_v, error_v))
    except Exception as e:
        resultados.append(("Vandermonde", "Error", str(e)))

    # Método Newton
    try:
        _, polinomio_n, x_eval, p_eval, error_n = newton_interpolation(x, y, 1)
        resultados.append(("Newton", polinomio_n, error_n))
    except Exception as e:
        resultados.append(("Newton", "Error", str(e)))

    # Método Lagrange
    try:
        polinomio_l, x_eval, p_eval, error_l = lagrange_interpolation(x, y, 1)
        resultados.append(("Lagrange", polinomio_l, error_l))
    except Exception as e:
        resultados.append(("Lagrange", "Error", str(e)))

    # Spline Lineal
    try:
        polinomio_sp, x_eval, p_eval, error_sp = spline_interpolation(x, y, 'lineal')
        resultados.append(("Spline Lineal", polinomio_sp, error_sp))
    except Exception as e:
        resultados.append(("Spline Lineal", "Error", str(e)))

    # Spline Cúbico
    try:
        polinomio_spcub, x_eval, p_eval, error_spcub = spline_interpolation(x, y, 'cubico')
        resultados.append(("Spline Cúbico", polinomio_spcub, error_spcub))
    except Exception as e:
        resultados.append(("Spline Cúbico", "Error", str(e)))

    return resultados