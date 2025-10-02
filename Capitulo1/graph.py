import matplotlib
matplotlib.use('Agg')  # Usa un backend no interactivo apto para servidores web
import matplotlib.pyplot as plt
import numpy as np
import io
import base64
from sympy import sympify, lambdify

def graficar_funcion(funcion_str, xi, xs, raiz):
    x = np.linspace(xi, xs, 400)
    try:
        f_expr = sympify(funcion_str)  # Convierte la cadena a una expresión simbólica
        f = lambdify('x', f_expr, 'numpy')  # Crea una función evaluable con numpy
    except Exception as e:
        return None  # Error en la función


    # Generar valores de x y calcular los valores de f(x)
    x = np.linspace(xi, xs, 400)
    try:
        y = f(x)  # Evaluar f(x)
    except Exception as e:
        print(f"Error al evaluar la función: {e}")
        return None  # Error al evaluar la función

    plt.figure(figsize=(8, 4))
    plt.plot(x, y, label=f"f(x) = {funcion_str}", color='cyan')
    plt.axhline(0, color='white', linestyle='--', linewidth=1)
    plt.scatter([raiz], [f(raiz)], color='red', zorder=5, label=f"Raíz aproximada ≈ {raiz:.6f}")

    # Ejes en blanco
    ax = plt.gca()
    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['right'].set_color('white')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.title.set_color('white')
    ax.legend(facecolor='#2C2C2C', edgecolor='white', labelcolor='white')

    plt.title("Gráfica de f(x) con raíz marcada")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.grid(True, color='gray', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.style.use('dark_background')


    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    imagen_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close()
    return imagen_base64