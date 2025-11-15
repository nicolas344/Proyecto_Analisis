import matplotlib.pyplot as plt
from io import BytesIO
import base64

def graficar(x, y, xpol, p):
    plt.figure()
    plt.plot(x, y, 'ro', label='Puntos')
    plt.plot(xpol, p, 'b-', label='Polinomio interpolante')
    plt.grid(True)
    plt.legend()
    plt.title('Interpolación')
    
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    plt.close()
    return base64.b64encode(buffer.getvalue()).decode()