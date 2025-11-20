"""
Módulo para generar informes en PDF de los métodos numéricos del Capítulo 1
"""
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak, Image
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from io import BytesIO
from datetime import datetime
import base64


def generar_informe_pdf(metodo_nombre, funcion, parametros, tabla, resultado, mensaje, 
                        tipo_error, resultados_comparativos=None, grafico_base64=None):
    """
    Genera un informe detallado en PDF de la ejecución del método numérico.
    
    Args:
        metodo_nombre: Nombre del método ejecutado
        funcion: String de la función evaluada
        parametros: Dict con los parámetros usados (xi, xs, x0, tol, niter, etc.)
        tabla: Lista de tuplas con los resultados de cada iteración
        resultado: Valor de la raíz encontrada
        mensaje: Mensaje de estado final
        tipo_error: Tipo de error usado ('absoluto', 'relativo', 'condicion')
        resultados_comparativos: Lista de resultados de comparación (opcional)
        grafico_base64: Imagen del gráfico en base64 (opcional)
    
    Returns:
        BytesIO: Buffer con el PDF generado
    """
    # Crear buffer en memoria
    buffer = BytesIO()
    
    # Crear el documento PDF
    doc = SimpleDocTemplate(buffer, pagesize=letter,
                           rightMargin=72, leftMargin=72,
                           topMargin=72, bottomMargin=18)
    
    # Contenedor de elementos del PDF
    elements = []
    
    # Estilos
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=18,
        textColor=colors.HexColor('#2c3e50'),
        spaceAfter=30,
        alignment=TA_CENTER
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#34495e'),
        spaceAfter=12,
        spaceBefore=12
    )
    
    normal_style = styles['Normal']
    
    # ==== TÍTULO ====
    titulo = Paragraph(f"<b>Informe de Ejecución - {metodo_nombre}</b>", title_style)
    elements.append(titulo)
    elements.append(Spacer(1, 12))
    
    # ==== INFORMACIÓN GENERAL ====
    fecha_hora = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    info_general = [
        ["<b>Fecha de ejecución:</b>", fecha_hora],
        ["<b>Método utilizado:</b>", metodo_nombre],
        ["<b>Función evaluada:</b>", f"f(x) = {funcion}"],
        ["<b>Tipo de error:</b>", _obtener_nombre_tipo_error(tipo_error)],
    ]
    
    # Agregar parámetros específicos
    for key, value in parametros.items():
        info_general.append([f"<b>{key}:</b>", str(value)])
    
    tabla_info = Table(info_general, colWidths=[2.5*inch, 4*inch])
    tabla_info.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#ecf0f1')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
    ]))
    
    elements.append(tabla_info)
    elements.append(Spacer(1, 20))
    
    # ==== RESULTADO ====
    elements.append(Paragraph("<b>Resultado de la Ejecución</b>", heading_style))
    
    if resultado is not None:
        resultado_texto = f"""
        <b>Raíz aproximada:</b> {resultado}<br/>
        <b>Estado:</b> {mensaje}<br/>
        <b>Iteraciones realizadas:</b> {len(tabla) - 1 if tabla else 0}
        """
    else:
        resultado_texto = f"<b>Estado:</b> {mensaje}"
    
    elements.append(Paragraph(resultado_texto, normal_style))
    elements.append(Spacer(1, 20))
    
    # ==== TABLA DE ITERACIONES ====
    if tabla and len(tabla) > 0:
        elements.append(Paragraph("<b>Tabla de Iteraciones</b>", heading_style))
        elements.append(Spacer(1, 12))
        
        # Preparar datos de la tabla
        # Detectar headers según el método
        if len(tabla[0]) == 4:  # (iter, x, f(x), error)
            headers = ['Iteración', 'x', 'f(x)', 'Error']
        else:
            headers = ['Iteración', 'Valores', 'f(x)', 'Error']
        
        datos_tabla = [headers]
        
        # Limitar a las primeras y últimas iteraciones si es muy grande
        max_rows = 20
        if len(tabla) > max_rows:
            # Primeras 10
            for row in tabla[:10]:
                datos_tabla.append([_format_value(v) for v in row])
            # Fila de separación
            datos_tabla.append(['...', '...', '...', '...'])
            # Últimas 9
            for row in tabla[-9:]:
                datos_tabla.append([_format_value(v) for v in row])
        else:
            for row in tabla:
                datos_tabla.append([_format_value(v) for v in row])
        
        # Crear tabla
        tabla_iteraciones = Table(datos_tabla, colWidths=[1*inch, 1.8*inch, 1.8*inch, 1.5*inch])
        tabla_iteraciones.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498db')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('FONTSIZE', (0, 1), (-1, -1), 8),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8f9fa')]),
        ]))
        
        elements.append(tabla_iteraciones)
        elements.append(Spacer(1, 20))
    
    # ==== GRÁFICO ====
    if grafico_base64:
        try:
            elements.append(Paragraph("<b>Gráfico de la Función</b>", heading_style))
            elements.append(Spacer(1, 12))
            
            # Decodificar imagen base64
            img_data = base64.b64decode(grafico_base64)
            img_buffer = BytesIO(img_data)
            
            # Agregar imagen
            img = Image(img_buffer, width=5*inch, height=3.5*inch)
            elements.append(img)
            elements.append(Spacer(1, 20))
        except Exception as e:
            print(f"Error al incluir gráfico en PDF: {e}")
    
    # ==== COMPARACIÓN CON OTROS MÉTODOS ====
    if resultados_comparativos and len(resultados_comparativos) > 0:
        elements.append(PageBreak())
        elements.append(Paragraph("<b>Comparación con Otros Métodos</b>", heading_style))
        elements.append(Spacer(1, 12))
        
        # Explicación del tipo de error
        explicacion_error = _obtener_explicacion_tipo_error(tipo_error)
        elements.append(Paragraph(explicacion_error, normal_style))
        elements.append(Spacer(1, 12))
        
        # Preparar tabla comparativa
        headers_comp = ['Método', 'Raíz (xs)', 'Iteraciones', 'Error Final', 'Estado']
        datos_comp = [headers_comp]
        
        mejor_metodo = None
        for res in resultados_comparativos:
            estado = ''
            if res.get('mejor'):
                estado = '✓ MEJOR'
                mejor_metodo = res['metodo']
            elif res.get('info'):
                estado = res['info'][:30]  # Truncar si es muy largo
            
            datos_comp.append([
                res['metodo'],
                _format_value(res['xs']),
                _format_value(res['n']),
                _format_value(res['error']),
                estado
            ])
        
        tabla_comp = Table(datos_comp, colWidths=[1.5*inch, 1.3*inch, 1*inch, 1.2*inch, 1.5*inch])
        
        # Estilo base
        style_list = [
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2ecc71')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8f9fa')]),
        ]
        
        # Resaltar el mejor método
        if mejor_metodo:
            for i, res in enumerate(resultados_comparativos, 1):
                if res.get('mejor'):
                    style_list.append(('BACKGROUND', (0, i), (-1, i), colors.HexColor('#d4edda')))
                    style_list.append(('FONTNAME', (0, i), (-1, i), 'Helvetica-Bold'))
        
        tabla_comp.setStyle(TableStyle(style_list))
        elements.append(tabla_comp)
        elements.append(Spacer(1, 12))
        
        # Conclusión
        if mejor_metodo:
            conclusion = f"""
            <b>Conclusión:</b> El método <b>{mejor_metodo}</b> resultó ser el más eficiente para 
            esta función con los parámetros dados, convergiendo en el menor número de iteraciones 
            con el error final más bajo usando {_obtener_nombre_tipo_error(tipo_error).lower()}.
            """
            elements.append(Paragraph(conclusion, normal_style))
    
    # ==== PIE DE PÁGINA ====
    elements.append(Spacer(1, 30))
    footer_text = f"""
    <i>Informe generado automáticamente el {fecha_hora}<br/>
    Sistema de Análisis Numérico - Capítulo 1: Métodos de Solución de Ecuaciones</i>
    """
    footer_style = ParagraphStyle('Footer', parent=normal_style, fontSize=8, 
                                  textColor=colors.grey, alignment=TA_CENTER)
    elements.append(Paragraph(footer_text, footer_style))
    
    # Construir PDF
    doc.build(elements)
    
    # Retornar buffer
    buffer.seek(0)
    return buffer


def _format_value(value):
    """Formatea valores para mostrar en el PDF"""
    if value is None or value == "N/A":
        return "N/A"
    if isinstance(value, (int, float)):
        if abs(value) < 1e-10 and value != 0:
            return f"{value:.6e}"
        elif isinstance(value, int):
            return str(value)
        else:
            return f"{value:.8f}"
    return str(value)


def _obtener_nombre_tipo_error(tipo_error):
    """Retorna el nombre completo del tipo de error"""
    nombres = {
        'absoluto': 'Error Absoluto',
        'relativo': 'Error Relativo',
        'condicion': 'Error de Condición'
    }
    return nombres.get(tipo_error, 'Error Absoluto')


def _obtener_explicacion_tipo_error(tipo_error):
    """Retorna una explicación del tipo de error usado"""
    explicaciones = {
        'absoluto': """
        <b>Error Absoluto:</b> Se calcula como |x_n - x_(n-1)|, mide la diferencia absoluta 
        entre dos aproximaciones consecutivas. Es útil cuando se conoce la escala de la solución.
        """,
        'relativo': """
        <b>Error Relativo:</b> Se calcula como |x_n - x_(n-1)|/|x_n|, mide el error como 
        porcentaje del valor actual. Es útil cuando se desea precisión relativa independiente 
        de la escala.
        """,
        'condicion': """
        <b>Error de Condición:</b> Se calcula como |f(x_n)|, mide qué tan cerca está la función 
        de cero. Es útil para verificar que realmente se encontró una raíz.
        """
    }
    return explicaciones.get(tipo_error, explicaciones['absoluto'])
