from django import template

register = template.Library()

@register.filter
def index(list_obj, i):
    """Permite acceder a list_obj[i] desde el template."""
    try:
        return list_obj[i]
    except:
        return ''