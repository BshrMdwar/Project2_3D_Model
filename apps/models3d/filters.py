import django_filters
from .models import Model3D
from django.db.models import F

class Model3DFilter(django_filters.FilterSet):
    vertices_min = django_filters.NumberFilter(field_name='vertices',    lookup_expr='gte')
    vertices_max = django_filters.NumberFilter(field_name='vertices',    lookup_expr='lte')
    faces_min    = django_filters.NumberFilter(field_name='faces',       lookup_expr='gte')
    faces_max    = django_filters.NumberFilter(field_name='faces',       lookup_expr='lte')


    uploaded_after  = django_filters.DateTimeFilter(field_name='uploaded_at', lookup_expr='gte')
    uploaded_before = django_filters.DateTimeFilter(field_name='uploaded_at', lookup_expr='lte')

    class Meta:
        model  = Model3D
        fields = []