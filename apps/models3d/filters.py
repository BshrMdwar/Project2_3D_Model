import django_filters
from .models import Model3D
from django.db.models import F

class Model3DFilter(django_filters.FilterSet):
    # ── Mesh ──────────────────────────────────────────
    vertices_min = django_filters.NumberFilter(field_name='vertices',    lookup_expr='gte')
    vertices_max = django_filters.NumberFilter(field_name='vertices',    lookup_expr='lte')
    faces_min    = django_filters.NumberFilter(field_name='faces',       lookup_expr='gte')
    faces_max    = django_filters.NumberFilter(field_name='faces',       lookup_expr='lte')

    # ── Dimensions (فلاتر مخصصة تحسب البعد بناءً على مصفوفة الـ Bounding Box) ──
    # المصفوفة مرتبة كالتالي: [0: min_x, 1: min_y, 2: min_z, 3: max_x, 4: max_y, 5: max_z]
    width_min  = django_filters.NumberFilter(method='filter_width_min')
    width_max  = django_filters.NumberFilter(method='filter_width_max')
    depth_min  = django_filters.NumberFilter(method='filter_depth_min')
    depth_max  = django_filters.NumberFilter(method='filter_depth_max')
    height_min = django_filters.NumberFilter(method='filter_height_min')
    height_max = django_filters.NumberFilter(method='filter_height_max')

    # ── Quality ──────────────────────────────────────
    is_manifold     = django_filters.BooleanFilter(field_name='is_manifold')
    stability_min   = django_filters.NumberFilter(field_name='stability_score', lookup_expr='gte')
    elongation_max  = django_filters.NumberFilter(field_name='elongation',      lookup_expr='lte')

    # ── Shape ────────────────────────────────────────
    symmetry_axis   = django_filters.CharFilter(field_name='symmetry_axis', lookup_expr='iexact')

    # ── Date ─────────────────────────────────────────
    uploaded_after  = django_filters.DateTimeFilter(field_name='uploaded_at', lookup_expr='gte')
    uploaded_before = django_filters.DateTimeFilter(field_name='uploaded_at', lookup_expr='lte')

    class Meta:
        model  = Model3D
        fields = []

    # ── دالات المعالجة الديناميكية للأبعاد (PostgreSQL Array Indexing) ──
    # نقوم بطرح العنصر Max من العنصر Min لكل محور داخل قاعدة البيانات مباشرة

    def filter_width_min(self, queryset, name, value):
        # width = max_x (index 3) - min_x (index 0)
        return queryset.annotate(
            calculated_width=F('bounding_box__4') - F('bounding_box__1')
        ).filter(calculated_width__gte=value)

    def filter_width_max(self, queryset, name, value):
        return queryset.annotate(
            calculated_width=F('bounding_box__4') - F('bounding_box__1')
        ).filter(calculated_width__lte=value)

    def filter_depth_min(self, queryset, name, value):
        # depth = max_y (index 4) - min_y (index 1)
        return queryset.annotate(
            calculated_depth=F('bounding_box__5') - F('bounding_box__2')
        ).filter(calculated_depth__gte=value)

    def filter_depth_max(self, queryset, name, value):
        return queryset.annotate(
            calculated_depth=F('bounding_box__5') - F('bounding_box__2')
        ).filter(calculated_depth__lte=value)

    def filter_height_min(self, queryset, name, value):
        # height = max_z (index 5) - min_z (index 2)
        return queryset.annotate(
            calculated_height=F('bounding_box__6') - F('bounding_box__3')
        ).filter(calculated_height__gte=value)

    def filter_height_max(self, queryset, name, value):
        return queryset.annotate(
            calculated_height=F('bounding_box__6') - F('bounding_box__3')
        ).filter(calculated_height__lte=value)