from django.contrib import admin
from .models import Model3D, Report, Tag, Category


@admin.register(Tag)
class TagAdmin(admin.ModelAdmin):
    list_display = ('id', 'name')
    search_fields = ('name',)


@admin.register(Category)
class CategoryAdmin(admin.ModelAdmin):
    list_display = ('id', 'name')
    search_fields = ('name',)




@admin.register(Report)
class ReportAdmin(admin.ModelAdmin):
    list_display  = ('id', 'model', 'reporter', 'status', 'created_at')
    list_filter   = ('status',)
    search_fields = ('model__id', 'reporter__email')
    list_editable = ('status',)



@admin.register(Model3D)
class Model3DAdmin(admin.ModelAdmin):
    list_display = (
        'id', 'ai_label', 'category', 'is_active',
        'downloads_count', 'usage_count', 'views_count', 'uploaded_at',
    )
    list_filter  = ('is_active', 'ai_label', 'category', 'is_manifold')
    search_fields = ('id', 'ai_label')
    list_editable = ('is_active',)
    filter_horizontal = ('tags',)
    readonly_fields = ('uploaded_at', 'updated_at')
