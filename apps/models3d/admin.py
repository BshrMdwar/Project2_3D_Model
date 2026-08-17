from django.contrib import admin
from .models import Model3D, Report



@admin.register(Report)
class ReportAdmin(admin.ModelAdmin):
    list_display  = ('id', 'model', 'reporter', 'status', 'created_at')
    list_filter   = ('status',)
    search_fields = ('model__id', 'reporter__email')
    list_editable = ('status',)



@admin.register(Model3D)
class Model3DAdmin(admin.ModelAdmin):
    list_display = (
        'id', 'ai_label', 'is_active',
        'downloads_count', 'views_count', 'uploaded_at',
    )
    list_filter  = ('is_active', 'ai_label')
    search_fields = ('id', 'ai_label')
    list_editable = ('is_active',)
    readonly_fields = ('uploaded_at', 'updated_at')
