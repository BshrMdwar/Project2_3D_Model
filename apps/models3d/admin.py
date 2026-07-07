from django.contrib import admin
from .models import Model3D, Report, Tag, Category, RenderImage


@admin.register(Tag)
class TagAdmin(admin.ModelAdmin):
    list_display = ('id', 'name')
    search_fields = ('name',)


@admin.register(Category)
class CategoryAdmin(admin.ModelAdmin):
    list_display = ('id', 'name')
    search_fields = ('name',)


@admin.register(RenderImage)
class RenderImageAdmin(admin.ModelAdmin):
    list_display = ('id', 'model', 'angle', 'created_at')
    list_filter  = ('angle',)


@admin.register(Report)
class ReportAdmin(admin.ModelAdmin):
    list_display  = ('id', 'model', 'reporter', 'status', 'created_at')
    list_filter   = ('status',)
    search_fields = ('model__id', 'reporter__email')
    list_editable = ('status',)


class RenderImageInline(admin.TabularInline):
    model  = RenderImage
    extra  = 0
    fields = ('image', 'angle')


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
    inlines = [RenderImageInline]
    readonly_fields = ('uploaded_at', 'updated_at')
