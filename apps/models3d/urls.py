from django.urls import path
from .views import (
    # ── Public / User ──────────────────────────────────────
    # Model3DListView,
    # Model3DDetailView,
    # Model3DTopRatedView,
    # Model3DImportView,
    Model3DUploadView,
    # Model3DDownloadView,
    # Model3DDeleteView,
    # Model3DStatsView,
    # Model3DRecommendView,
    # ReportCreateView,
    # Model3DRendersListView,

    # ── Admin ──────────────────────────────────────────────
#     AdminModel3DListView,
#     AdminModel3DUpdateView,
#     AdminModel3DSoftDeleteView,
#     AdminReportListView,
#     AdminReportUpdateView,
#     AdminDashboardStatsView,
#     TagListCreateView,
#     TagDetailView,
#     CategoryListCreateView,
#     CategoryDetailView,
)

urlpatterns = [

    # ── Upload & Import ────────────────────────────────────
    path('upload-user/',   Model3DUploadView.as_view(), name='model-upload-user'),
    # path('import/',        Model3DImportView.as_view(),     name='model-import'),

    # # ── Read ───────────────────────────────────────────────
    # path('',   Model3DListView.as_view(),       name='model-list'),
    # path('top-rated/',     Model3DTopRatedView.as_view(),   name='model-top-rated'),
    # path('stats/',         Model3DStatsView.as_view(),      name='model-stats'),
    # # ── Report ─────────────────────────────────────────────
    # path('report/',        ReportCreateView.as_view(),      name='model-report-create'),

    # # ── Per-object ─────────────────────────────────────────
    # path('<str:id>/',              Model3DDetailView.as_view(),   name='model-detail'),
    # path('<str:id>/delete/',       Model3DDeleteView.as_view(),   name='model-delete'),
    # path('<str:id>/recommend/',    Model3DRecommendView.as_view(), name='model-recommend'),
    # path('<str:id>/renders/', Model3DRendersListView.as_view(), name='model-renders-list'),

    # # ── Admin Namespace (/api/admin/…) ─────────────────────
    # path('admin/models/',                  AdminModel3DListView.as_view(),     name='admin-model-list'),
    # path('admin/models/<str:id>/update/',  AdminModel3DUpdateView.as_view(),   name='admin-model-update'),
    # path('admin/models/<str:id>/hide/',    AdminModel3DSoftDeleteView.as_view(), name='admin-model-hide'),
    # path('admin/reports/',                 AdminReportListView.as_view(),       name='admin-report-list'),
    # path('admin/reports/<int:pk>/',        AdminReportUpdateView.as_view(),     name='admin-report-update'),
    # path('admin/dashboard-stats/',         AdminDashboardStatsView.as_view(),   name='admin-dashboard-stats'),
    # path('admin/tags/',                    TagListCreateView.as_view(),          name='admin-tag-list'),
    # path('admin/tags/<int:pk>/',           TagDetailView.as_view(),             name='admin-tag-detail'),
    # path('admin/categories/',              CategoryListCreateView.as_view(),     name='admin-category-list'),
    # path('admin/categories/<int:pk>/',     CategoryDetailView.as_view(),         name='admin-category-detail'),
    # path('<str:id>/download/',     Model3DDownloadView.as_view(), name='admin-model-download'),
]
