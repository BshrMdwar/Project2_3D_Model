from __future__ import annotations
import os
import uuid
import subprocess
import json
import shutil
import httpx
import re
from django.conf import settings
from django.core.files import File
from django.http import JsonResponse
from django.views import View
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator

from rest_framework.parsers import FormParser, MultiPartParser
from rest_framework import permissions, status
# 1️⃣ الاستيراد الصحيح للدوال الرياضية والـ Expressions
from django.db.models import Case, When, Value,  ExpressionWrapper, IntegerField, F, Q
# لحماية حقول الـ Null أثناء الجمع
from django.db.models.functions import Coalesce
import requests
from django.db.models import Count, Avg, Q, Sum
from django.db.models.functions import TruncMonth
from rest_framework import generics, status, permissions
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser, FormParser
from django_filters.rest_framework import DjangoFilterBackend
from rest_framework.filters import SearchFilter, OrderingFilter
from itertools import chain

from django.shortcuts import get_object_or_404

from .models import Model3D, Report
from .serializers import (
    Model3DListSerializer,
    Model3DRecommendSerializer,
    ReportCreateSerializer,
    ReportDetailSerializer,
)
from .filters import Model3DFilter

TEMP_FOLDER = "./assets/temp"
PUBLIC_ROOT = "./assets/public"
BLENDER_EXECUTABLE = ["docker", "compose", "exec", "blender", "blender"]
RENDER_SCRIPT = "/scripts/renderer.py"
ALLOWED_EXTENSIONS = {".glb", ".gltf"}


class IsAdminUser(permissions.BasePermission):
    def has_permission(self, request, view):
        return (
            request.user and
            request.user.is_authenticated and
            (request.user.is_staff or getattr(request.user, 'role', '') == 'admin')
        )

class Model3DListView(generics.ListAPIView):
    """GET /api/models/getAllModels/ — قائمة الموديلات النشطة"""
    serializer_class = Model3DListSerializer
    permission_classes = [permissions.AllowAny]
    filterset_class = Model3DFilter
    filter_backends = [DjangoFilterBackend, SearchFilter, OrderingFilter]
    search_fields = ['id', 'ai_label']
    ordering_fields = [
        'uploaded_at', 'vertices', 'faces',
        'stability_score', 'downloads_count', 'usage_count',
    ]
    ordering = ['-uploaded_at']

    def get_queryset(self):
        return Model3D.objects.filter(is_active=True)


class Model3DDetailView(generics.RetrieveAPIView):
    """GET /api/models/<id>/ — تفاصيل موديل + يزيد عداد المشاهدات"""
    serializer_class = Model3DListSerializer
    permission_classes = [permissions.AllowAny]
    lookup_field = 'id'

    def get_queryset(self):
        return Model3D.objects.filter(is_active=True)

    def retrieve(self, request, *args, **kwargs):
        instance = self.get_object()
        instance.increment_views()          # عداد المشاهدات
        instance.refresh_from_db()
        serializer = self.get_serializer(instance)
        return Response(serializer.data)


class Model3DTopRatedView(generics.ListAPIView):
    """GET /api/models/top-rated/ — الموديلات الأعلى تقييماً (downloads+usage)"""
    serializer_class = Model3DListSerializer
    permission_classes = [permissions.AllowAny]

    def get_queryset(self):
        from django.db.models import ExpressionWrapper, IntegerField, F
        return (
            Model3D.objects
            .filter(is_active=True)
            .annotate(
                rating=ExpressionWrapper(
                    F('downloads_count') + F('usage_count'),
                    output_field=IntegerField()
                )
            )
            .order_by('-rating')[:20]
        )

    def _apply_geometry(self, model, geometry_path):
        import json
        if not os.path.isfile(geometry_path):
            return
        with open(geometry_path) as f:
            data = json.load(f)

        dims = data.get("dimensions", {})
        model.vertices = data.get("_debug_raw", {}).get("edges")
        model.edges = data.get("_debug_raw", {}).get("edges")


# class Model3DDownloadView(APIView):
#     """
#     POST /api/models/<id>/download/
#     يسجّل تحميلاً ويعيد URL أو البيانات اللازمة للتحميل.
#     """
#     permission_classes = [permissions.IsAuthenticated]

#     def post(self, request, id, *args, **kwargs):
#         # 1. جلب الموديل والتأكد أنه نشط
#         try:
#             model = Model3D.objects.get(id=id, is_active=True)
#         except Model3D.DoesNotExist:
#             return Response({'error': 'Model not found'}, status=status.HTTP_404_NOT_FOUND)

#         # 2. التأكد من أن حقل الملف ليس فارغاً في قاعدة البيانات
#         if not model.model_file:
#             return Response({'error': 'No file associated with this model record'}, status=status.HTTP_400_BAD_REQUEST)

#         # 3. تسجيل عملية التحميل وزيادة العداد عبر دالتك الخاصة
#         model.increment_downloads()

#         # 4. بناء الرابط المطلق الكامل للملف (مثال: http://127.0.0.1:8000/media/models/...)
#         file_url = request.build_absolute_uri(model.model_file.url)

#         return Response({
#             'status': 'download_registered',
#             'model_id': id,
#             'file_url': file_url,
#             'message': 'Use this link to download the file directly to your local storage.'
#         }, status=status.HTTP_200_OK)


class Model3DStatsView(APIView):
    """GET /api/models/stats/ — إحصائيات عامة (متاحة للجميع)"""
    permission_classes = [permissions.AllowAny]

    def get(self, request):
        qs = Model3D.objects.filter(is_active=True)
        stats = qs.aggregate(
            total_models=Count('id'),
            avg_vertices=Avg('vertices'),
            avg_faces=Avg('faces'),
            total_downloads=Sum('downloads_count'),
            total_views=Sum('views_count'),
        )
        return Response(stats)


class AdminDashboardStatsView(APIView):
    permission_classes = [IsAdminUser]

    def get(self, request):
        qs = Model3D.objects.all()
        rep_qs = Report.objects.all()

        # إحصائيات أساسية
        total_models = qs.count()
        active_models = qs.filter(is_active=True).count()
        inactive_models = total_models - active_models

        # أكثر الموديلات تحميلاً
        top_downloads = (
            qs.filter(is_active=True)
            .order_by('-downloads_count')[:5]
            .values('id', 'ai_label', 'downloads_count', 'usage_count')
        )

        # توزيع التصنيفات AI
        ai_distribution = (
            qs.filter(is_active=True)
            .values('ai_label')
            .annotate(count=Count('id'))
            .order_by('-count')
        )

        # البلاغات
        reports_summary = rep_qs.values('status').annotate(count=Count('id'))

        # موديلات مضافة شهرياً (آخر 6 أشهر)
        monthly_uploads = (
            qs.annotate(month=TruncMonth('uploaded_at'))
            .values('month')
            .annotate(count=Count('id'))
            .order_by('-month')[:6]
        )

        return Response({
            'models': {
                'total':    total_models,
                'active':   active_models,
                'inactive': inactive_models,
            },
            'top_downloads':    list(top_downloads),
            'ai_distribution':  list(ai_distribution),
            'reports':          list(reports_summary),
            'monthly_uploads':  list(monthly_uploads),
        })


# ════════════════════════════════════════════════════════════
#  Admin — إدارة الموديلات
# ════════════════════════════════════════════════════════════

class AdminModel3DListView(generics.ListAPIView):
    """GET /api/admin/models/ — قائمة جميع الموديلات (نشطة + مخفية)"""
    queryset = Model3D.objects.all()
    serializer_class = Model3DListSerializer
    permission_classes = [IsAdminUser]
    filter_backends = [DjangoFilterBackend, SearchFilter, OrderingFilter]
    filterset_class = Model3DFilter
    search_fields = ['id', 'ai_label']
    ordering_fields = ['uploaded_at', 'downloads_count', 'usage_count']
    ordering = ['-uploaded_at']


class AdminModel3DUpdateView(generics.UpdateAPIView):
    """PATCH /api/admin/models/<id>/update/ — تعديل التصنيف والوسوم والإخفاء"""
    queryset = Model3D.objects.all()
    permission_classes = [IsAdminUser]
    lookup_field = 'id'
    http_method_names = ['patch']


class Model3DDeleteView(generics.DestroyAPIView):
    """DELETE /api/models/<id>/delete/ — حذف نهائي (مستخدم مصادَق)"""
    queryset = Model3D.objects.all()
    permission_classes = [permissions.IsAuthenticated]
    lookup_field = 'id'


class AdminModel3DSoftDeleteView(APIView):
    """PATCH /api/admin/models/<id>/hide/ — إخفاء / إظهار (Soft Delete)"""
    permission_classes = [IsAdminUser]

    def patch(self, request, id):
        try:
            model = Model3D.objects.get(id=id)
        except Model3D.DoesNotExist:
            return Response({'error': 'Not found'}, status=status.HTTP_404_NOT_FOUND)

        model.is_active = not model.is_active
        model.save(update_fields=['is_active'])
        return Response({'id': id, 'is_active': model.is_active})


# ════════════════════════════════════════════════════════════
#  Report — نظام الإبلاغ
# ════════════════════════════════════════════════════════════

class ReportCreateView(generics.CreateAPIView):
    """POST /api/models/report/ — تقديم بلاغ (مستخدم مسجَّل)"""
    serializer_class = ReportCreateSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_serializer_context(self):
        ctx = super().get_serializer_context()
        ctx['request'] = self.request
        return ctx


class AdminReportListView(generics.ListAPIView):
    """GET /api/admin/reports/ — قائمة البلاغات (آدمن)"""
    queryset = Report.objects.select_related('model', 'reporter')
    serializer_class = ReportDetailSerializer
    permission_classes = [IsAdminUser]
    filter_backends = [DjangoFilterBackend]
    filterset_fields = ['status', 'model']


class AdminReportUpdateView(generics.UpdateAPIView):
    """PATCH /api/admin/reports/<pk>/resolve/ — تحديث حالة البلاغ"""
    queryset = Report.objects.all()
    serializer_class = ReportDetailSerializer
    permission_classes = [IsAdminUser]
    http_method_names = ['patch']

    def partial_update(self, request, *args, **kwargs):
        kwargs['partial'] = True
        return self.update(request, *args, **kwargs)


# ════════════════════════════════════════════════════════════
#  Recommendation — اقتراحات ذكية
# ════════════════════════════════════════════════════════════

class Model3DRecommendView(APIView):
    """
    GET /api/models/<id>/recommend/
    يقترح موديلات مشابهة بناءً على:
      1. نفس التصنيف AI (ai_label)
      2. تقاطع عدد الأوجه (faces) بفارق ≤ 20%
      3. نفس الفئة (category)
    يُرجع أقصى 10 نتائج مرتبة بـ rating.
    """
    permission_classes = [permissions.AllowAny]

    def get(self, request, id):
        try:
            base = Model3D.objects.get(id=id, is_active=True)
        except Model3D.DoesNotExist:
            return Response({'error': 'Model not found'}, status=status.HTTP_404_NOT_FOUND)

        # 🎯 تم حذف سطر الاستيراد القديم المسبب للخطأ من هنا والاعتماد على الاستيراد الأعلى

        filters = Q(is_active=True) & ~Q(id=id)

        # شرط التشابه: ai_label أو category أو faces متقاربة
        similarity_q = Q()
        if base.ai_label:
            similarity_q |= Q(ai_label=base.ai_label)
        if base.category_id:
            similarity_q |= Q(category_id=base.category_id)
        if base.faces:
            face_margin = int(base.faces * 0.2) or 1
            similarity_q |= Q(
                faces__gte=base.faces - face_margin,
                faces__lte=base.faces + face_margin,
            )

        # 2️⃣ بناء الاستعلام المحمي من الـ Null والمرتب برمجياً بشكل سليم
        recommendations = (
            Model3D.objects
            .filter(filters & similarity_q)
            .annotate(
                rating=ExpressionWrapper(
                    Coalesce(F('downloads_count'), 0) + Coalesce(F('usage_count'), 0),
                    output_field=IntegerField()
                )
            )
            .order_by('-rating')[:10]
        )

        serializer = Model3DRecommendSerializer(recommendations, many=True)
        return Response({
            'base_model':       id,
            'recommendations':  serializer.data,
        })



class Model3DRendersListView(APIView):
    """
    GET /api/models/<id>/renders/
    جلب كافة الصور المُرندرة الخاصة بموديل معين
    """
    permission_classes = [permissions.AllowAny]

    def get(self, request, id):
        # try:
        #     model_instance = Model3D.objects.get(id=id, is_active=True)
        # except Model3D.DoesNotExist:
        #     return Response({'error': 'Model not found'}, status=status.HTTP_404_NOT_FOUND)

        # # 🚨 التعديل هنا: استبدال model_3d بـ model بناءً على بنية قاعدة البيانات عندك
        # renders = RenderImage.objects.filter(model=model_instance)

        # # تجميع روابط الصور بشكل مطلق ومباشر
        # urls = [request.build_absolute_uri(img.image.url) for img in renders if img.image]

        return Response({
            "model_id": id,
            # "renders_count": len(urls),
            # "images": urls
        }, status=status.HTTP_200_OK)

@method_decorator(csrf_exempt, name="dispatch")
class Model3DUploadView(View):

    def post(self, request):
        uploaded_model = request.FILES.get("model")

        if not uploaded_model:
            return JsonResponse(
                {"error": "No file provided under 'model'."},
                status=400,
            )

        ext = os.path.splitext(uploaded_model.name)[1].lower()

        if ext not in ALLOWED_EXTENSIONS:
            return JsonResponse(
                {
                    "error": (
                        f"Unsupported extension '{ext}'. "
                        "Use .glb or .gltf."
                    )
                },
                status=400,
            )

        model_id = uuid.uuid4().hex

        os.makedirs(TEMP_FOLDER, exist_ok=True)

        temp_filename = f"{model_id}{ext}"
        temp_path = os.path.join(TEMP_FOLDER, temp_filename)

        with open(temp_path, "wb") as dest:
            for chunk in uploaded_model.chunks():
                dest.write(chunk)

        model = Model3D.objects.create(
            id=model_id,
            title = request.POST.get("title"),
            description = request.POST.get("description") or None
        )

        cmd = [
            *BLENDER_EXECUTABLE,
            "-b",
            "--python",
            RENDER_SCRIPT,
            "--",
            "--model",
            temp_filename,
            "--uid",
            model_id,
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=900,
                check=True,
            )

        except subprocess.CalledProcessError as exc:
            model.is_active = False
            model.save(update_fields=["is_active"])

            return JsonResponse(
                {
                    "error": "Render failed.",
                    "returncode": exc.returncode,
                    "stderr": exc.stderr[-4000:],
                },
                status=500,
            )

        except subprocess.TimeoutExpired:
            model.is_active = False
            model.save(update_fields=["is_active"])

            return JsonResponse(
                {"error": "Render timed out."},
                status=504,
            )
        geometry_path = os.path.join(
            PUBLIC_ROOT,
            model_id,
            "geometry",
            "geometry.json",
        )

        self._apply_geometry(model, geometry_path)

        model_banner = request.FILES.get("banner")

        banner_dir = os.path.join(
            PUBLIC_ROOT,
            model_id
        )

        if model_banner:
            os.makedirs(banner_dir, exist_ok=True)

            banner_path = os.path.join(
                banner_dir,
                'banner.' + model_banner.name.rsplit('.', 1)[-1],
            )

            with open(banner_path, "wb") as dest:
                for chunk in model_banner.chunks():
                    dest.write(chunk)
            model.banner_url = "banner" + os.path.splitext(model_banner.name)[1].lower()

        else:
            renders_dir = os.path.join(
                PUBLIC_ROOT,
                model_id,
                "renders",
            )

            preferred_path = os.path.join(
                renders_dir,
                "front_left_high.png",
            )

            if os.path.isfile(preferred_path):
                os.makedirs(banner_dir, exist_ok=True)

                banner_path = os.path.join(
                    banner_dir,
                    "banner.png",
                )

                shutil.copy2(
                    preferred_path,
                    banner_path,
                )
            model.banner_url = "banner.png"


        model.model_url = "model" + ext
        model.save()

        # ---------------------------------------------------------
        # CALL FASTAPI PREDICTION SERVICE
        # ---------------------------------------------------------

        prediction = self._run_prediction(model_id)

        if prediction is None:
            return JsonResponse(
                {
                    "id": model.id,
                    "status": "rendered",
                    "prediction_status": "failed",
                    "renders_dir": os.path.join(
                        PUBLIC_ROOT,
                        model_id,
                        "renders",
                    ),
                },
                status=502,
            )
        
        self._apply_prediction(model, prediction)
        model.save()
        
        # ---------------------------------------------------------
        # Final response
        # ---------------------------------------------------------

        return JsonResponse(
            {
                "model" : Model3DListSerializer(model).data,
                "status": "rendered",
                "prediction_status": "completed",
                "prediction": prediction,
                "renders_dir": os.path.join(
                    PUBLIC_ROOT,
                    model_id,
                    "renders",
                ),
            }
        )

    def _run_prediction(self, model_id):
        
        prediction_url = (
            f"{settings.PREDICTION_API_URL.rstrip('/')}/predict"
        )

        try:
            response = httpx.post(
                prediction_url,
                json={
                    "model_id": model_id,
                },
                timeout=900.0,
            )

            response.raise_for_status()

            return response.json()

        except httpx.TimeoutException:
            print(
                f"Prediction service timed out for model {model_id}"
            )
            return None

        except httpx.HTTPStatusError as exc:
            print(
                f"Prediction service returned "
                f"{exc.response.status_code}: "
                f"{exc.response.text}"
            )
            return None

        except httpx.RequestError as exc:
            print(
                f"Could not connect to prediction service: {exc}"
            )
            return None

        except ValueError:
            print(
                f"Prediction service returned invalid JSON "
                f"for model {model_id}"
            )
            return None

    def _apply_geometry(self, model, geometry_path):
        import json

        if not os.path.isfile(geometry_path):
            return

        with open(geometry_path) as f:
            data = json.load(f)

        model.vertices = data.get("vertices")
        model.faces = data.get("faces")
        model.edges = data.get("_debug_raw", {}).get("edges")

        dims = data.get("dimensions") or {}

        if dims:
            model.bounding_box = [
                dims.get("width"),
                dims.get("depth"),
                dims.get("height"),
                dims.get("aspect_hw"),
                dims.get("aspect_hd"),
                dims.get("aspect_wd"),
            ]

    def _apply_prediction(self, model, prediction):
        preds = prediction.get("predictions", {}) if prediction else {}

        super_category = preds.get("super_category") or {}
        model.ai_label = super_category.get("label")
        model.ai_confidence = super_category.get("confidence")

        object_category = preds.get("object_category") or {}
        model.object_category = object_category.get("label")

        style_class = preds.get("style_class") or []
        model.style = style_class[0].get("label") if style_class else None

        materials_primary = preds.get("materials_primary") or {}
        model.material = materials_primary.get("label")

        model.prediction = prediction




class SimiliarModelsView(APIView):
    def get(self, request):
        model_id = request.query_params.get('model_id')
        if not model_id:
            return Response(
                {"detail": "model_id query param is required."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        reference = get_object_or_404(Model3D, pk=model_id)

        try:
            limit = int(request.query_params.get('limit', 12))
        except ValueError:
            limit = 12
        limit = max(1, min(limit, 100))

        label_quota = round(limit * 0.7)
        style_quota = limit - label_quota

        base_qs = Model3D.objects.filter(is_active=True).exclude(pk=reference.pk)

        seen_ids = {reference.pk}

        # ── Bucket 1: same ai_label, favoring same object_category ──
        label_results = []
        if reference.ai_label:
            label_qs = (
                base_qs
                .filter(ai_label=reference.ai_label)
                .annotate(
                    category_match=Case(
                        When(object_category=reference.object_category, then=Value(1)),
                        default=Value(0),
                        output_field=IntegerField(),
                    ),
                    rating_score_annot=F('downloads_count') + F('usage_count'),
                )
                .order_by('-category_match', '-rating_score_annot', '-uploaded_at')
            )
            label_results = list(label_qs[:label_quota])
            seen_ids.update(m.pk for m in label_results)

        # ── Bucket 2: same style ──
        style_results = []
        if reference.style:
            style_qs = (
                base_qs
                .exclude(pk__in=seen_ids)
                .filter(style=reference.style)
                .annotate(rating_score_annot=F('downloads_count') + F('usage_count'))
                .order_by('-rating_score_annot', '-uploaded_at')
            )
            style_results = list(style_qs[:style_quota])
            seen_ids.update(m.pk for m in style_results)

        results = label_results + style_results

        # ── Fallback: if either bucket came up short, top up from
        # whatever's left (prefer the other bucket's criteria first,
        # then just generally popular active models) ──
        missing = limit - len(results)
        if missing > 0:
            fallback_qs = (
                base_qs
                .exclude(pk__in=seen_ids)
                .annotate(rating_score_annot=F('downloads_count') + F('usage_count'))
                .order_by('-rating_score_annot', '-uploaded_at')
            )
            fallback_results = list(fallback_qs[:missing])
            results += fallback_results
            seen_ids.update(m.pk for m in fallback_results)

        serializer = Model3DListSerializer(results, many=True)
        return Response(serializer.data)