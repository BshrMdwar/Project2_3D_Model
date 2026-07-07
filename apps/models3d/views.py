"""
views.py — architecture_api v2
يشمل:
  • Pipeline رفع ملف 3D → بلندر → AI → قاعدة البيانات
  • نظام تقييم (downloads + usage)
  • نظام إبلاغ (Report)
  • نظام اقتراح ذكي (Recommendation)
  • وظائف الآدمن الكاملة مع صلاحيات صارمة
  • لوحة إحصائيات متطورة
"""
from __future__ import annotations
import os
from rest_framework.parsers import FormParser, MultiPartParser
from rest_framework import permissions, status
# 1️⃣ الاستيراد الصحيح للدوال الرياضية والـ Expressions
from django.db.models import ExpressionWrapper, IntegerField, F, Q
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

from .models import Model3D, Report, Tag, Category, RenderImage
from .serializers import (
    Model3DListSerializer,
    Model3DDetailSerializer,
    Model3DImportSerializer,
    Model3DAdminUpdateSerializer,
    Model3DRecommendSerializer,
    ReportCreateSerializer,
    ReportDetailSerializer,
    TagSerializer,
    CategorySerializer,
)
from .filters import Model3DFilter


# ════════════════════════════════════════════════════════════
#  Permission Helpers
# ════════════════════════════════════════════════════════════

class IsAdminUser(permissions.BasePermission):
    """يسمح فقط للمستخدمين ذوي role=admin أو is_staff"""

    def has_permission(self, request, view):
        return (
            request.user and
            request.user.is_authenticated and
            (request.user.is_staff or getattr(request.user, 'role', '') == 'admin')
        )


# ════════════════════════════════════════════════════════════
#  Model3D — قراءة عامة
# ════════════════════════════════════════════════════════════

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
    serializer_class = Model3DDetailSerializer
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


# ════════════════════════════════════════════════════════════
#  Model3D — رفع واستيراد
# ════════════════════════════════════════════════════════════

class Model3DImportView(APIView):
    """POST /api/models/import/ — استيراد JSON صافٍ من بلندر (admin)"""
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request):
        serializer = Model3DImportSerializer(data=request.data)
        if serializer.is_valid():
            model = serializer.save()
            return Response({'status': 'imported', 'id': model.id}, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


# class Model3DUploadUserView(APIView):
#     """
#     POST /api/models/upload-user/
#     1. يستقبل ملف 3D (FileField)
#     2. يرسله لسيرفر بلندر Headless (Colab/Ngrok)
#     3. يستقبل JSON + تصنيف AI
#     4. يحفظ في الداتابيز ويعيد الاستجابة الفورية
#     """
#     permission_classes = [permissions.IsAuthenticated]
#     parser_classes     = [MultiPartParser, FormParser]

#     COLAB_API_URL = "https://coeditor-resisting-bogus.ngrok-free.dev/render-and-classify/"

#     def post(self, request, *args, **kwargs):
#         uploaded_file = request.data.get('file')
#         if not uploaded_file:
#             return Response({"error": "No file uploaded"}, status=status.HTTP_400_BAD_REQUEST)

#         try:
#             files = {
#                 'file': (uploaded_file.name, uploaded_file.read(), uploaded_file.content_type)
#             }
#             colab_response = requests.post(self.COLAB_API_URL, files=files, timeout=120)

#             if colab_response.status_code != 200:
#                 return Response({
#                     "error": "Cloud server error during rendering or classification",
#                     "details": colab_response.text,
#                 }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

#             colab_data       = colab_response.json()
#             ai_classification = colab_data.get('ai_classification', {})

#             internal_import_data = {
#                 "id":                f"user_{uploaded_file.name.split('.')[0]}",
#                 "mesh_stats":        colab_data.get('mesh_stats', {}),
#                 "bounding_box":      colab_data.get('bounding_box'),
#                 "surface_area":      colab_data.get('surface_area'),
#                 "volume_estimate":   colab_data.get('volume_estimate'),
#                 "topology":          colab_data.get('topology', {}),
#                 "shape_descriptors": colab_data.get('shape_descriptors', {}),
#                 "physics_proxy":     colab_data.get('physics_proxy', {}),
#                 "render_links":      colab_data.get('render_links', {}),
#                 "ai_label":          ai_classification.get('label'),
#                 "ai_confidence":     ai_classification.get('confidence'),
#             }

#             serializer = Model3DImportSerializer(data=internal_import_data)
#             if serializer.is_valid():
#                 model = serializer.save()
#                 # ربط المستخدم الرافع
#                 model.uploaded_by = request.user
#                 model.save(update_fields=['uploaded_by'])

#                 return Response({
#                     "status":           "success",
#                     "message":          "Model uploaded, rendered, and classified successfully!",
#                     "model_db_id":      model.id,
#                     "detected_class":   ai_classification.get('label'),
#                     "confidence_score": ai_classification.get('confidence'),
#                     "saved_data":       Model3DDetailSerializer(model).data,
#                 }, status=status.HTTP_201_CREATED)

#             return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

#         except requests.exceptions.Timeout:
#             return Response({"error": "Cloud Render request timed out."}, status=status.HTTP_504_GATEWAY_TIMEOUT)
#         except requests.exceptions.ConnectionError:
#             return Response({"error": "Could not connect to Blender server."}, status=status.HTTP_503_SERVICE_UNAVAILABLE)
#         except Exception as e:
#             return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

# تأكد من استيراد الـ Serializers والموديلات الخاصة بك هنا
# from .serializers import Model3DImportSerializer, Model3DDetailSerializer


class Model3DUploadUserView(APIView):
    """
    POST /api/models/upload-user/
    1. يستقبل ملف 3D (FileField) من المستخدم.
    2. يرسله مباشرة إلى API سكريبت بلندر الخاص بك لمعالجته محلياً.
    3. يستقبل الـ JSON المحسوب من بلندر ويخزنه في قاعدة البيانات.
    """
    permission_classes = [permissions.IsAuthenticated]
    parser_classes = [MultiPartParser, FormParser]

    # 🔗 تم إبقاء الرابط متوافقاً مع سيرفر Flask المحلي الجديد
    BLENDER_API_URL = "http://127.0.0.1:5000/process-3d/"

    def post(self, request, *args, **kwargs):
        uploaded_file = request.data.get('file')
        if not uploaded_file:
            return Response({"error": "No file uploaded"}, status=status.HTTP_400_BAD_REQUEST)

        try:
            # 📦 تجهيز الملف لإرساله إلى الـ API الخاص ببلندر
            files = {
                'file': (uploaded_file.name, uploaded_file.read(), uploaded_file.content_type)
            }

            # 🔄 حرج جداً: إعادة مؤشر قراءة الملف إلى الصفر
            # لأن uploaded_file.read() استهلكت البيانات، وبدون الـ seek سيفشل دجانغو بحفظ الملف على القرص.
            uploaded_file.seek(0)

            # 🚀 إرسال الطلب إلى سيرفر بلندر المحلي
            # رفعنا الـ timeout لـ 610 ثوانٍ ليتوافق مع معالجة الرندر الثقيلة في بلندر (Cycles)
            blender_response = requests.post(
                self.BLENDER_API_URL, files=files, timeout=610)

            if blender_response.status_code != 200:
                try:
                    error_details = blender_response.json()
                except Exception:
                    error_details = blender_response.text

                return Response({
                    "error": "Blender local server error during processing",
                    "details": error_details,
                }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

            # 📥 استلام البيانات من بلندر
            blender_data = blender_response.json()
            ai_classification = blender_data.get('ai_classification', {})

            # 🧱 تفكيك البيانات القادمة وحقنها في الـ Serializer ليتم حفظها في جدول Model3D
            internal_import_data = {
                "id":                f"user_{uploaded_file.name.split('.')[0]}",
                # الآن سيحفظ بشكل سليم بفضل الـ seek(0)
                "model_file":        uploaded_file,
                "mesh_stats":        blender_data.get('mesh_stats', {}),
                "bounding_box":      blender_data.get('bounding_box'),
                "surface_area":      blender_data.get('surface_area'),
                "volume_estimate":   blender_data.get('volume_estimate'),
                "topology":          blender_data.get('topology', {}),
                "shape_descriptors": blender_data.get('shape_descriptors', {}),
                "physics_proxy":     blender_data.get('physics_proxy', {}),
                "render_links":      blender_data.get('render_links', {}),
                "ai_label":          ai_classification.get('label'),
                "ai_confidence":     ai_classification.get('confidence'),
            }

            # نمرر الـ request داخل الـ context للـ serializer كأفضل ممارسة
            serializer = Model3DImportSerializer(
                data=internal_import_data, context={'request': request})
            if serializer.is_valid():
                # نقوم بحفظ الكائن وتحديد المستخدم مباشرة بدون الحاجة لعمل save مرتين
                model = serializer.save(uploaded_by=request.user)
                if uploaded_file:
                    model.model_file.save(
                        uploaded_file.name, uploaded_file, save=True)
                    for i in range(1, 13):
                        RenderImage.objects.create(
                            model=model,  # ربط الصورة بالموديل الحالي عبر حقل model
                            image=f"models3d/renders/{model.id}/angle_{i}.png",  # المسار النسبي المحفوظ على الهارد في الميديا
                            angle=i * 30  # حساب الزاوية (30, 60, 90... حتى 360)
                        )
                return Response({
                    "status":           "success",
                    "message":          "Model processed via Blender and saved successfully!",
                    "model_db_id":      model.id,
                    "detected_class":   ai_classification.get('label'),
                    "confidence_score": ai_classification.get('confidence'),
                    "saved_data":       Model3DDetailSerializer(model).data,
                }, status=status.HTTP_201_CREATED)

            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        except requests.exceptions.Timeout:
            return Response({"error": "Blender local script request timed out (Max 10 minutes)."}, status=status.HTTP_504_GATEWAY_TIMEOUT)
        except requests.exceptions.ConnectionError:
            return Response({"error": "Could not connect to local Blender API service. Make sure Flask is running on port 5000."}, status=status.HTTP_503_SERVICE_UNAVAILABLE)
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

# ════════════════════════════════════════════════════════════
#  Downloads / Usage Counter
# ════════════════════════════════════════════════════════════


class Model3DDownloadView(APIView):
    """
    POST /api/models/<id>/download/
    يسجّل تحميلاً ويعيد URL أو البيانات اللازمة للتحميل.
    """
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request, id):
        try:
            model = Model3D.objects.get(id=id, is_active=True)
        except Model3D.DoesNotExist:
            return Response({'error': 'Model not found'}, status=status.HTTP_404_NOT_FOUND)

        model.increment_downloads()
        return Response({
            'status':      'download_registered',
            'model_id':    id,
            'file_url':    request.build_absolute_uri(model.model_file.url) if model.model_file else None,
        })


# تأكد من استيراد الموديل الخاص بك هنا
# from .models import Model3D

# تأكد من استيراد الموديل الخاص بك
# from apps.models3d.models import Model3D


class Model3DDownloadView(APIView):
    """
    POST /api/models/<id>/download/
    يسجّل تحميلاً ويعيد الرابط المباشر لتنزيل ملف الـ 3D على جهاز المستخدم.
    """
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request, id, *args, **kwargs):
        try:
            # 1. جلب الموديل والتأكد أنه نشط
            model = Model3D.objects.get(id=id, is_active=True)
        except Model3D.DoesNotExist:
            return Response({'error': 'Model not found'}, status=status.HTTP_404_NOT_FOUND)

        # 2. التأكد من أن حقل الملف ليس فارغاً في قاعدة البيانات
        if not model.model_file:
            return Response({'error': 'No file associated with this model record'}, status=status.HTTP_400_BAD_REQUEST)

        # 3. تسجيل عملية التحميل وزيادة العداد عبر دالتك الخاصة
        model.increment_downloads()

        # 4. بناء الرابط المطلق الكامل للملف (مثال: http://127.0.0.1:8000/media/models/...)
        file_url = request.build_absolute_uri(model.model_file.url)

        return Response({
            'status': 'download_registered',
            'model_id': id,
            'file_url': file_url,
            'message': 'Use this link to download the file directly to your local storage.'
        }, status=status.HTTP_200_OK)
# ════════════════════════════════════════════════════════════
#  Stats
# ════════════════════════════════════════════════════════════


class Model3DStatsView(APIView):
    """GET /api/models/stats/ — إحصائيات عامة (متاحة للجميع)"""
    permission_classes = [permissions.AllowAny]

    def get(self, request):
        qs = Model3D.objects.filter(is_active=True)
        stats = qs.aggregate(
            total_models=Count('id'),
            avg_vertices=Avg('vertices'),
            avg_faces=Avg('faces'),
            avg_stability=Avg('stability_score'),
            manifold_count=Count('id', filter=Q(is_manifold=True)),
            total_downloads=Sum('downloads_count'),
            total_views=Sum('views_count'),
        )
        return Response(stats)


class AdminDashboardStatsView(APIView):
    """
    GET /api/admin/dashboard-stats/
    لوحة إحصائيات متطورة للمشرفين.
    """
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
    serializer_class = Model3DAdminUpdateSerializer
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


# ════════════════════════════════════════════════════════════
#  Tag & Category CRUD — للآدمن
# ════════════════════════════════════════════════════════════

class TagListCreateView(generics.ListCreateAPIView):
    """GET|POST /api/admin/tags/"""
    queryset = Tag.objects.all()
    serializer_class = TagSerializer
    permission_classes = [IsAdminUser]


class TagDetailView(generics.RetrieveUpdateDestroyAPIView):
    """GET|PATCH|DELETE /api/admin/tags/<pk>/"""
    queryset = Tag.objects.all()
    serializer_class = TagSerializer
    permission_classes = [IsAdminUser]


class CategoryListCreateView(generics.ListCreateAPIView):
    """GET|POST /api/admin/categories/"""
    queryset = Category.objects.all()
    serializer_class = CategorySerializer
    permission_classes = [IsAdminUser]


class CategoryDetailView(generics.RetrieveUpdateDestroyAPIView):
    """GET|PATCH|DELETE /api/admin/categories/<pk>/"""
    queryset = Category.objects.all()
    serializer_class = CategorySerializer
    permission_classes = [IsAdminUser]
class Model3DRendersListView(APIView):
    """
    GET /api/models/<id>/renders/
    جلب كافة الصور المُرندرة الخاصة بموديل معين
    """
    permission_classes = [permissions.AllowAny]

    def get(self, request, id):
        try:
            model_instance = Model3D.objects.get(id=id, is_active=True)
        except Model3D.DoesNotExist:
            return Response({'error': 'Model not found'}, status=status.HTTP_404_NOT_FOUND)
            
        # 🚨 التعديل هنا: استبدال model_3d بـ model بناءً على بنية قاعدة البيانات عندك
        renders = RenderImage.objects.filter(model=model_instance)
        
        # تجميع روابط الصور بشكل مطلق ومباشر
        urls = [request.build_absolute_uri(img.image.url) for img in renders if img.image]
        
        return Response({
            "model_id": id,
            "renders_count": len(urls),
            "images": urls
        }, status=status.HTTP_200_OK)