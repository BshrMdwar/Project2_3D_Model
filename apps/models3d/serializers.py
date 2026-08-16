from rest_framework import serializers
from .models import Model3D, Report



# ── Model3D — List (dashboard, خفيف) ─────────────────────

class Model3DListSerializer(serializers.ModelSerializer):
    rating_score = serializers.IntegerField(read_only=True)
    class Meta:
        model  = Model3D
        fields = (
            'id', 'title', 'description', 'banner_url', 'model_url','uploaded_at', 'is_active',
            'ai_label', 'ai_confidence',
            'vertices', 'faces',
            'views_count', 'downloads_count', 'usage_count', 'rating_score', 'prediction'
        )


# ── Upload User (رفع ملف + استجابة AI فورية) ─────────────

# class Model3DUploadResponseSerializer(serializers.Serializer):
#     """استجابة endpoint رفع الملف الفعلي"""
#     status          = serializers.CharField()
#     message         = serializers.CharField()
#     model_db_id     = serializers.CharField()
#     detected_class  = serializers.CharField(allow_null=True)
#     confidence_score = serializers.FloatField(allow_null=True)
#     saved_data      = Model3DDetailSerializer()


# ── Report ────────────────────────────────────────────────

class ReportCreateSerializer(serializers.ModelSerializer):
    """يستقبله المستخدم لإنشاء بلاغ جديد"""
    class Meta:
        model  = Report
        fields = ('model', 'reason')

    def create(self, validated_data):
        request = self.context['request']
        return Report.objects.create(
            reporter=request.user,
            **validated_data
        )


class ReportDetailSerializer(serializers.ModelSerializer):
    """يستخدمه الآدمن لعرض وتعديل البلاغ"""
    class Meta:
        model  = Report
        fields = '__all__'
        read_only_fields = ('reporter', 'created_at', 'updated_at')


# ── Recommendation ────────────────────────────────────────

class Model3DRecommendSerializer(serializers.ModelSerializer):
    """تسلسل خفيف لنتائج الاقتراحات"""
    rating_score = serializers.IntegerField(read_only=True)

    class Meta:
        model  = Model3D
        fields = (
            'id', 'ai_label',
            'vertices', 'faces',
            'downloads_count', 'usage_count', 'rating_score',
        )
