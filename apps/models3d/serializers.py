from rest_framework import serializers
from .models import Model3D, Report, Tag, Category


# ── Tag & Category ────────────────────────────────────────

class TagSerializer(serializers.ModelSerializer):
    class Meta:
        model  = Tag
        fields = ('id', 'name')


class CategorySerializer(serializers.ModelSerializer):
    class Meta:
        model  = Category
        fields = ('id', 'name', 'description')



# ── Model3D — List (dashboard, خفيف) ─────────────────────

class Model3DListSerializer(serializers.ModelSerializer):
    rating_score = serializers.IntegerField(read_only=True)
    category     = CategorySerializer(read_only=True)
    tags         = TagSerializer(many=True, read_only=True)

    class Meta:
        model  = Model3D
        fields = (
            'id', 'uploaded_at', 'is_active',
            'ai_label', 'ai_confidence',
            'category', 'tags',
            'vertices', 'faces',
            'views_count', 'downloads_count', 'usage_count', 'rating_score',
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
            'id', 'ai_label', 'category',
            'vertices', 'faces',
            'downloads_count', 'usage_count', 'rating_score',
        )


# ── Admin: update tags/category on a model ────────────────

class Model3DAdminUpdateSerializer(serializers.ModelSerializer):
    """يتيح للآدمن تعديل الوسوم والتصنيف وحالة الإخفاء"""
    tag_ids      = serializers.PrimaryKeyRelatedField(
        queryset=Tag.objects.all(), many=True, source='tags', required=False
    )
    category_id  = serializers.PrimaryKeyRelatedField(
        queryset=Category.objects.all(), source='category', required=False, allow_null=True
    )

    class Meta:
        model  = Model3D
        fields = ('is_active', 'ai_label', 'category_id', 'tag_ids')

    def update(self, instance, validated_data):
        tags = validated_data.pop('tags', None)
        instance = super().update(instance, validated_data)
        if tags is not None:
            instance.tags.set(tags)
        return instance
