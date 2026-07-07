from rest_framework import serializers
from .models import Model3D, RenderImage, Report, Tag, Category


# ── Tag & Category ────────────────────────────────────────

class TagSerializer(serializers.ModelSerializer):
    class Meta:
        model  = Tag
        fields = ('id', 'name')


class CategorySerializer(serializers.ModelSerializer):
    class Meta:
        model  = Category
        fields = ('id', 'name', 'description')


# ── RenderImage ───────────────────────────────────────────

class RenderImageSerializer(serializers.ModelSerializer):
    class Meta:
        model  = RenderImage
        fields = ('id', 'image', 'angle', 'created_at')


# ── Model3D — List (dashboard, خفيف) ─────────────────────

class Model3DListSerializer(serializers.ModelSerializer):
    rating_score = serializers.IntegerField(read_only=True)
    category     = CategorySerializer(read_only=True)
    tags         = TagSerializer(many=True, read_only=True)

    class Meta:
        model  = Model3D
        fields = (
            'id', 'source_file', 'uploaded_at', 'is_active',
            'ai_label', 'ai_confidence',
            'category', 'tags',
            'vertices', 'faces', 'is_manifold',
            'stability_score', 'elongation',
            'views_count', 'downloads_count', 'usage_count', 'rating_score',
        )


# ── Model3D — Detail (كامل) ───────────────────────────────

class Model3DDetailSerializer(serializers.ModelSerializer):
    rating_score  = serializers.IntegerField(read_only=True)
    category      = CategorySerializer(read_only=True)
    tags          = TagSerializer(many=True, read_only=True)
    render_images = RenderImageSerializer(many=True, read_only=True)

    class Meta:
        model  = Model3D
        fields = '__all__'


# ── Model3D — Import (JSON من بلندر) ─────────────────────

class Model3DImportSerializer(serializers.Serializer):
    """
    يتوقع الهيكل الصافي من سكريبت بلندر المحدَّث
    ويقوم بتفكيك الكائنات المتداخلة وحفظها في جدول Model3D.
    """
    id                = serializers.CharField()
    mesh_stats        = serializers.DictField()
    bounding_box      = serializers.ListField(child=serializers.FloatField(), allow_null=True)
    surface_area      = serializers.FloatField(allow_null=True)
    volume_estimate   = serializers.FloatField(allow_null=True)
    topology          = serializers.DictField()
    shape_descriptors = serializers.DictField()
    physics_proxy     = serializers.DictField()
    render_links      = serializers.DictField()

    # حقول AI اختيارية (قد يمررها كولاب معاً أو منفصلة)
    ai_label          = serializers.CharField(allow_null=True, required=False)
    ai_confidence     = serializers.FloatField(allow_null=True, required=False)

    def create(self, validated_data):
        mesh    = validated_data.get('mesh_stats', {})
        topo    = validated_data.get('topology', {})
        shape   = validated_data.get('shape_descriptors', {})
        physics = validated_data.get('physics_proxy', {})
        render  = validated_data.get('render_links', {})

        model, _ = Model3D.objects.update_or_create(
            id=validated_data['id'],
            defaults={
                'source_file':          f"{validated_data['id']}.fbx",
                'model_file':           validated_data.get('model_file'),
                'ai_label':             validated_data.get('ai_label'),
                'ai_confidence':        validated_data.get('ai_confidence'),

                # Mesh Stats
                'vertices':             mesh.get('vertices'),
                'faces':                mesh.get('faces'),
                'edges':                mesh.get('edges'),

                # Measurements
                'bounding_box':         validated_data.get('bounding_box'),
                'surface_area':         validated_data.get('surface_area'),
                'volume_estimate':      validated_data.get('volume_estimate'),

                # Topology
                'is_manifold':          topo.get('is_manifold'),
                'holes_count':          topo.get('holes_count'),
                'connected_components': topo.get('connected_components'),

                # Shape Descriptors
                'symmetry_axis':        shape.get('symmetry_axis'),
                'compactness':          shape.get('compactness'),
                'elongation':           shape.get('elongation'),
                'curvature_mean':       shape.get('curvature_mean'),

                # Physics Proxy
                'center_of_mass':       physics.get('center_of_mass'),
                'stability_score':      physics.get('stability_score'),

                # Render Links
                'views_count':          render.get('views_count', 0),
            }
        )
        return model


# ── Upload User (رفع ملف + استجابة AI فورية) ─────────────

class Model3DUploadResponseSerializer(serializers.Serializer):
    """استجابة endpoint رفع الملف الفعلي"""
    status          = serializers.CharField()
    message         = serializers.CharField()
    model_db_id     = serializers.CharField()
    detected_class  = serializers.CharField(allow_null=True)
    confidence_score = serializers.FloatField(allow_null=True)
    saved_data      = Model3DDetailSerializer()


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
            'vertices', 'faces', 'stability_score',
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
