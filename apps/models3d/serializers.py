from rest_framework import serializers

from apps.users.serializers import UserSerializer
from .models import Model3D, Report


class Model3DListSerializer(serializers.ModelSerializer):
    uploaded_by = UserSerializer(read_only=True)
    rating_score = serializers.IntegerField(read_only=True)
    class Meta:
        model  = Model3D
        fields = (
            'id', 'title', 'description', 'banner_url', 'model_url','uploaded_at', 'is_active',
            'uploaded_by',
            'ai_label', 'ai_confidence',
            'vertices', 'faces', 'object_category',
            'views_count', 'downloads_count', 'rating_score', 'prediction'
        )


class ReportCreateSerializer(serializers.ModelSerializer):
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
    model = Model3DListSerializer(read_only=True)
    reporter = UserSerializer(read_only=True)
    class Meta:
        model  = Report
        fields = '__all__'
        read_only_fields = ('reporter', 'created_at', 'updated_at')


class Model3DRecommendSerializer(serializers.ModelSerializer):
    """تسلسل خفيف لنتائج الاقتراحات"""
    rating_score = serializers.IntegerField(read_only=True)

    class Meta:
        model  = Model3D
        fields = (
            'id', 'ai_label',
            'vertices', 'faces',
            'downloads_count', 'rating_score',
        )
