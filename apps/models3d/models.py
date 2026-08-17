from django.db import models
from django.contrib.postgres.fields import ArrayField
from django.contrib.auth import get_user_model
from django.db.models import F

User = get_user_model()


class Model3D(models.Model):

    id          = models.CharField(max_length=100, primary_key=True, db_index=True)
    title = models.CharField(max_length=100)
    description = models.CharField(blank=True,null=True)

    uploaded_at = models.DateTimeField(auto_now_add=True)
    updated_at  = models.DateTimeField(auto_now=True)


    model_url = models.CharField(blank=True, null=True)
    banner_url = models.CharField(blank=True, null=True)

    uploaded_by = models.ForeignKey(
        User, null=True, blank=True,
        on_delete=models.SET_NULL,
        related_name='uploaded_models'
    )

    is_active = models.BooleanField(default=True)

    # ── AI Classification ─────────────────────────────────
    ai_label      = models.CharField(max_length=100, null=True, blank=True)
    ai_confidence = models.FloatField(null=True, blank=True)
    material=models.CharField(blank=True, null=True)
    object_category = models.CharField(blank=True, null=True)
    style = models.CharField(blank=True, null=True)
    prediction = models.JSONField(blank=True, null=True)

    # ── Mesh Stats ────────────────────────────────────────
    vertices = models.IntegerField(null=True)
    faces    = models.IntegerField(null=True)
    edges    = models.IntegerField(null=True)

    # ── Bounding Box ─────────────────────────────────────
    bounding_box = ArrayField(models.FloatField(), size=6, null=True, blank=True)

    # ── Platform Counters ─────────────────────────────────
    views_count     = models.PositiveIntegerField(default=0)
    downloads_count = models.PositiveIntegerField(default=0)
    rating = models.FloatField(default=0)
    rating_count = models.PositiveIntegerField(default=0)

    # التقييم المحفوظ: متوسط تقييمات المستخدمين من 5
    @property
    def rating_score(self):
        return self.rating

    def apply_rating(self, user_rating):
        total_rating = (self.rating * self.rating_count) + user_rating
        self.rating_count += 1
        self.rating = total_rating / self.rating_count

    class Meta:
        ordering            = ['-uploaded_at']
        verbose_name        = '3D Model'
        verbose_name_plural = '3D Models'

    def __str__(self):
        return self.id

    # ── Helper: atomic increment ──────────────────────────
    def increment_downloads(self):
        Model3D.objects.filter(pk=self.pk).update(
            downloads_count=F('downloads_count') + 1,
        )

    def increment_views(self):
        Model3D.objects.filter(pk=self.pk).update(views_count=F('views_count') + 1)


class Report(models.Model):
    class Status(models.TextChoices):
        PENDING  = 'pending',  'قيد المراجعة'
        RESOLVED = 'resolved', 'تم الحل'
        DISMISSED = 'dismissed', 'مرفوض'

    model      = models.ForeignKey(Model3D, on_delete=models.CASCADE, related_name='reports')
    reporter   = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, related_name='submitted_reports')
    reason     = models.TextField()
    status     = models.CharField(max_length=20, choices=Status.choices, default=Status.PENDING)
    admin_note = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"Report({self.model_id} by {self.reporter_id} — {self.status})"


class UserCollection(models.Model):
    user = models.OneToOneField(
        User,
        on_delete=models.CASCADE,
        related_name='user_collection',
    )
    saved_models = models.ManyToManyField(
        'models3d.Model3D',
        related_name='saved_in_collections',
        blank=True,
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-updated_at']

    def __str__(self):
        return f'UserCollection({self.user_id})'
