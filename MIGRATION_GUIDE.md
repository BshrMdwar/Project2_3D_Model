# تعليمات تطبيق الـ Migrations

بعد نسخ الملفات، شغّل هذه الأوامر بالترتيب:

```bash
# 1. إنشاء ملفات الـ migration من الـ models الجديدة
python manage.py makemigrations models3d

# 2. تطبيق جميع الـ migrations على قاعدة البيانات
python manage.py migrate

# 3. (اختياري) إنشاء superuser للآدمن
python manage.py createsuperuser
```

| الرابط | الوصف |
|--------|-------|
| `GET /api/models/top-rated/` | الموديلات الأعلى تقييماً |
| `POST /api/models/<id>/download/` | تسجيل تحميل وإعادة رابط الملف |
| `GET /api/models/<id>/recommend/` | اقتراحات ذكية |
| `POST /api/models/report/` | تقديم بلاغ |
| `GET /api/models/admin/models/` | قائمة جميع الموديلات (آدمن) |
| `PATCH /api/models/admin/models/<id>/update/` | تعديل tags/category/is_active |
| `PATCH /api/models/admin/models/<id>/hide/` | إخفاء/إظهار (Soft Delete) |
| `GET /api/models/admin/reports/` | قائمة البلاغات |
| `PATCH /api/models/admin/reports/<pk>/` | تحديث حالة البلاغ |
| `GET /api/models/admin/dashboard-stats/` | لوحة إحصائيات متطورة |
| `GET\|POST /api/models/admin/tags/` | إدارة الوسوم |
| `GET\|POST /api/models/admin/categories/` | إدارة التصنيفات |
