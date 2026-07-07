# Architecture 3D API


## Setup

```bash

python -m venv venv
source venv/bin/activate       


pip install -r requirements.txt


cp .env.example .env



createdb architecture_db


python manage.py makemigrations users
python manage.py makemigrations models3d
python manage.py migrate


python manage.py createsuperuser


python manage.py runserver
```

---

## Endpoints

### Auth
| Method | URL | Description |
|--------|-----|-------------|
| POST | `/api/auth/register/` | تسجيل مستخدم جديد |
| POST | `/api/auth/login/` | الحصول على JWT token |
| POST | `/api/auth/refresh/` | تجديد الـ token |
| GET  | `/api/auth/me/` | بيانات المستخدم الحالي |

### Models
| Method | URL | Description |
|--------|-----|-------------|
| GET    | `/api/models/` | قائمة + بحث + فلترة |
| POST   | `/api/models/import/` | استيراد JSON من بلندر (مطلوب JWT) |
| GET    | `/api/models/stats/` | إحصائيات الـ dataset |
| GET    | `/api/models/<id>/` | تفاصيل موديل كامل |
| DELETE | `/api/models/<id>/delete/` | حذف موديل (مطلوب JWT) |
| GET    | `/api/models/<id>/renders/` | زوايا الرندر لموديل معين |

### Docs
- Swagger UI: `http://localhost:8000/api/docs/`
- OpenAPI Schema: `http://localhost:8000/api/schema/`

---

## فلاتر البحث (GET /api/models/)

```
?search=0000004               # بحث بالاسم
?is_manifold=true             # موديلات سليمة فقط
?vertices_min=1000            # حد أدنى للـ vertices
?vertices_max=50000           # حد أقصى
?height_min=10&height_max=30  # نطاق الارتفاع
?stability_min=0.8            # stability score أعلى من 0.8
?symmetry_axis=X              # محور التماثل
?ordering=-stability_score    # ترتيب تنازلي
```

---

## استيراد JSON من بلندر

```bash
curl -X POST http://localhost:8000/api/models/import/ \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d @0000004.json
```

---

## هيكل المشروع

```
architecture_api/
├── config/
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
├── apps/
│   ├── users/
│   │   ├── models.py       # Custom User
│   │   ├── serializers.py
│   │   ├── views.py
│   │   └── urls.py
│   └── models3d/
│       ├── models.py       # Model3D, RenderView, Material, LOD, RepairLog
│       ├── serializers.py  # Import serializer يقبل JSON بلندر مباشرة
│       ├── filters.py      # Advanced filtering
│       ├── views.py
│       └── urls.py
├── requirements.txt
├── manage.py
└── .env.example
```
