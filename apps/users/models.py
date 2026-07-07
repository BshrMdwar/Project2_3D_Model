from django.contrib.auth.models import AbstractUser
from django.db import models


class User(AbstractUser):
    class Role(models.TextChoices):
        ADMIN    = 'admin',    'Admin'
        DESIGNER = 'designer', 'Designer'
        VIEWER   = 'viewer',   'Viewer'

    email = models.EmailField(unique=True)
    role  = models.CharField(max_length=20, choices=Role.choices, default=Role.VIEWER)

    USERNAME_FIELD  = 'email'
    REQUIRED_FIELDS = ['username']

    def __str__(self):
        return self.email
