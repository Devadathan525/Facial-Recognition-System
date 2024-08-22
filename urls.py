"""StudentAttendence URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('', views.home, name='home'),
    path('create_datsets', views.create_datsets, name='create_datsets'),
    path('training', views.training, name='training'),
    path('create_datsets_hod', views.create_datsets_hod, name='create_datsets_hod'),
    path('training_hod', views.training_hod, name='training_hod'),
    path('attendence', views.attendence, name='attendence'),
    #path('persondetection',views.persondetection,name='persondetection'),
]+ static(settings.MEDIA_URL, document_root = settings.MEDIA_ROOT)
