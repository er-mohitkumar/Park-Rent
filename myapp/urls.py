from django.urls import path
from django.contrib.auth import views as auth_views
from . import views
from django.conf import settings
from django.conf.urls.static import static


urlpatterns = [
    path('',views.landingPage, name = 'landingPage'),
    path('register',views.register,name='register'),
    path('login',views.login,name='login'),
    path('logout',views.logout,name='logout'),
    path('display',views.display,name='display'),
    path('provider',views.provider,name='provider'),
    path('need',views.need,name='need'),
    path('pdashboard',views.pdashboard,name='pdashboard'),
    path('delLocation/<int:pk>',views.delLocation,name='delLocation'),
    path('show/',views.show,name='show'),
    path('myBookings/<int:id>', views.myBookings, name='myBookings'),
    path('book',views.book,name='book'),
    path('find/<int:id>', views.find, name='find'),
    path('tripOver/<int:id>', views.tripOver, name='tripOver'),
    path('payment/', views.payment, name='payment'),
    path('redirecting/', views.redirecting, name='redirecting'),
    path('confirmed/', views.confirmed, name='confirmed'),
    path('profile/', views.profile, name='profile'),
    path('profileShow/', views.profileShow, name='profileShow'),
    path('my-view/', views.my_view, name='my_view'),
    path('password_reset/', auth_views.PasswordResetView.as_view(template_name='password_reset_form.html'), name='password_reset'),
    path('password_reset_done/', auth_views.PasswordResetDoneView.as_view(template_name='password_reset_done.html'), name='password_reset_done'),
    path('password_reset_confirm/<uidb64>/<token>/',auth_views.PasswordResetConfirmView.as_view(template_name='password_reset_confirm.html'), name='password_reset_confirm'),
    path('password_reset_complete/', auth_views.PasswordResetCompleteView.as_view(template_name='password_reset_complete.html'), name='password_reset_complete'),
    path('live-footage/<int:id>/', views.live_footage, name='live_footage'),

] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)