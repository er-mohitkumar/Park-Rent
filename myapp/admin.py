from django.contrib import admin
from .models import mapPointers, myBooking1, Booked, Chat, Earning, Previous

# Register your models here.
admin.site.register(mapPointers)
admin.site.register(myBooking1)
admin.site.register(Booked)
admin.site.register(Chat)
admin.site.register(Earning)
admin.site.register(Previous)