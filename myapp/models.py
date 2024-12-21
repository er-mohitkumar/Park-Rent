
from django.db import models
from django.contrib.auth.models import User


def upload_to_media(instance, filename):
    return 'media/{0}'.format(filename)


class mapPointers(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    latitude = models.FloatField()
    longitude = models.FloatField()
    rate = models.IntegerField()
    photo = models.ImageField(upload_to=upload_to_media)
    status = models.BooleanField(default=False)
    booked_by = models.CharField(default="empty", max_length=500)
    email = models.EmailField(default = "megh.shah2003@gmail.com",max_length=254)
    Booked_email = models.EmailField(default = "sample@gmail.com",max_length=254)

    def __str__(self):
        return f'MapPointer {self.id} - User: {self.user.username}'


class myBooking1(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    name = models.CharField(max_length=50)
    latitude = models.FloatField()
    longitude = models.FloatField()
    rate = models.IntegerField()
    photo = models.ImageField()
    var = models.IntegerField(default = 0)
    email = models.EmailField(default = "mohitkumarmk112003@gmail.com",max_length=254)


class Booked(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    latitude = models.FloatField()
    longitude = models.FloatField()
    rate = models.IntegerField()
    photo = models.ImageField()
    status = models.BooleanField(default=False)


    def __str__(self):
        return f'MapPointer {self.id} - User: {self.user.username}'

class Chat(models.Model):
    sender = models.ForeignKey(User, related_name='sender', on_delete=models.CASCADE)
    receiver = models.ForeignKey(User, related_name='receiver', on_delete=models.CASCADE)
    message = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True)


class Earning(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    earning = models.IntegerField(default=0)


class Previous(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    name = models.CharField(max_length=50)
    latitude = models.FloatField()
    longitude = models.FloatField()
    rate = models.IntegerField()
    