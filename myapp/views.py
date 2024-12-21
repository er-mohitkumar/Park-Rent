from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.models import User, auth
from django.contrib import messages
from django.contrib.auth import authenticate, login
from django.http import HttpResponse
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
from .models import mapPointers, myBooking1, Booked, Earning, Previous
from django.shortcuts import render, get_object_or_404, redirect
import uuid
import time
from django.core.mail import send_mail
from django.template.loader import render_to_string
from django.core.management.base import BaseCommand
from datetime import datetime, timedelta
from django.urls import reverse

import cv2
import numpy as np
import cvzone
import requests
from ultralytics import YOLO
import pickle
import pandas as pd
import easyocr
import threading

# Initialize the live camera feed
LIVE_CAMERA_URL = "http://192.168.236.227:8080/video"

model = YOLO('yolov8s.pt')

# Define COCO class names as a list
class_list = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

# Load previous polylines and area names
try:
    with open("manish", "rb") as f:
        data = pickle.load(f)
        polylines, area_names = data['polylines'], data['area_names']
except FileNotFoundError:
    polylines = []
    area_names = []

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

# Define placeholders for real-time data
real_time_data = {"free_space": 0, "car_count": 0}

# Background thread function to process live feed
def process_live_feed_continuously():
    global real_time_data
    LIVE_CAMERA_URL = "http://example.com/live_feed"
    while True:
        try:
            # Fetch the live camera feed
            response = requests.get(LIVE_CAMERA_URL, timeout=5)
            video = np.array(bytearray(response.content), dtype=np.uint8)
            frame = cv2.imdecode(video, -1)
        except Exception as e:
            print(f"Error fetching live camera feed: {e}")
            real_time_data = {"free_space": 0, "car_count": 0}
            time.sleep(5)
            continue

        if frame is not None:
            # Resize the frame
            frame = cv2.resize(frame, (1020, 500))
            
            # Predict detections using YOLO model
            results = model.predict(frame)
            detections = results[0].boxes.data
            detections_df = pd.DataFrame(detections).astype("float")

            # Find car centroids
            car_centroids = []
            for _, row in detections_df.iterrows():
                x1, y1, x2, y2, _, class_id = map(int, row)
                class_name = class_list[class_id]
                if 'car' in class_name:
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                    car_centroids.append((cx, cy))

            # Calculate free space
            free_space_counter = []
            for i, polyline in enumerate(polylines):
                for cx, cy in car_centroids:
                    result = cv2.pointPolygonTest(np.array(polyline, dtype=np.int32), (cx, cy), False)
                    if result >= 0:
                        free_space_counter.append(tuple(map(tuple, polyline)))

            car_count = len(set(free_space_counter))
            free_space = len(polylines) - car_count

            real_time_data = {"free_space": free_space, "car_count": car_count}
        else:
            real_time_data = {"free_space": 0, "car_count": 0}

# Start the background thread
threading.Thread(target=process_live_feed_continuously, daemon=True).start()

# Create your views here.
def live_footage(request, id):
    video_url = "http://192.168.236.227:8080/video"
    return render(request, 'live_footage.html', {'video_url': video_url})

def landingPage(request):
    return render(request,'landingPage.html')

def register(request):
    if request.method == 'POST':
        username = request.POST['username'].strip()
        email = request.POST['email'].strip().lower()
        password = request.POST['password']
        password2 = request.POST['password2']

        if password == password2:
            if User.objects.filter(email=email).exists():
                messages.info(request, 'email already exists')
                return redirect('register')
            elif User.objects.filter(username=username).exists():
                messages.info(request, 'username already exists')
                return redirect('register')
            else:
                user = User.objects.create_user(username=username, email=email, password=password)
                Earning.objects.create(user=user, earning=0)
                return redirect('login')
        else:
            messages.info(request, 'password not same')
            return redirect('register')
    
    else:
        return render(request, 'register.html')


def login(request):
    if request.method == 'POST':
        username = request.POST['username'].strip()
        password = request.POST['password']
        user = authenticate(request, username=username , password = password)

        if user is not None:
            auth.login(request,user)
            return redirect('display')
        else:
            messages.info(request, 'Invalid credentials')
            return redirect('login')
    else:
        return render(request, 'login.html')
    
    
def logout(request):
    auth.logout(request)
    return redirect('login')

def display(request):
    user = request.user
    return render(request, 'display.html')

def need(request):
    return render(request, 'need.html')

def provider(request):
    if request.method == 'POST':
        curr = mapPointers()
        curr.user = request.user
        curr.photo = request.FILES['photo']
        curr.latitude = request.POST['latitude']
        curr.longitude = request.POST['longitude']
        curr.rate = request.POST['rate']
        curr.status = False
        curr.email = request.user.email
        curr.save()
        return redirect('pdashboard')
    else:
        return render(request, 'provider.html')


def pdashboard(request):
    lists = mapPointers.objects.filter(user = request.user)
    earn = Earning.objects.get(user = request.user)
    return render(request,'pdashboard.html',locals())


def delLocation(request, pk=None):
    hw = get_object_or_404(mapPointers, id=pk)
    current_url = request.META.get('HTTP_REFERER')

    hw.delete()

    if 'pdashboard' in current_url:
        redirect_url = reverse('pdashboard')
    elif 'profile' in current_url:
        redirect_url = reverse('profile')
    else:
        redirect_url = reverse('display')

    return redirect(redirect_url)


def show(request):
    lists = mapPointers.objects.filter(user = request.user)
    return render(request, 'show.html',locals())

def need(request):
    lists = mapPointers.objects.all()
    return render(request, 'need.html',locals())

# def myBookings(request, id):
#     try:
#         curr = get_object_or_404(mapPointers, id=id)
        
#         new_booking = myBooking1()
#         new_booking.user = request.user
#         new_booking.name = curr.user
#         new_booking.photo = curr.photo 
#         new_booking.rate = curr.rate  
#         new_booking.latitude = curr.latitude  
#         new_booking.longitude = curr.longitude  
#         new_booking.var = curr.id
#         new_booking.save()
        
#         curr.status = True 
#         curr.save() 
#         return redirect('book')
#     except mapPointers.DoesNotExist:
#         return redirect('book')


def book(request):
    lists = myBooking1.objects.filter(user = request.user)
    return render(request,'book.html',locals())

def find(request,id):
    curr = myBooking1.objects.get(id = id)
    latitude = curr.latitude
    longitude = curr.longitude
    return render(request, 'find.html',locals())

def tripOver(request, id):
    try:
        curr = get_object_or_404(myBooking1, id=id)
        
        new_booking = mapPointers(id=curr.var)
        new_booking.user = User.objects.get(username=curr.name)
        new_booking.email = new_booking.user.email
        new_booking.status = False
        new_booking.photo = curr.photo 
        new_booking.rate = curr.rate  
        new_booking.latitude = curr.latitude  
        new_booking.longitude = curr.longitude
        new_booking.booked_by = "empty"
        new_booking.save()

        past = Previous()
        past.user = request.user
        past.name = User.objects.get(username=curr.name)
        past.latitude = curr.latitude  
        past.longitude = curr.longitude
        past.rate = curr.rate

        past.save()
        
        parkerOver(curr.user.email,new_booking.user)
        providerOver(new_booking.email,curr.user)

        
        curr.delete()
        return redirect('book')
    except mapPointers.DoesNotExist:
        return redirect('book')

def parkerOver(user_email, curr):
    subject = 'Booking over'
    context = {'curr': curr}
    message = render_to_string('parkerOver.html', context)
    sender_email = 'team.wheelos@gmail.com'
    send_mail(subject, message, sender_email, [user_email])

def providerOver(user_email,curr):
    subject = 'Booking over'
    context = {'curr': curr,}
    message = render_to_string('providerOver.html', context)
    sender_email = 'team.wheelos@gmail.com'
    send_mail(subject, message, sender_email, [user_email])



def payment(request):
    return render(request, 'payment.html')

def myBookings(request, id):
    try:
        curr = get_object_or_404(mapPointers, id=id)
        
        new_booking = myBooking1()
        new_booking.user = request.user
        new_booking.name = curr.user
        new_booking.photo = curr.photo 
        new_booking.rate = curr.rate  
        new_booking.latitude = curr.latitude  
        new_booking.longitude = curr.longitude  
        new_booking.var = curr.id
        new_booking.email = curr.email
        new_booking.save()

        earn = Earning.objects.get(user = curr.user)
        earn.earning += curr.rate
        earn.save()
        
        curr.status = True 
        curr.booked_by = request.user.username
        curr.Booked_email = request.user.email
        curr.save()   

        confirmParker(request.user.email, curr)
        confirmProvider(curr.email, curr, request.user.username)

        return redirect('payment')
    except mapPointers.DoesNotExist:
        return redirect('book')

def confirmParker(user_email, curr):
    subject = 'Parking Booking Confirmation'
    context = {'booking_details': curr}
    message = render_to_string('confirmParker.html', context)
    sender_email = 'team.wheelos@gmail.com'
    send_mail(subject, message, sender_email, [user_email])


def confirmProvider(user, curr, username):
    subject = 'Parking Booking Confirmation'
    context = {'curr': curr,
                'username':username,    
            }
    message = render_to_string('confirmProvider.html', context)
    sender_email = 'team.wheelos@gmail.com'
    send_mail(subject, message, sender_email, [user])


def redirecting(request):
    return render(request, 'redirecting.html')

def confirmed(request):
    return render(request, 'confirmed.html')

def profile(request):
    booked = myBooking1.objects.filter(user=request.user)
    myBookings = mapPointers.objects.filter(user=request.user)
    earn = Earning.objects.get(user=request.user)
    user = request.user
    try:
        past = Previous.objects.filter(user=request.user)
    except Previous.DoesNotExist:
        past = None

    # Add real-time data to context
    free_space = real_time_data["free_space"]
    car_count = real_time_data["car_count"]
    context = locals()
    print("Context passed to template:", context)
    return render(request, 'profile.html', context)

def profileShow(request):
    lists = mapPointers.objects.filter(user = request.user)
    return render(request, 'profileShow.html',locals())


def my_view(request):
    current_time = datetime.now().time()
    if current_time.hour == 10 and current_time.hour < 11:
        bookings_to_update = myBooking1.objects.all()
        
        for booking in bookings_to_update:
            try:
                new_booking = mapPointers.objects.get(id=booking.var)
                new_booking.user = User.objects.get(username=booking.name)
                new_booking.email = new_booking.user.email
                new_booking.status = False
                new_booking.photo = booking.photo 
                new_booking.rate = booking.rate  
                new_booking.latitude = booking.latitude  
                new_booking.longitude = booking.longitude
                new_booking.booked_by = "empty"
                new_booking.save()

                past = Previous()
                past.user = new_booking.user
                past.name = new_booking.user
                past.latitude = booking.latitude  
                past.longitude = booking.longitude
                past.rate = booking.rate
                past.save()
                
                booking.delete()
                
                print(f"Booking updated successfully")
            except Exception as e:
                print(f"An error occurred: {str(e)}")
    return render(request, 'display.html')