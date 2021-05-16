from django.shortcuts import render
from django.http import HttpResponse
from django.template import loader
from django.core.files.storage import FileSystemStorage
from django.conf import settings
import os

def index(request):

    context = {
        'val': 20,
    }

    return render(request, 'pathology_viewer/index.html', context)

def wsi_upload(request):
    if request.method == 'POST' and request.FILES['wsi']:
        myfile = request.FILES['wsi']
        fs = FileSystemStorage()
        filename = fs.save(myfile.name, myfile)
        uploaded_file_url = fs.url(filename)
        path = settings.MEDIA_ROOT
        img_list = os.listdir(path)
        print(path)
        context = {'images' : img_list,'uploaded_file_url': uploaded_file_url}

        return render(request, 'pathology_viewer/viewer.html',context)

    return render(request, 'pathology_viewer/viewer.html')
