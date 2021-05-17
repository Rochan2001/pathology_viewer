from django.shortcuts import render
from django.http import HttpResponse
from django.template import loader
from django.core.files.storage import FileSystemStorage
from django.conf import settings
import os
import pyvips

def index(request):

    context = {
        'val': 20,
    }

    return render(request, 'pathology_viewer/index.html', context)

def wsi_upload(request):

    media_url = settings.MEDIA_URL

    if request.method == 'POST' and request.FILES['wsi']:

        myfile = request.FILES['wsi']
        fs = FileSystemStorage()
        filename = fs.save(myfile.name, myfile)
        uploaded_file_url = fs.url(filename)
        wsi_location = media_url+filename 
        abs_path = os.path.abspath("../")+r"\pathology_project\media"
        wsi_path = abs_path+'\\'+filename
        #image = pyvips.Image.new_from_file(wsi_path,access='sequential')
        #image.dzsave(wsi_path,tile_height=256,tile_width=256,overlap=0)
        path = settings.MEDIA_ROOT
        #img_list = os.listdir(abs_path)
        context = {'uploaded_file_url': uploaded_file_url}

        return render(request, 'pathology_viewer/viewer.html',context)

    return render(request, 'pathology_viewer/viewer.html')
