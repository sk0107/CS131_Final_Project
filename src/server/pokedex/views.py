from django.views.decorators.csrf import csrf_exempt
from django.http import HttpResponse, JsonResponse
from django.conf import settings
import os
import shutil

from .inference import inference

import pandas as pd

PATH = 'pokemon.csv'

@csrf_exempt
def upload_file(request):
    if request.method == 'POST':
        uploaded_file = request.FILES['file']
        save_path = os.path.join(settings.MEDIA_ROOT, 'inputs', uploaded_file.name)
        with open(save_path, 'wb+') as destination:
            for chunk in uploaded_file.chunks():
                destination.write(chunk)

        species = inference()

        df = pd.read_csv(PATH)
        row = df[df['name'] == species]
        description = row.iloc[0][1]
        print(description)

        # wipe files in directory
        directory = os.path.join(settings.MEDIA_ROOT, 'inputs')

        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    pass
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")

        return JsonResponse({ 'species': species, 'description': description })
    return HttpResponse('Failed to upload image')
