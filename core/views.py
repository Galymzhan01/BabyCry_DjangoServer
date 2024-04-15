# views.py

import io, os
from django.core.files import File
from django.shortcuts import render, redirect, get_object_or_404
from rest_framework.decorators import api_view
from django.views.decorators.csrf import ensure_csrf_cookie
from rest_framework.response import Response
import joblib
import numpy as np
import librosa

import pandas as pd
# import tempfile




knn_model = joblib.load('core/models/best_model.pkl')
binary_clf = joblib.load("core/models/binary_classifier.pkl")



@ensure_csrf_cookie
def main_simple(request):
    return render(request, 'main_simple.html')

@api_view(['POST'])
def audio_upload_from_flutter(request):
    if request.method == 'POST':
        audio_file = request.FILES['file'] 
        temp_file_path = 'temporary_audio.wav'

        with open(temp_file_path, 'wb+') as destination: 
            for chunk in audio_file.chunks():
                destination.write(chunk)
        # Feature Extraction
        features = features_extractor_binary(temp_file_path)  
        features = features.reshape(1, -1)

        isBaby = binary_clf.predict(features)[0] 
        # Return prediction as JSON
        if isBaby == 'NotBC_training':
          os.remove(temp_file_path)
          print("Not a cry!")
          return Response({'prediction': isBaby}) 
        else: 
            prediction = knn_model.predict(features)[0]
            os.remove(temp_file_path)
            print(prediction)
            return Response({'prediction': prediction}) 
            
    else:
        return redirect('main_simple')  # Redirect to main page if not a POST request




def features_extractor_binary(file_name):
    audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)

    return mfccs_scaled_features
