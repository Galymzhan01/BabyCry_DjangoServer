# views.py

# import io, os
# import requests
# from pydub import AudioSegment
# from django.core.files.base import ContentFile
# from django.core.files import File
from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse
import os
# from .forms import AudioRecordingForm
# from .models import AudioRecording
import joblib
import numpy as np
import librosa
# import tempfile
# import matplotlib
# import matplotlib.pyplot as plt
# from PIL import Image
# import torch.nn as nn
# import torch
# import torchvision.transforms as transforms


# class Conf:
#   sampling_rate = 16000
#   duration = 7
#   hop_length = 100*duration
#   fmin = 20
#   fmax = sampling_rate//2
#   n_mels = 128
#   n_fft = n_mels*20
#   samples = sampling_rate * duration

# Load the KNN model
# knn_model = joblib.load('/Users/galymzan/Downloads/cs409/baby-cry-main/core/load_KNN.pkl')
binary_clf = joblib.load("core/models/binary_classifier.pkl")
# dnn_clf = joblib.load('/Users/galymzan/Downloads/cs409/baby-cry-main/core/dnn.pkl')
# dnn_clf.eval()
# conf = Conf()
# def main_page(request):
#     if request.method == 'POST':
#         print("accepted a sound!")
#         form = AudioRecordingForm(request.POST, request.FILES)
#         if form.is_valid():
#             print("Valid")
#             audio_data_url = form.cleaned_data['audio_data']
#             print()
#             audio_data = fetch_audio_data(audio_data_url)
#             audio_file = convert_and_save_to_wav(audio_data)

#             # Save the AudioRecording instance
#             audio_recording = form.save(commit=False)
#             audio_recording.audio_file.save('audio.wav', audio_file)
#             #audio_recording.save()

#             classification_result = classify_audio(audio_file.path)
#             audio_recording.classification_result = classification_result
#             audio_recording.save()
#             print("Done!")
#             return redirect('result_page', pk=audio_recording.pk)
#         else:
#             print("problems!")
#             print("Form errors:", form.errors)
#     else:
#         form = AudioRecordingForm()

#     return render(request, 'main_page.html', {'form': form})
def main_simple(request):
    return render(request, 'main_simple.html')
            
# def main_page(request):
#     if request.method == 'POST':
#         form = AudioRecordingForm(request.POST, request.FILES)
#         if form.is_valid():
#             # Assuming 'audio_file' is the name of the file field in your form
#             audio_file = request.FILES['audio_file']
#             audio_recording = form.save(commit=False)
#             audio_recording.audio_file = audio_file
            
#             with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
#                 for chunk in audio_file.chunks():
#                     tmp.write(chunk)

#                 # Get the path of the saved temporary file
#                 temp_file_path = tmp.name

#             # Process the audio file as needed, then save
#             features = features_extractor_binary(temp_file_path)
#             features = features.reshape(1, -1)

#             isBaby = binary_clf.predict(features)[0]
#             if isBaby == 'NotBC_training':
#                 audio_recording.classification_result = 'NotBC_training'
#                 audio_recording.save()
#             else: 
#                 print(isBaby)
#                 # transformed_sound = read_audio(temp_file_path, True)
#                 # mel_img = audio_to_melspectrogram(transformed_sound)
#                 save_image_from_sound(temp_file_path, audio_recording)
#                 predictions = ["hungry", "burping", "discomfort", "belly_pain", "tired"]

#                 img = Image.open(audio_recording.image_file).convert('RGB')
#                 img_tensor = preprocess_image_for_dnn(img) 

#                 # Inference with DNN
#                 output = dnn_clf(img_tensor)  
#                 prob = nn.functional.softmax(output, dim=1)
#                 _, predicted = torch.max(output, 1)  

#                 # Save classified result (assuming 'predicted' is a tensor)
#                 audio_recording.classification_result = predictions[predicted.item()] 
#                 audio_recording.save()


#                 #classification_result = "hungry"
#                 # audio_recording.classification_result = classification_result[0]
#                 # audio_recording.classification_result = predicted  # Store as text based on classes 
#                 # audio_recording.save()

#             os.remove(temp_file_path)
#             # Add classification logic as needed
            
#             return redirect('result_page', pk=audio_recording.pk)
#         else:
#             print("Form errors:", form.errors)
#     else:
#         form = AudioRecordingForm()

#     return render(request, 'main_page.html', {'form': form})


def audio_upload_from_flutter(request):
    if request.method == 'POST':
        audio_file = request.FILES['file'] 

        # Feature Extraction
        features = features_extractor_binary(audio_file)  

        # Prediction from your KNN Model (Let me know if you need help with this part)
        isBaby = binary_clf.predict(features)[0] 

        # Return prediction as JSON
        return JsonResponse({'prediction': isBaby}) 
    else:
        return redirect('main_page')  # Redirect to main page if not a POST request


# def result_page(request, pk):
#     audio_recording = get_object_or_404(AudioRecording, pk=pk)
#     # You can pass the classification result to the template
#     return render(request, 'result_page.html', {'audio_recording': audio_recording})

# def fetch_audio_data(audio_data_url):
#     response = requests.get(audio_data_url)
#     return response.content

# def convert_and_save_to_wav(audio_data):
#     audio = AudioSegment.from_file(io.BytesIO(audio_data))
#     wav_data = io.BytesIO()
#     audio.export(wav_data, format='wav')
#     return ContentFile(wav_data.getvalue(), 'audio.wav')

# def classify_audio(audio_path):
#     features = extract_features(audio_path)
#     # prediction = knn_model.predict(features.reshape(1, -1))
#     prediction = knn_model.predict(features.reshape(1, -1))
#     return prediction

# def extract_features(audio_path):
#     audio, sample_rate = librosa.load(audio_path, res_type='kaiser_fast')
#     mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
#     mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)

#     print(mfccs_scaled_features.shape)
#     return mfccs_scaled_features

def features_extractor_binary(file_name):
    audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)

    return mfccs_scaled_features

# def read_audio(pathname, trim_long_data, conf=conf):
#   y, sr = librosa.load(pathname, sr=conf.sampling_rate)

#   #trim silence
#   if 0<len(y): #workaround: 0 length causes error
#     y, _ = librosa.effects.trim(y) # trim, top_db=default(60)


#   #make it unified length to conf.samples
#   if len(y) > conf.samples: #long enough
#     if trim_long_data:
#       y = y[0:conf.samples]
#   else:
#     padding = conf.samples - len(y)
#     offset = padding//2
#     y = np.pad(y, (offset, conf.samples-len(y)-offset), mode = 'constant')

#   return y


# def audio_to_melspectrogram(audio, conf=conf):
#   spectrogram = librosa.feature.melspectrogram(y = audio,
#                                              sr = conf.sampling_rate,
#                                              n_mels = conf.n_mels,
#                                              hop_length = conf.hop_length,
#                                              n_fft = conf.n_fft,
#                                              fmin = conf.fmin,
#                                              fmax = conf.fmax)
#   spectrogram = librosa.power_to_db(spectrogram)
#   spectrogram = spectrogram.astype(np.float128)
  
#   return spectrogram

# def save_image_from_sound(path, audio_recording):
#   y = read_audio(path, True)
#   spectrogram = audio_to_melspectrogram(y)
  
#   matplotlib.use('Agg')

#   buffer = io.BytesIO()

#   plt = matplotlib.pyplot 
#   plt.figure(figsize=(10, 4))
#   librosa.display.specshow(spectrogram, sr=conf.sampling_rate, hop_length=conf.hop_length, x_axis='time', y_axis='mel')
#   plt.title('Mel spectrogram')
#   plt.savefig(buffer, format='png')  # Save the plot to the buffer
#   plt.close()

#   buffer.seek(0)

#   image = Image.open(buffer)
#   audio_recording.image_file.save(f"melimg{audio_recording.pk}.jpeg", File(buffer), save=True)


# def preprocess_image_for_dnn(img):
#     transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
#     ])
#     img_tensor = transform(img) 
#     img_tensor = img_tensor.unsqueeze(0)  # Add the batch dimension
#     return img_tensor
