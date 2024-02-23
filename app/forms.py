# forms.py

from django import forms
from .models import AudioRecording

class AudioRecordingForm(forms.ModelForm):
    audio_data = forms.CharField(widget=forms.HiddenInput(), required=False)

    class Meta:
        model = AudioRecording
        fields = ['audio_file', 'audio_data']
