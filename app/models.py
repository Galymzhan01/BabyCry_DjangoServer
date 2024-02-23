from django.db import models

# Create your models here.
class AudioRecording(models.Model):
    audio_file = models.FileField(upload_to='audio_uploads/')
    classification_result = models.CharField(max_length=255, null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    image_file = models.ImageField(upload_to='spectrogram_uploads/', null=True, blank=True)

    def __str__(self):
        return f"AudioRecording-{self.id}" 
