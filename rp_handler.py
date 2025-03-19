import runpod
from faster_whisper import WhisperModel
import base64
import tempfile
import os

# Load the model once when the container starts
model = WhisperModel("large-v3", device="cuda", compute_type="float16")

def handler(job):
    """Handle RunPod serverless requests."""
    job_input = job["input"]

    # Expecting base64-encoded audio
    audio_base64 = job_input.get("audio_base64")
    if not audio_base64:
        return {"error": "No audio_base64 provided"}

    # Decode base64 and write to a temporary file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        audio_data = base64.b64decode(audio_base64)
        temp_file.write(audio_data)
        temp_file_path = temp_file.name

    try:
        # Transcribe the audio file
        segments, info = model.transcribe(temp_file_path, beam_size=5)
        result = {
            "language": info.language,
            "transcription": [{"start": segment.start, "end": segment.end, "text": segment.text} for segment in segments]
        }
    finally:
        # Clean up the temporary file
        os.unlink(temp_file_path)

    return result

# Start the RunPod serverless endpoint
runpod.serverless.start({"handler": handler})
