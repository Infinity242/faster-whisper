import runpod
from faster_whisper import WhisperModel

# Load the model once when the container starts
model = WhisperModel("large-v3", device="cuda", compute_type="float16")

def handler(job):
    """Handle RunPod serverless requests."""
    job_input = job["input"]  # Input from RunPod (e.g., {"audio_url": "https://example.com/audio.mp3"})
    
    # Example: Expecting an audio URL in the input
    audio_url = job_input.get("audio_url")
    if not audio_url:
        return {"error": "No audio_url provided"}

    # Transcribe (assuming the model can handle URLs or you download the file)
    segments, info = model.transcribe(audio_url, beam_size=5)
    
    # Format output
    result = {
        "language": info.language,
        "transcription": [{"start": segment.start, "end": segment.end, "text": segment.text} for segment in segments]
    }
    
    return result

# Start the RunPod serverless endpoint
runpod.serverless.start({"handler": handler})
