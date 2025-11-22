"""
Cloud API Integration Module for WhisperBar
Supports OpenAI Whisper API, Google Speech-to-Text, Deepgram, and Custom APIs
"""

import os
import sys
import time
import tempfile
import requests
from typing import Optional, Dict, Any
import numpy as np
import scipy.io.wavfile as wav

# Cloud API providers
CLOUD_PROVIDERS = {
    "openai": "OpenAI Whisper API",
    "google": "Google Speech-to-Text",
    "deepgram": "Deepgram",
    "custom": "Custom API"
}

class CloudTranscriptionError(Exception):
    """Custom exception for cloud API errors"""
    pass

class CloudAPIBase:
    """Base class for cloud transcription APIs"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.timeout = 30  # seconds
    
    def transcribe(self, audio_data: np.ndarray, sample_rate: int = 16000) -> str:
        """Transcribe audio data using cloud API.
        
        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate in Hz (default 16000)
            
        Returns:
            Transcribed text string
            
        Raises:
            CloudTranscriptionError: If transcription fails
        """
        raise NotImplementedError("Subclasses must implement transcribe()")
    
    def _save_audio_temp(self, audio_data: np.ndarray, sample_rate: int) -> str:
        """Save audio data to temporary WAV file."""
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        temp_path = temp_file.name
        temp_file.close()
        
        # Ensure audio is float32 and normalized
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
        
        # Normalize to prevent clipping
        audio_max = np.abs(audio_data).max()
        if audio_max > 0:
            audio_data = audio_data / audio_max * 0.95
        
        # Convert to int16 for WAV file
        audio_int16 = (audio_data * 32767).astype(np.int16)
        
        wav.write(temp_path, sample_rate, audio_int16)
        return temp_path


class OpenAIWhisperAPI(CloudAPIBase):
    """OpenAI Whisper API implementation"""
    
    def __init__(self, api_key: Optional[str] = None):
        super().__init__(api_key)
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise CloudTranscriptionError("OpenAI API key not found. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
        
        self.api_url = "https://api.openai.com/v1/audio/transcriptions"
        self.model = "whisper-1"
    
    def transcribe(self, audio_data: np.ndarray, sample_rate: int = 16000) -> str:
        """Transcribe using OpenAI Whisper API."""
        try:
            # Save audio to temp file
            temp_path = self._save_audio_temp(audio_data, sample_rate)
            
            try:
                # Prepare request
                with open(temp_path, 'rb') as audio_file:
                    files = {'file': ('audio.wav', audio_file, 'audio/wav')}
                    data = {'model': self.model}
                    headers = {'Authorization': f'Bearer {self.api_key}'}
                    
                    # Make API request
                    response = requests.post(
                        self.api_url,
                        headers=headers,
                        files=files,
                        data=data,
                        timeout=self.timeout
                    )
                    
                    response.raise_for_status()
                    result = response.json()
                    
                    return result.get('text', '').strip()
                    
            finally:
                # Clean up temp file
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                    
        except requests.exceptions.RequestException as e:
            raise CloudTranscriptionError(f"OpenAI API request failed: {e}")
        except Exception as e:
            raise CloudTranscriptionError(f"OpenAI transcription failed: {e}")


class GoogleSpeechToTextAPI(CloudAPIBase):
    """Google Speech-to-Text API implementation"""
    
    def __init__(self, api_key: Optional[str] = None, credentials_path: Optional[str] = None):
        super().__init__(api_key)
        self.api_key = api_key or os.getenv('GOOGLE_API_KEY')
        self.credentials_path = credentials_path or os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
        
        if not self.api_key and not self.credentials_path:
            raise CloudTranscriptionError("Google API key or credentials not found. Set GOOGLE_API_KEY or GOOGLE_APPLICATION_CREDENTIALS.")
        
        try:
            from google.cloud import speech
            self.speech_client = speech.SpeechClient() if self.credentials_path else None
        except ImportError:
            raise CloudTranscriptionError("google-cloud-speech library not installed. Install with: pip install google-cloud-speech")
    
    def transcribe(self, audio_data: np.ndarray, sample_rate: int = 16000) -> str:
        """Transcribe using Google Speech-to-Text API."""
        try:
            from google.cloud import speech
            
            # Save audio to temp file
            temp_path = self._save_audio_temp(audio_data, sample_rate)
            
            try:
                # Read audio file
                with open(temp_path, 'rb') as audio_file:
                    content = audio_file.read()
                
                # Configure recognition
                config = speech.RecognitionConfig(
                    encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                    sample_rate_hertz=sample_rate,
                    language_code='en-US',
                    enable_automatic_punctuation=True,
                )
                
                audio = speech.RecognitionAudio(content=content)
                
                # Use client if available, otherwise use REST API
                if self.speech_client:
                    response = self.speech_client.recognize(config=config, audio=audio)
                else:
                    # Fallback to REST API
                    api_url = f"https://speech.googleapis.com/v1/speech:recognize?key={self.api_key}"
                    payload = {
                        "config": {
                            "encoding": "LINEAR16",
                            "sampleRateHertz": sample_rate,
                            "languageCode": "en-US",
                            "enableAutomaticPunctuation": True
                        },
                        "audio": {
                            "content": content.hex()  # Base64 would be better, but hex works
                        }
                    }
                    response = requests.post(api_url, json=payload, timeout=self.timeout)
                    response.raise_for_status()
                    response = response.json()
                
                # Extract transcription
                if hasattr(response, 'results'):
                    # gRPC response
                    results = response.results
                else:
                    # REST API response
                    results = response.get('results', [])
                
                if results:
                    return results[0].alternatives[0].transcript.strip()
                else:
                    return ""
                    
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                    
        except Exception as e:
            raise CloudTranscriptionError(f"Google Speech-to-Text failed: {e}")


class DeepgramAPI(CloudAPIBase):
    """Deepgram API implementation"""
    
    def __init__(self, api_key: Optional[str] = None):
        super().__init__(api_key)
        self.api_key = api_key or os.getenv('DEEPGRAM_API_KEY')
        if not self.api_key:
            raise CloudTranscriptionError("Deepgram API key not found. Set DEEPGRAM_API_KEY environment variable or pass api_key parameter.")
        
        self.api_url = "https://api.deepgram.com/v1/listen"
    
    def transcribe(self, audio_data: np.ndarray, sample_rate: int = 16000) -> str:
        """Transcribe using Deepgram API."""
        try:
            # Save audio to temp file
            temp_path = self._save_audio_temp(audio_data, sample_rate)
            
            try:
                # Prepare request
                with open(temp_path, 'rb') as audio_file:
                    headers = {
                        'Authorization': f'Token {self.api_key}',
                        'Content-Type': 'audio/wav'
                    }
                    params = {
                        'model': 'nova-2',
                        'language': 'en-US',
                        'punctuate': 'true',
                        'diarize': 'false'
                    }
                    
                    # Make API request
                    response = requests.post(
                        self.api_url,
                        headers=headers,
                        params=params,
                        data=audio_file,
                        timeout=self.timeout
                    )
                    
                    response.raise_for_status()
                    result = response.json()
                    
                    # Extract transcription
                    if 'results' in result and 'channels' in result['results']:
                        transcripts = result['results']['channels'][0].get('alternatives', [])
                        if transcripts:
                            return transcripts[0].get('transcript', '').strip()
                    
                    return ""
                    
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                    
        except requests.exceptions.RequestException as e:
            raise CloudTranscriptionError(f"Deepgram API request failed: {e}")
        except Exception as e:
            raise CloudTranscriptionError(f"Deepgram transcription failed: {e}")


class CustomAPI(CloudAPIBase):
    """Custom API implementation for generic HTTP endpoints"""
    
    def __init__(self, api_url: str, api_key: Optional[str] = None, 
                 headers: Optional[Dict[str, str]] = None,
                 auth_type: str = "bearer"):
        super().__init__(api_key)
        self.api_url = api_url
        self.api_key = api_key
        self.headers = headers or {}
        self.auth_type = auth_type.lower()
        
        if not self.api_url:
            raise CloudTranscriptionError("Custom API URL is required")
    
    def transcribe(self, audio_data: np.ndarray, sample_rate: int = 16000) -> str:
        """Transcribe using custom API endpoint."""
        try:
            # Save audio to temp file
            temp_path = self._save_audio_temp(audio_data, sample_rate)
            
            try:
                # Prepare headers
                headers = self.headers.copy()
                headers['Content-Type'] = 'audio/wav'
                
                # Add authentication
                if self.api_key:
                    if self.auth_type == "bearer":
                        headers['Authorization'] = f'Bearer {self.api_key}'
                    elif self.auth_type == "api_key":
                        headers['X-API-Key'] = self.api_key
                    elif self.auth_type == "basic":
                        import base64
                        auth_str = base64.b64encode(f":{self.api_key}".encode()).decode()
                        headers['Authorization'] = f'Basic {auth_str}'
                
                # Prepare request
                with open(temp_path, 'rb') as audio_file:
                    response = requests.post(
                        self.api_url,
                        headers=headers,
                        data=audio_file,
                        timeout=self.timeout
                    )
                    
                    response.raise_for_status()
                    
                    # Try to parse JSON response
                    try:
                        result = response.json()
                        # Common response formats
                        if isinstance(result, dict):
                            return result.get('text', result.get('transcription', result.get('result', ''))).strip()
                        elif isinstance(result, str):
                            return result.strip()
                    except:
                        # If not JSON, return text response
                        return response.text.strip()
                    
                    return ""
                    
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                    
        except requests.exceptions.RequestException as e:
            raise CloudTranscriptionError(f"Custom API request failed: {e}")
        except Exception as e:
            raise CloudTranscriptionError(f"Custom API transcription failed: {e}")


def get_cloud_provider(provider_name: str, **kwargs) -> CloudAPIBase:
    """Factory function to get cloud API provider instance.
    
    Args:
        provider_name: Name of provider ("openai", "google", "deepgram", "custom")
        **kwargs: Provider-specific arguments
        
    Returns:
        CloudAPIBase instance
    """
    provider_name = provider_name.lower()
    
    if provider_name == "openai":
        return OpenAIWhisperAPI(api_key=kwargs.get('api_key'))
    elif provider_name == "google":
        return GoogleSpeechToTextAPI(
            api_key=kwargs.get('api_key'),
            credentials_path=kwargs.get('credentials_path')
        )
    elif provider_name == "deepgram":
        return DeepgramAPI(api_key=kwargs.get('api_key'))
    elif provider_name == "custom":
        return CustomAPI(
            api_url=kwargs.get('api_url', ''),
            api_key=kwargs.get('api_key'),
            headers=kwargs.get('headers', {}),
            auth_type=kwargs.get('auth_type', 'bearer')
        )
    else:
        raise ValueError(f"Unknown cloud provider: {provider_name}")

