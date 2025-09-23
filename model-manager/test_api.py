#!/usr/bin/env python3

"""
Test script for the Model Manager API
"""

import requests
import time

def test_health():
    """Test health endpoint"""
    print("🔍 Testing health endpoint...")
    try:
        response = requests.get("http://localhost:8000/health")
        print(f"✅ Health check: {response.status_code}")
        print(f"   Response: {response.json()}")
        return True
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

def test_text_translation():
    """Test text-to-text translation"""
    print("\n🔍 Testing text translation...")
    try:
        data = {
            "model": "opus-mt",
            "sourceLang": "en", 
            "targetLang": "es",
            "text": "Hello, how are you today?",
            "options": {}
        }
        response = requests.post("http://localhost:8000/translate/text", json=data)
        print(f"✅ Text translation: {response.status_code}")
        result = response.json()
        print(f"   Input: {data['text']}")
        print(f"   Output: {result['translation']['text']}")
        print(f"   Confidence: {result['translation']['confidence']}")
        return True
    except Exception as e:
        print(f"❌ Text translation failed: {e}")
        return False

def test_speech_synthesis():
    """Test text-to-speech synthesis"""
    print("\n🔍 Testing speech synthesis...")
    try:
        data = {
            "model": "espeak-ng",
            "text": "Hello world",
            "language": "en",
            "options": {}
        }
        response = requests.post("http://localhost:8000/synthesize", json=data)
        print(f"✅ Speech synthesis: {response.status_code}")
        result = response.json()
        print(f"   Input: {data['text']}")
        print(f"   Audio format: {result['synthesis']['format']}")
        print(f"   Duration: {result['synthesis']['duration']}s")
        print(f"   Audio size: {len(result['synthesis']['audioBase64'])} chars")
        return True
    except Exception as e:
        print(f"❌ Speech synthesis failed: {e}")
        return False

def test_audio_transcription():
    """Test speech-to-text transcription"""
    print("\n🔍 Testing audio transcription...")
    try:
        # Create a small dummy audio file
        dummy_audio = b"RIFF" + b"\x00" * 100  # Minimal dummy WAV
        
        files = {
            'audio': ('test.wav', dummy_audio, 'audio/wav')
        }
        data = {
            'model': 'whisper-base',
            'language': 'auto',
            'task': 'transcribe'
        }
        
        response = requests.post("http://localhost:8000/transcribe", files=files, data=data)
        print(f"✅ Audio transcription: {response.status_code}")
        result = response.json()
        print(f"   Transcribed text: {result['transcription']['text']}")
        print(f"   Confidence: {result['transcription']['confidence']}")
        print(f"   Language: {result['transcription']['language']}")
        return True
    except Exception as e:
        print(f"❌ Audio transcription failed: {e}")
        return False

def test_batch_processing():
    """Test batch processing"""
    print("\n🔍 Testing batch processing...")
    try:
        data = {
            "batchId": "test_batch_001",
            "requests": [
                {
                    "id": "req_001",
                    "type": "translate",
                    "model": "opus-mt",
                    "sourceLang": "en",
                    "targetLang": "es", 
                    "text": "Hello"
                },
                {
                    "id": "req_002",
                    "type": "translate",
                    "model": "opus-mt",
                    "sourceLang": "en",
                    "targetLang": "fr",
                    "text": "World"
                }
            ]
        }
        response = requests.post("http://localhost:8000/batch/process", json=data)
        print(f"✅ Batch processing: {response.status_code}")
        result = response.json()
        print(f"   Batch ID: {result['batchId']}")
        print(f"   Total requests: {result['summary']['totalRequests']}")
        print(f"   Successful: {result['summary']['successfulRequests']}")
        return True
    except Exception as e:
        print(f"❌ Batch processing failed: {e}")
        return False

def test_pipelines():
    """Test pipeline endpoints"""
    print("\n🔍 Testing pipeline endpoints...")
    try:
        # Get all pipelines
        response = requests.get("http://localhost:8000/pipelines")
        print(f"✅ Get pipelines: {response.status_code}")
        pipelines = response.json()
        print(f"   Found {len(pipelines)} pipelines")
        
        # Get specific pipeline
        if pipelines:
            pipeline_id = pipelines[0]['id']
            response = requests.get(f"http://localhost:8000/pipelines/{pipeline_id}")
            print(f"✅ Get pipeline {pipeline_id}: {response.status_code}")
            pipeline = response.json()
            print(f"   Pipeline: {pipeline['name']}")
        
        return True
    except Exception as e:
        print(f"❌ Pipeline endpoints failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Model Manager API Test Suite")
    print("=" * 50)
    
    tests = [
        test_health,
        test_pipelines,
        test_text_translation,
        test_speech_synthesis,
        test_audio_transcription,
        test_batch_processing
    ]
    
    results = []
    for test in tests:
        results.append(test())
        time.sleep(0.5)  # Small delay between tests
    
    print("\n" + "=" * 50)
    print("📊 Test Summary")
    print(f"✅ Passed: {sum(results)}/{len(results)}")
    print(f"❌ Failed: {len(results) - sum(results)}/{len(results)}")
    
    if all(results):
        print("\n🎉 All tests passed! Model Manager is working correctly.")
    else:
        print("\n⚠️  Some tests failed. Check the logs above.")

if __name__ == "__main__":
    main()