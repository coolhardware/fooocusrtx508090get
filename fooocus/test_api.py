#!/usr/bin/env python3
"""
Test script for Fooocus API endpoints
Usage: python test_api.py
"""

import requests
import time
import json
from urllib.parse import urlencode

# Configuration
BASE_URL = "http://localhost:7865"  # Default Fooocus URL
USERNAME = "sitting-duck-1"  # From auth-example.json
PASSWORD = "very-bad-publicly-known-password-change-it"

def test_status_endpoint():
    """Test the status endpoint"""
    print("Testing /api/status endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/api/status")
        if response.status_code == 200:
            data = response.json()
            print("✅ Status endpoint working")
            print(f"   Version: {data.get('version')}")
            print(f"   Available models: {len(data.get('available_models', []))}")
            print(f"   Available styles: {len(data.get('available_styles', []))}")
            return True
        else:
            print(f"❌ Status endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Status endpoint error: {e}")
        return False

def test_generate_simple():
    """Test simple generation via GET parameters"""
    print("\nTesting simple generation...")
    
    params = {
        'prompt': 'a beautiful sunset over mountains',
        'steps': 10,  # Low steps for faster testing
        'image_number': 1
    }
    
    url = f"{BASE_URL}/generate?" + urlencode(params)
    print(f"Request URL: {url}")
    
    try:
        # Add authentication if needed
        auth = (USERNAME, PASSWORD) if USERNAME and PASSWORD else None
        
        response = requests.get(url, auth=auth, timeout=120)
        
        if response.status_code == 200:
            if response.headers.get('content-type', '').startswith('image/'):
                print("✅ Simple generation successful - received image")
                # Save the image
                with open('test_output.png', 'wb') as f:
                    f.write(response.content)
                print("   Image saved as test_output.png")
                return True
            else:
                print("✅ Generation successful - received JSON response")
                try:
                    data = response.json()
                    print(f"   Response: {json.dumps(data, indent=2)}")
                except:
                    print(f"   Response text: {response.text[:200]}...")
                return True
        else:
            print(f"❌ Generation failed: {response.status_code}")
            print(f"   Response: {response.text[:200]}...")
            return False
            
    except requests.exceptions.Timeout:
        print("❌ Generation timed out")
        return False
    except Exception as e:
        print(f"❌ Generation error: {e}")
        return False

def test_generate_advanced():
    """Test advanced generation with more parameters"""
    print("\nTesting advanced generation...")
    
    params = {
        'prompt': 'a cyberpunk city at night, neon lights, futuristic',
        'negative_prompt': 'blurry, low quality',
        'steps': 15,
        'cfg_scale': 7.0,
        'width': 512,
        'height': 512,
        'performance': 'Speed',
        'output_format': 'png'
    }
    
    url = f"{BASE_URL}/generate?" + urlencode(params)
    
    try:
        auth = (USERNAME, PASSWORD) if USERNAME and PASSWORD else None
        response = requests.get(url, auth=auth, timeout=180)
        
        if response.status_code == 200:
            print("✅ Advanced generation successful")
            return True
        else:
            print(f"❌ Advanced generation failed: {response.status_code}")
            print(f"   Response: {response.text[:200]}...")
            return False
    except Exception as e:
        print(f"❌ Advanced generation error: {e}")
        return False

def test_error_handling():
    """Test API error handling"""
    print("\nTesting error handling...")
    
    # Test with empty prompt
    params = {'prompt': ''}
    url = f"{BASE_URL}/generate?" + urlencode(params)
    
    try:
        auth = (USERNAME, PASSWORD) if USERNAME and PASSWORD else None
        response = requests.get(url, auth=auth)
        
        if response.status_code == 400:
            print("✅ Error handling working - empty prompt rejected")
            return True
        else:
            print(f"❌ Error handling failed: expected 400, got {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error handling test error: {e}")
        return False

def main():
    """Run all tests"""
    print("Fooocus API Test Suite")
    print("=" * 50)
    
    tests = [
        test_status_endpoint,
        test_generate_simple,
        test_generate_advanced,
        test_error_handling
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        time.sleep(1)  # Brief pause between tests
    
    print(f"\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! API is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the implementation.")
        
    print("\nExample usage URLs:")
    print(f"Status: {BASE_URL}/api/status")
    print(f"Simple: {BASE_URL}/generate?prompt=a+cat&steps=20")
    print(f"Advanced: {BASE_URL}/generate?prompt=a+dog&negative_prompt=blurry&steps=25&cfg_scale=7.5")

if __name__ == "__main__":
    main()