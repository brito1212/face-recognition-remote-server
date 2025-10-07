"""Example script demonstrating how to use the face recognition server."""
import requests
import sys

def test_server(image_path, server_url="http://localhost:5000"):
    """
    Test the face recognition server with an image.
    
    Args:
        image_path: Path to the .jpg image file
        server_url: URL of the server (default: http://localhost:5000)
    """
    # Check server health
    try:
        health_response = requests.get(f"{server_url}/health")
        if health_response.status_code == 200:
            print("✓ Server is healthy")
        else:
            print("✗ Server health check failed")
            return
    except requests.exceptions.ConnectionError:
        print("✗ Cannot connect to server. Make sure it's running on", server_url)
        return
    
    # Send image for recognition
    try:
        with open(image_path, 'rb') as image_file:
            files = {'file': image_file}
            response = requests.post(f"{server_url}/recognize", files=files)
        
        if response.status_code == 200:
            data = response.json()
            print("\n--- Recognition Result ---")
            print(f"Recognized: {data['recognized']}")
            print(f"Identity: {data['identity']}")
            if 'message' in data:
                print(f"Message: {data['message']}")
        else:
            print(f"✗ Request failed with status code {response.status_code}")
            print(f"Response: {response.json()}")
            
    except FileNotFoundError:
        print(f"✗ Image file not found: {image_path}")
    except Exception as e:
        print(f"✗ Error: {str(e)}")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python example_usage.py <path_to_image.jpg> [server_url]")
        print("\nExample:")
        print("  python example_usage.py test_image.jpg")
        print("  python example_usage.py test_image.jpg http://localhost:5000")
        sys.exit(1)
    
    image_path = sys.argv[1]
    server_url = sys.argv[2] if len(sys.argv) > 2 else "http://localhost:5000"
    
    test_server(image_path, server_url)
