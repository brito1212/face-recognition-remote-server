"""Simple test script to verify the Flask app can be initialized."""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_app_creation():
    """Test that the Flask app can be created."""
    from app import create_app
    
    app = create_app()
    assert app is not None
    print("✓ Flask app created successfully")
    
    # Check routes are registered
    with app.app_context():
        routes = [str(rule) for rule in app.url_map.iter_rules()]
        assert '/recognize' in str(routes)
        assert '/health' in str(routes)
        print("✓ Routes registered correctly")
    
    return True

def test_health_endpoint():
    """Test the health endpoint."""
    from app import create_app
    
    app = create_app()
    client = app.test_client()
    
    response = client.get('/health')
    assert response.status_code == 200
    data = response.get_json()
    assert data['status'] == 'healthy'
    print("✓ Health endpoint working")
    
    return True

def test_recognize_endpoint_no_file():
    """Test the recognize endpoint without a file."""
    from app import create_app
    
    app = create_app()
    client = app.test_client()
    
    response = client.post('/recognize')
    assert response.status_code == 400
    data = response.get_json()
    assert 'error' in data
    assert data['recognized'] is False
    print("✓ Recognize endpoint validates file presence")
    
    return True

if __name__ == '__main__':
    print("Running basic tests...\n")
    
    try:
        test_app_creation()
        test_health_endpoint()
        test_recognize_endpoint_no_file()
        
        print("\n✓ All basic tests passed!")
        print("\nNote: Full functionality requires a TensorFlow model in models/face_model.zip")
        
    except Exception as e:
        print(f"\n✗ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
