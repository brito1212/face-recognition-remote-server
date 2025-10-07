"""Flask app initialization."""
from flask import Flask

def create_app():
    """Create and configure the Flask application."""
    app = Flask(__name__)
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
    
    from app import routes
    app.register_blueprint(routes.bp)
    
    return app
