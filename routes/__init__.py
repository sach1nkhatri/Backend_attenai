from .video_feed import video_feed_bp
from .register import register_bp
from .recognize import recognize_bp

def register_routes(app):
    app.register_blueprint(video_feed_bp, url_prefix="/video")
    app.register_blueprint(register_bp, url_prefix="/register")
    app.register_blueprint(recognize_bp, url_prefix="/recognize")
