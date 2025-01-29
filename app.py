from flask import Flask
from flask_cors import CORS
from routes import register_routes  # Ensure this file exists and contains `register_bp`

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Register all routes
register_routes(app)

if __name__ == '__main__':
    app.run(debug=True)
