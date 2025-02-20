from flask import Flask, request, jsonify
from markupsafe import escape  # Import escape
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

app = Flask(__name__)
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["200 per day", "50 per hour"]
)

@app.route('/search')
def search():
    query = request.args.get('q')
    # Sanitize the input using escape
    safe_query = escape(query)
    # ... use safe_query in your search logic ...
    return jsonify({"result": f"Searching for: {safe_query}"})

@app.route('/api/resource')
@limiter.limit("10/minute")  # Limit this specific route
def my_resource():
    return jsonify({"data": "Some data"}) 