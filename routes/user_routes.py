from flask import Blueprint, request, jsonify
from .models import User
from controllers.user_controller import UserController

# Create a blueprint for user routes
user_bp = Blueprint("user", __name__, url_prefix="/api/users")


@user_bp.route("/", methods=["GET"])
def get_all_users():
    """Get all users"""
    users = User.query.all()
    return jsonify([user.as_dict for user in users]), 200


@user_bp.route("/<int:user_id>", methods=["GET"])
def get_user(user_id):
    """Get a user by ID"""
    user = UserController.get_user_by_id(user_id)
    if not user:
        return jsonify({"error": "User not found"}), 404
    return jsonify(user.as_dict), 200


@user_bp.route("/", methods=["POST"])
def create_user():
    """Create a new user"""
    data = request.get_json()
    user = UserController.create_user(data)
    return jsonify(user.as_dict), 201


@user_bp.route("/<int:user_id>", methods=["PUT"])
def update_user(user_id):
    """Update a user"""
    data = request.get_json()
    user = UserController.update_user(user_id, data)
    if not user:
        return jsonify({"error": "User not found"}), 404
    return jsonify(user.as_dict), 200


@user_bp.route("/<int:user_id>", methods=["DELETE"])
def delete_user(user_id):
    """Delete a user"""
    success = UserController.delete_user(user_id)
    if not success:
        return jsonify({"error": "User not found"}), 404
    return jsonify({"message": "User deleted successfully"}), 200
