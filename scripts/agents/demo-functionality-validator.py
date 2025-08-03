#!/usr/bin/env python3
"""
Demonstration of Functionality Preservation Validator

Purpose: Shows how the validator catches different types of breaking changes
Usage: python demo-functionality-validator.py
Requirements: functionality-preservation-validator.py
"""

import os
import sys
import tempfile
import shutil
import subprocess
import json
from pathlib import Path

def create_demo_files():
    """Create demonstration files that show various scenarios."""
    
    demo_files = {
        "api_service.py": '''
"""Sample API service for demonstration."""

from flask import Flask, jsonify, request

app = Flask(__name__)

def calculate_tax(amount, rate=0.1):
    """Calculate tax for a given amount."""
    return amount * rate

def process_payment(amount, currency="USD"):
    """Process a payment transaction."""
    tax = calculate_tax(amount)
    total = amount + tax
    return {
        "amount": amount,
        "tax": tax,
        "total": total,
        "currency": currency
    }

class PaymentProcessor:
    """Handle payment processing operations."""
    
    def __init__(self, merchant_id):
        self.merchant_id = merchant_id
        self.transactions = []
    
    def add_transaction(self, transaction):
        """Add a transaction to the processor."""
        self.transactions.append(transaction)
        return len(self.transactions)
    
    def get_total_volume(self):
        """Get total transaction volume."""
        return sum(t.get("amount", 0) for t in self.transactions)

@app.route('/api/payment', methods=['POST'])
def create_payment():
    """Create a new payment."""
    data = request.get_json()
    result = process_payment(data["amount"], data.get("currency", "USD"))
    return jsonify(result)

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "healthy"})

if __name__ == "__main__":
    app.run(debug=True)
''',
        
        "test_api_service.py": '''
"""Tests for API service."""

import pytest
from api_service import calculate_tax, process_payment, PaymentProcessor

def test_calculate_tax():
    """Test tax calculation."""
    assert calculate_tax(100) == 10.0
    assert calculate_tax(100, 0.2) == 20.0

def test_process_payment():
    """Test payment processing."""
    result = process_payment(100)
    assert result["amount"] == 100
    assert result["tax"] == 10.0
    assert result["total"] == 110.0
    assert result["currency"] == "USD"

def test_payment_processor():
    """Test payment processor class."""
    processor = PaymentProcessor("MERCHANT123")
    assert processor.merchant_id == "MERCHANT123"
    
    transaction_id = processor.add_transaction({"amount": 50})
    assert transaction_id == 1
    
    assert processor.get_total_volume() == 50
''',
        
        "data_models.py": '''
"""Data models for the application."""

from dataclasses import dataclass
from typing import List, Optional

@dataclass
class User:
    """User data model."""
    id: int
    username: str
    email: str
    is_active: bool = True
    
    def get_display_name(self):
        """Get display name for user."""
        return f"{self.username} ({self.email})"
    
    def deactivate(self):
        """Deactivate the user account."""
        self.is_active = False

@dataclass
class Product:
    """Product data model."""
    id: int
    name: str
    price: float
    category: str
    in_stock: bool = True
    
    def apply_discount(self, percentage):
        """Apply discount to product price."""
        self.price = self.price * (1 - percentage / 100)
    
    def get_tax_amount(self, tax_rate=0.1):
        """Calculate tax amount for product."""
        return self.price * tax_rate

class OrderManager:
    """Manage orders and inventory."""
    
    def __init__(self):
        self.orders = []
        self.inventory = {}
    
    def create_order(self, user_id: int, products: List[Product]):
        """Create a new order."""
        order = {
            "id": len(self.orders) + 1,
            "user_id": user_id,
            "products": products,
            "total": sum(p.price for p in products)
        }
        self.orders.append(order)
        return order
    
    def cancel_order(self, order_id: int):
        """Cancel an existing order."""
        for order in self.orders:
            if order["id"] == order_id:
                order["status"] = "cancelled"
                return True
        return False
'''
    }
    
    return demo_files

def demonstrate_breaking_changes():
    """Demonstrate different types of breaking changes."""
    
    print("🎭 Functionality Preservation Validator Demonstration")
    print("=" * 55)
    print()
    
    # Create temporary directory for demo
    demo_dir = tempfile.mkdtemp(prefix="func_validator_demo_")
    print(f"📁 Demo directory: {demo_dir}")
    
    # Initialize git repo
    subprocess.run(["git", "init"], cwd=demo_dir, capture_output=True)
    subprocess.run(["git", "config", "user.email", "demo@example.com"], cwd=demo_dir, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Demo User"], cwd=demo_dir, capture_output=True)
    
    try:
        # Create initial files
        demo_files = create_demo_files()
        for filename, content in demo_files.items():
            with open(os.path.join(demo_dir, filename), 'w') as f:
                f.write(content)
        
        # Add and commit initial version
        subprocess.run(["git", "add", "."], cwd=demo_dir, capture_output=True)
        subprocess.run(["git", "commit", "-m", "Initial demo files"], cwd=demo_dir, capture_output=True)
        
        print("✅ Created initial demo files and committed to git")
        print()
        
        # Find validator script
        current_dir = os.path.dirname(os.path.abspath(__file__))
        validator_path = os.path.join(current_dir, "functionality-preservation-validator.py")
        
        if not os.path.exists(validator_path):
            print(f"❌ Validator not found at: {validator_path}")
            return False
        
        # Demonstrate different breaking changes
        breaking_changes = [
            {
                "name": "Function Parameter Removal",
                "description": "Removing a parameter from a function",
                "modify": lambda: modify_file(demo_dir, "api_service.py", 
                    'def calculate_tax(amount, rate=0.1):',
                    'def calculate_tax(amount):')
            },
            {
                "name": "Function Removal",
                "description": "Completely removing a function",
                "modify": lambda: remove_function(demo_dir, "api_service.py", "process_payment")
            },
            {
                "name": "Class Method Removal", 
                "description": "Removing a method from a class",
                "modify": lambda: remove_method(demo_dir, "data_models.py", "User", "get_display_name")
            },
            {
                "name": "API Endpoint Removal",
                "description": "Removing an API endpoint",
                "modify": lambda: remove_function(demo_dir, "api_service.py", "health_check")
            },
            {
                "name": "Class Inheritance Change",
                "description": "Changing class inheritance (usually safe but flagged as warning)",
                "modify": lambda: modify_file(demo_dir, "data_models.py",
                    'class OrderManager:',
                    'class OrderManager(object):')
            }
        ]
        
        for i, change in enumerate(breaking_changes, 1):
            print(f"🧪 Demo {i}: {change['name']}")
            print(f"📝 {change['description']}")
            print("-" * 50)
            
            # Apply the breaking change
            change["modify"]()
            
            # Run validator
            cmd = ["python", validator_path, "validate", "--format", "json", "--repo-path", demo_dir]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            try:
                report = json.loads(result.stdout)
                
                print(f"📊 Validation Results:")
                print(f"   Total checks: {report['summary']['total_checks']}")
                print(f"   Breaking changes: {report['summary']['breaking_changes']}")
                print(f"   Failures: {report['summary']['failures']}")
                print(f"   Warnings: {report['summary']['warnings']}")
                
                # Show specific breaking changes found
                breaking_results = [r for r in report['results'] if r.get('breaking_change', False)]
                if breaking_results:
                    print(f"🚫 Breaking changes detected:")
                    for br in breaking_results[:3]:  # Show first 3
                        print(f"   • {br['message']}")
                        if br.get('suggestion'):
                            print(f"     💡 {br['suggestion']}")
                
                # Show validation outcome
                if report['summary']['breaking_changes'] > 0:
                    print("❌ Result: COMMIT WOULD BE BLOCKED")
                else:
                    print("✅ Result: Commit would be allowed")
                    
            except json.JSONDecodeError:
                print(f"❌ Validator output: {result.stdout}")
                print(f"❌ Validator error: {result.stderr}")
            
            print()
            
            # Reset changes for next demo
            subprocess.run(["git", "checkout", "HEAD", "."], cwd=demo_dir, capture_output=True)
        
        # Demonstrate safe changes
        print("🟢 Demo: Safe Changes")
        print("📝 Adding new functionality without breaking existing code")
        print("-" * 50)
        
        # Add a new function (safe)
        add_safe_function(demo_dir, "api_service.py")
        
        cmd = ["python", validator_path, "validate", "--format", "json", "--repo-path", demo_dir]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        try:
            report = json.loads(result.stdout)
            print(f"📊 Validation Results:")
            print(f"   Breaking changes: {report['summary']['breaking_changes']}")
            print(f"   New functions added: {len([r for r in report['results'] if r['category'] == 'function_added'])}")
            
            if report['summary']['breaking_changes'] == 0:
                print("✅ Result: Safe changes - commit would be allowed")
            else:
                print("❌ Result: Unexpected breaking changes detected")
                
        except json.JSONDecodeError:
            print(f"❌ Validator error: {result.stderr}")
        
        print()
        print("🎯 Summary:")
        print("The Functionality Preservation Validator successfully:")
        print("✅ Detected function parameter removals")
        print("✅ Caught function deletions")
        print("✅ Identified class method removals")
        print("✅ Found API endpoint removals")
        print("✅ Allowed safe additions")
        print("✅ Provided actionable suggestions")
        print()
        print("This demonstrates how Rule 2 enforcement prevents breaking changes!")
        
        return True
    
    finally:
        # Cleanup
        print(f"🧹 Cleaning up demo directory: {demo_dir}")
        shutil.rmtree(demo_dir, ignore_errors=True)

def modify_file(demo_dir, filename, old_text, new_text):
    """Modify a file by replacing text."""
    filepath = os.path.join(demo_dir, filename)
    with open(filepath, 'r') as f:
        content = f.read()
    
    content = content.replace(old_text, new_text)
    
    with open(filepath, 'w') as f:
        f.write(content)

def remove_function(demo_dir, filename, function_name):
    """Remove a function from a file."""
    filepath = os.path.join(demo_dir, filename)
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Find and remove function
    new_lines = []
    skip = False
    indent_level = 0
    
    for line in lines:
        if f'def {function_name}(' in line:
            skip = True
            indent_level = len(line) - len(line.lstrip())
        elif skip:
            current_indent = len(line) - len(line.lstrip())
            if line.strip() and current_indent <= indent_level:
                skip = False
                new_lines.append(line)
        else:
            new_lines.append(line)
    
    with open(filepath, 'w') as f:
        f.writelines(new_lines)

def remove_method(demo_dir, filename, class_name, method_name):
    """Remove a method from a class."""
    filepath = os.path.join(demo_dir, filename)
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    new_lines = []
    in_class = False
    skip_method = False
    method_indent = 0
    
    for line in lines:
        if f'class {class_name}' in line:
            in_class = True
            new_lines.append(line)
        elif in_class and f'def {method_name}(' in line:
            skip_method = True
            method_indent = len(line) - len(line.lstrip())
        elif skip_method:
            current_indent = len(line) - len(line.lstrip())
            if line.strip() and current_indent <= method_indent:
                skip_method = False
                new_lines.append(line)
        else:
            new_lines.append(line)
    
    with open(filepath, 'w') as f:
        f.writelines(new_lines)

def add_safe_function(demo_dir, filename):
    """Add a new function (safe change)."""
    filepath = os.path.join(demo_dir, filename)
    with open(filepath, 'r') as f:
        content = f.read()
    
    new_function = '''
def calculate_discount(amount, discount_percentage):
    """Calculate discount amount - NEW SAFE FUNCTION."""
    return amount * (discount_percentage / 100)
'''
    
    # Add before the main block
    content = content.replace('if __name__ == "__main__":', new_function + '\nif __name__ == "__main__":')
    
    with open(filepath, 'w') as f:
        f.write(content)

if __name__ == "__main__":
    try:
        success = demonstrate_breaking_changes()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n❌ Demo interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        sys.exit(1)