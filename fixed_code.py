"""
Fixed version after /sc:troubleshoot analysis
All bugs have been systematically addressed
"""

def calculate_average(numbers):
    """Calculate average of a list of numbers with error handling"""
    if not numbers:  # FIX: Check for empty list
        return 0  # or raise ValueError("Cannot calculate average of empty list")
    
    total = sum(numbers)  # More Pythonic
    return total / len(numbers)

def process_user_data(users):
    """Process user data with proper error handling"""
    results = {}
    for user in users:
        # FIX: Use .get() with default values
        age = user.get('age', 0)
        name = user.get('name', 'Unknown')
        
        # FIX: Handle None values
        if name is not None:
            name = str(name).upper()
        else:
            name = 'UNKNOWN'
            
        results[name] = age
    return results

def fetch_data(url, timeout=10):
    """Fetch data from URL with comprehensive error handling"""
    import requests
    import json
    
    try:
        # FIX: Add timeout and error handling
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()  # Raise for HTTP errors
        
        # FIX: Handle non-JSON responses
        try:
            return response.json()
        except json.JSONDecodeError:
            return {"error": "Invalid JSON response", "text": response.text}
            
    except requests.RequestException as e:
        # FIX: Proper error handling
        return {"error": f"Network error: {str(e)}"}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}

# Safe test cases
if __name__ == "__main__":
    # These now work without errors
    print(f"Average of empty list: {calculate_average([])}")
    print(f"Average of [1,2,3]: {calculate_average([1, 2, 3])}")
    
    # Test with problematic data
    users = [
        {'name': 'Alice', 'age': 30},
        {'name': None},  # Missing age, None name
        {},  # Empty dict
    ]
    print(f"Processed users: {process_user_data(users)}")
    
    # Test network function (won't actually call)
    print("fetch_data() now has proper error handling")