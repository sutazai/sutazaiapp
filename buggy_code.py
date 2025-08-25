"""
Buggy code for demonstration of /sc:troubleshoot
This code has intentional issues
"""

def calculate_average(numbers):
    """Calculate average of a list of numbers"""
    total = 0
    for num in numbers:
        total += num
    return total / len(numbers)  # Bug: Division by zero if empty list

def process_user_data(users):
    """Process user data and return stats"""
    results = {}
    for user in users:
        # Bug: KeyError if 'age' doesn't exist
        age = user['age']
        # Bug: TypeError if name is None
        name = user['name'].upper()
        results[name] = age
    return results

def fetch_data(url):
    """Fetch data from URL"""
    import requests
    # Bug: No error handling for network issues
    response = requests.get(url)
    # Bug: Assumes JSON response
    return response.json()

# Test the buggy code
if __name__ == "__main__":
    # This will cause errors
    print(calculate_average([]))  # Division by zero
    print(process_user_data([{'name': None}]))  # Multiple errors