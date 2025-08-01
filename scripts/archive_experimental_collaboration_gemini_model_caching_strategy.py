

import time
import hashlib

# --- Simple In-Memory Cache ---
# A dictionary to store cached model responses.
# The key is a hash of the prompt, and the value is the response.
CACHE = {}

def get_cache_key(prompt):
    """Creates a unique cache key for a given prompt using SHA256."""
    return hashlib.sha256(prompt.encode()).hexdigest()

def get_model_response(prompt, model_runner):
    """Gets a model response, using the cache if available."""
    cache_key = get_cache_key(prompt)

    # 1. Check if the response is in the cache
    if cache_key in CACHE:
        print("INFO: Cache hit! Returning cached response.")
        return CACHE[cache_key]

    # 2. If not in cache, run the model and store the result
    print("INFO: Cache miss. Running model to get new response.")
    response = model_runner(prompt)
    CACHE[cache_key] = response
    return response

# --- Example Usage ---
def dummy_model_runner(prompt):
    """A dummy function to simulate running a model.

    In a real application, this would be replaced with a call to the
    Ollama API or another model serving framework.
    """
    print("--- (Simulating model execution) ---")
    time.sleep(2)  # Simulate the time it takes to get a response
    return f"This is the model's response to: '{prompt}'"

def main():
    """Main function to demonstrate the caching mechanism."""
    print("--- Model Response Caching Demonstration ---")

    prompts = [
        "What is the capital of Japan?",
        "Write a short poem about the ocean.",
        "What is the capital of Japan?",  # This prompt is repeated to show caching
    ]

    for prompt in prompts:
        print(f"\nQuery: '{prompt}'")
        start_time = time.time()
        response = get_model_response(prompt, dummy_model_runner)
        end_time = time.time()

        print(f"Response: '{response}'")
        print(f"Time taken: {end_time - start_time:.2f} seconds")

    print("\n--- Cache Contents ---")
    print(json.dumps(CACHE, indent=2))

if __name__ == "__main__":
    import json
    main()

