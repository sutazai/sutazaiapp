import time

# --- Model Fallback Chain ---
# A list of models to try in order, from most to least preferred.
FALLBACK_CHAIN = [
    "deepseek-coder:33b",  # Primary, most capable model
    "mistral:7b",          # Secondary, good all-around model
    "llama3.2:1b",         # Tertiary, lightweight and fast
]

def model_runner_with_fallback(prompt):
    """Runs a prompt through the model fallback chain.

    It will try each model in the FALLBACK_CHAIN until one succeeds.
    """
    for model_name in FALLBACK_CHAIN:
        print(f"--- Attempting to run model: {model_name} ---")
        try:
            # In a real application, this would be a call to the Ollama API.
            # We simulate success and failure here for demonstration.
            response = simulate_model_run(model_name, prompt)
            print(f"✅ Success with model: {model_name}")
            return response
        except Exception as e:
            print(f"❌ Failed to run model {model_name}: {e}")

    # If all models in the chain fail
    print("❌ All models in the fallback chain failed.")
    return "Error: Unable to get a response from any model."

# --- Example Usage ---
def simulate_model_run(model_name, prompt):
    """A dummy function to simulate running a model.

    This function will randomly "fail" to demonstrate the fallback mechanism.
    """
    print(f"--- (Simulating model execution for {model_name}) ---")
    time.sleep(1)

    # Simulate a failure for the primary model
    if model_name == "deepseek-coder:33b" and "fail" in prompt.lower():
        raise ValueError("Simulated model failure: GPU out of memory")

    return f"Response from {model_name} for prompt: '{prompt}'"

def main():
    """Main function to demonstrate the fallback mechanism."""
    print("--- Model Fallback Mechanism Demonstration ---")

    # --- Prompt 1: Should succeed with the primary model ---
    print("\n--- Running prompt that should succeed ---")
    prompt1 = "What is the capital of Australia?"
    print(f"Query: '{prompt1}'")
    response1 = model_runner_with_fallback(prompt1)
    print(f"Final Response: '{response1}'")

    # --- Prompt 2: Should fail with the primary and succeed with the secondary ---
    print("\n--- Running prompt that should fail over ---")
    prompt2 = "Write a function to do something, but fail."
    print(f"Query: '{prompt2}'")
    response2 = model_runner_with_fallback(prompt2)
    print(f"Final Response: '{response2}'")

if __name__ == "__main__":
    main()
