import argparse

# --- Model Definitions ---
# This dictionary maps task types to recommended models.
# This can be expanded and customized based on model availability and performance.
MODEL_ROUTING = {
    "code_generation": "deepseek-coder:33b",
    "summarization": "mistral:7b",
    "general_q&a": "llama3.2:3b",
    "reasoning": "deepseek-r1:8b",
    "default": "llama3.2:1b",
}

# --- Keyword-based Task Identification ---
# This dictionary maps keywords to task types.
# This is a simple but effective way to infer the user's intent.
TASK_KEYWORDS = {
    "code_generation": ["code", "program", "script", "function", "class", "algorithm"],
    "summarization": ["summarize", "summary", "tl;dr", "in short", "key points"],
    "reasoning": ["why", "how", "explain", "reason", "logic"],
}

def select_model(prompt):
    """Selects the best model based on the prompt's content.

    Args:
        prompt (str): The user's input prompt.

    Returns:
        str: The name of the recommended model.
    """
    prompt_lower = prompt.lower()

    # 1. Identify task type based on keywords
    for task, keywords in TASK_KEYWORDS.items():
        if any(keyword in prompt_lower for keyword in keywords):
            print(f"INFO: Detected task type: {task}")
            return MODEL_ROUTING.get(task, MODEL_ROUTING["default"])

    # 2. If no specific task is identified, use the general Q&A model
    print("INFO: No specific task detected, using general Q&A model.")
    return MODEL_ROUTING["general_q&a"]

def main():
    """Main function to demonstrate model selection."""
    parser = argparse.ArgumentParser(description="Select the best AI model for a given prompt.")
    parser.add_argument("prompt", type=str, help="The user prompt to analyze.")
    args = parser.parse_args()

    selected_model = select_model(args.prompt)

    print(f"\nSelected Model: {selected_model}")
    print("\nTo run this model, you would typically use a command like:")
    print(f"  ollama run {selected_model} \"{args.prompt}\"")

if __name__ == "__main__":
    main()
