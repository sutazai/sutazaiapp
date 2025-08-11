"""
Example usage of Ollama Integration Agent by other agents.
Shows how to integrate LLM capabilities into agent workflows.
"""

import asyncio
import logging
from typing import Dict, Any, List
import sys
import os

sys.path.append('/opt/sutazaiapp')

from agents.ollama_integration.app import OllamaIntegrationAgent
from schemas.ollama_schemas import OllamaGenerateRequest

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TaskAnalyzerAgent:
    """
    Example agent that uses Ollama to analyze and categorize tasks.
    """
    
    def __init__(self):
        self.ollama = OllamaIntegrationAgent()
        
    async def __aenter__(self):
        await self.ollama.start()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.ollama.close()
        
    async def analyze_task(self, task_description: str) -> Dict[str, Any]:
        """
        Analyze a task using LLM and extract key information.
        """
        prompt = f"""
        Analyze this task and provide:
        1. Category (development, testing, documentation, deployment)
        2. Priority (high, medium, low)
        3. Estimated effort (hours)
        4. Required skills
        
        Task: {task_description}
        
        Respond in this format:
        Category: <category>
        Priority: <priority>
        Effort: <hours>
        Skills: <comma-separated skills>
        """
        
        try:
            result = await self.ollama.generate(
                prompt=prompt,
                temperature=0.3,  # Lower temperature for structured output
                max_tokens=100
            )
            
            # Parse the response
            analysis = self._parse_analysis(result["response"])
            analysis["original_task"] = task_description
            analysis["llm_latency_ms"] = result["latency"]
            
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze task: {e}")
            return {
                "error": str(e),
                "original_task": task_description
            }
            
    def _parse_analysis(self, response: str) -> Dict[str, Any]:
        """Parse structured response from LLM."""
        lines = response.strip().split('\n')
        result = {}
        
        for line in lines:
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip().lower()
                value = value.strip()
                
                if key == 'category':
                    result['category'] = value
                elif key == 'priority':
                    result['priority'] = value
                elif key == 'effort':
                    result['effort_hours'] = float(value.replace('hours', '').strip())
                elif key == 'skills':
                    result['required_skills'] = [s.strip() for s in value.split(',')]
                    
        return result


class CodeReviewAgent:
    """
    Example agent that uses Ollama to review code snippets.
    """
    
    def __init__(self):
        self.ollama = OllamaIntegrationAgent()
        
    async def __aenter__(self):
        await self.ollama.start()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.ollama.close()
        
    async def review_code(self, code: str, language: str = "python") -> Dict[str, Any]:
        """
        Review code and provide feedback.
        """
        prompt = f"""
        Review this {language} code and provide:
        1. Issues found (if any)
        2. Security concerns
        3. Performance suggestions
        4. Best practices violations
        
        Code:
        ```{language}
        {code}
        ```
        
        Be concise and specific.
        """
        
        try:
            result = await self.ollama.generate(
                prompt=prompt,
                temperature=0.5,
                max_tokens=300
            )
            
            return {
                "review": result["response"],
                "tokens_used": result["tokens"],
                "review_time_ms": result["latency"]
            }
            
        except Exception as e:
            logger.error(f"Failed to review code: {e}")
            return {"error": str(e)}


class DocumentationAgent:
    """
    Example agent that generates documentation using LLM.
    """
    
    def __init__(self):
        self.ollama = OllamaIntegrationAgent()
        
    async def __aenter__(self):
        await self.ollama.start()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.ollama.close()
        
    async def generate_docstring(self, function_code: str) -> str:
        """
        Generate a docstring for a function.
        """
        prompt = f"""
        Generate a comprehensive docstring for this function:
        
        {function_code}
        
        Include:
        - Description
        - Args with types
        - Returns with type
        - Example usage
        
        Use Google style docstring format.
        """
        
        try:
            result = await self.ollama.generate(
                prompt=prompt,
                temperature=0.4,
                max_tokens=200,
                stop=['"""', "'''", "def ", "class "]
            )
            
            return result["response"]
            
        except Exception as e:
            logger.error(f"Failed to generate docstring: {e}")
            return f"# Error generating docstring: {e}"
            
    async def summarize_readme(self, readme_content: str) -> str:
        """
        Generate a concise summary of a README file.
        """
        prompt = f"""
        Summarize this README in 3-5 bullet points:
        
        {readme_content[:1000]}  # Limit input size
        
        Focus on:
        - What the project does
        - Key features
        - How to use it
        """
        
        try:
            result = await self.ollama.generate(
                prompt=prompt,
                temperature=0.3,
                max_tokens=150
            )
            
            return result["response"]
            
        except Exception as e:
            logger.error(f"Failed to summarize README: {e}")
            return ""


class TestScenarioAgent:
    """
    Example agent that generates test scenarios using LLM.
    """
    
    def __init__(self):
        self.ollama = OllamaIntegrationAgent()
        
    async def __aenter__(self):
        await self.ollama.start()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.ollama.close()
        
    async def generate_test_cases(self, feature_description: str) -> List[str]:
        """
        Generate test cases for a feature.
        """
        prompt = f"""
        Generate 5 test cases for this feature:
        
        {feature_description}
        
        Format each test case as:
        TEST <number>: <description>
        GIVEN: <precondition>
        WHEN: <action>
        THEN: <expected result>
        """
        
        try:
            result = await self.ollama.generate(
                prompt=prompt,
                temperature=0.6,
                max_tokens=400
            )
            
            # Parse test cases
            test_cases = []
            current_test = []
            
            for line in result["response"].split('\n'):
                if line.startswith('TEST ') and current_test:
                    test_cases.append('\n'.join(current_test))
                    current_test = [line]
                elif line.strip():
                    current_test.append(line)
                    
            if current_test:
                test_cases.append('\n'.join(current_test))
                
            return test_cases
            
        except Exception as e:
            logger.error(f"Failed to generate test cases: {e}")
            return []


async def demo_task_analyzer():
    """Demo the task analyzer agent."""
    print("\n=== Task Analyzer Demo ===")
    
    async with TaskAnalyzerAgent() as agent:
        tasks = [
            "Implement user authentication with JWT tokens",
            "Write unit tests for the payment processing module",
            "Deploy the application to Kubernetes cluster",
            "Update API documentation with new endpoints"
        ]
        
        for task in tasks:
            print(f"\nAnalyzing: {task}")
            result = await agent.analyze_task(task)
            
            if "error" not in result:
                print(f"  Category: {result.get('category', 'unknown')}")
                print(f"  Priority: {result.get('priority', 'unknown')}")
                print(f"  Effort: {result.get('effort_hours', 0)} hours")
                print(f"  Skills: {', '.join(result.get('required_skills', []))}")
                print(f"  LLM Latency: {result.get('llm_latency_ms', 0):.2f}ms")
            else:
                print(f"  Error: {result['error']}")


async def demo_code_review():
    """Demo the code review agent."""
    print("\n=== Code Review Demo ===")
    
    code_sample = """
def process_user_data(user_id):
    query = f"SELECT * FROM users WHERE id = {user_id}"
    result = db.execute(query)
    
    password = result['password']
    print(f"User password: {password}")
    
    return result
"""
    
    async with CodeReviewAgent() as agent:
        print(f"\nReviewing code...")
        result = await agent.review_code(code_sample, "python")
        
        if "error" not in result:
            print(f"Review:\n{result['review']}")
            print(f"\nTokens used: {result['tokens_used']}")
            print(f"Review time: {result['review_time_ms']:.2f}ms")
        else:
            print(f"Error: {result['error']}")


async def demo_documentation():
    """Demo the documentation agent."""
    print("\n=== Documentation Generator Demo ===")
    
    function_code = """
def calculate_fibonacci(n: int) -> List[int]:
    if n <= 0:
        return []
    elif n == 1:
        return [0]
    elif n == 2:
        return [0, 1]
    
    fib = [0, 1]
    for i in range(2, n):
        fib.append(fib[i-1] + fib[i-2])
    
    return fib
"""
    
    async with DocumentationAgent() as agent:
        print("\nGenerating docstring...")
        docstring = await agent.generate_docstring(function_code)
        print(f"Generated docstring:\n{docstring}")


async def demo_test_scenarios():
    """Demo the test scenario generator."""
    print("\n=== Test Scenario Generator Demo ===")
    
    feature = """
    User Registration Feature:
    - Users can register with email and password
    - Email must be unique
    - Password must be at least 8 characters
    - User receives confirmation email
    - Account is inactive until email is confirmed
    """
    
    async with TestScenarioAgent() as agent:
        print("\nGenerating test cases...")
        test_cases = await agent.generate_test_cases(feature)
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n{test_case}")


async def main():
    """Run all demos."""
    print("Ollama Integration Agent - Usage Examples")
    print("=" * 50)
    
    # Verify Ollama is available
    async with OllamaIntegrationAgent() as agent:
        if await agent.verify_model("tinyllama"):
            print("✅ TinyLlama model is available")
        else:
            print("❌ TinyLlama model not found. Please run: ollama pull tinyllama")
            return
    
    # Run demos
    await demo_task_analyzer()
    await demo_code_review()
    await demo_documentation()
    await demo_test_scenarios()
    
    print("\n" + "=" * 50)
    print("All demos completed!")


if __name__ == "__main__":
    asyncio.run(main())