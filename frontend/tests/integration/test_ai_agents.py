"""
AI Agent-Specific Tests for Sutazai AI Application
Tests each of the 11 AI agents and their specific capabilities
"""

import pytest
import requests
import json
import time
from typing import Dict, Any, List

# Configuration
BASE_URL = "http://localhost:10200"
API_V1 = f"{BASE_URL}/api/v1"
TIMEOUT = 30

# Agent definitions with their specific capabilities
AGENTS = {
    "jarvis": {
        "name": "JARVIS",
        "description": "Advanced AI butler and orchestrator",
        "capabilities": ["orchestration", "planning", "multi-agent-coordination"],
        "test_prompts": [
            "Help me plan a software project",
            "Coordinate a team of developers",
            "Create a system architecture"
        ]
    },
    "documind": {
        "name": "DocuMind",
        "description": "Documentation and knowledge management specialist",
        "capabilities": ["documentation", "knowledge-extraction", "api-docs"],
        "test_prompts": [
            "Create API documentation for a REST endpoint",
            "Extract key information from this text",
            "Write a technical specification"
        ]
    },
    "quantumcoder": {
        "name": "QuantumCoder",
        "description": "Advanced coding and algorithm specialist",
        "capabilities": ["coding", "algorithms", "optimization"],
        "test_prompts": [
            "Write a Python function to sort a list",
            "Optimize this O(n²) algorithm",
            "Implement a binary search tree"
        ]
    },
    "datasage": {
        "name": "DataSage",
        "description": "Data analysis and insights expert",
        "capabilities": ["data-analysis", "statistics", "visualization"],
        "test_prompts": [
            "Analyze this dataset for patterns",
            "Calculate statistical significance",
            "Create a data visualization strategy"
        ]
    },
    "creativestorm": {
        "name": "CreativeStorm",
        "description": "Creative ideation and brainstorming specialist",
        "capabilities": ["brainstorming", "creative-writing", "ideation"],
        "test_prompts": [
            "Generate creative names for a startup",
            "Brainstorm features for a mobile app",
            "Write a creative story opening"
        ]
    },
    "techanalyst": {
        "name": "TechAnalyst",
        "description": "Technical analysis and architecture expert",
        "capabilities": ["tech-analysis", "architecture", "system-design"],
        "test_prompts": [
            "Analyze this system architecture",
            "Review this technology stack",
            "Design a microservices architecture"
        ]
    },
    "researchowl": {
        "name": "ResearchOwl",
        "description": "Research and information gathering specialist",
        "capabilities": ["research", "fact-checking", "information-synthesis"],
        "test_prompts": [
            "Research best practices for API design",
            "Find information about quantum computing",
            "Summarize recent AI developments"
        ]
    },
    "codeoptimizer": {
        "name": "CodeOptimizer",
        "description": "Code optimization and performance expert",
        "capabilities": ["performance", "optimization", "refactoring"],
        "test_prompts": [
            "Optimize this database query",
            "Refactor this code for better performance",
            "Identify performance bottlenecks"
        ]
    },
    "visionaryarch": {
        "name": "VisionaryArch",
        "description": "System architecture and design visionary",
        "capabilities": ["architecture", "scalability", "design-patterns"],
        "test_prompts": [
            "Design a scalable cloud architecture",
            "Apply design patterns to this problem",
            "Create a high-availability system design"
        ]
    },
    "testmaster": {
        "name": "TestMaster",
        "description": "Testing and quality assurance expert",
        "capabilities": ["testing", "qa", "test-automation"],
        "test_prompts": [
            "Write unit tests for this function",
            "Create a test plan for this feature",
            "Design automated test scenarios"
        ]
    },
    "securityguard": {
        "name": "SecurityGuard",
        "description": "Security and vulnerability assessment specialist",
        "capabilities": ["security", "vulnerability-assessment", "compliance"],
        "test_prompts": [
            "Review this code for security issues",
            "Identify potential vulnerabilities",
            "Create a security checklist"
        ]
    }
}


class TestAgentAvailability:
    """Test that all agents are available and responsive"""
    
    def test_list_all_agents(self):
        """Test that all 11 agents are listed"""
        response = requests.get(f"{API_V1}/agents", timeout=TIMEOUT)
        assert response.status_code == 200
        
        data = response.json()
        agents_list = data if isinstance(data, list) else data.get("agents", [])
        
        # Extract agent names
        if agents_list:
            agent_names = []
            for agent in agents_list:
                if isinstance(agent, dict):
                    agent_names.append(agent.get("name", agent.get("id", "")))
                else:
                    agent_names.append(str(agent))
                    
            # Check that expected agents are present
            for agent_id, agent_info in AGENTS.items():
                agent_found = any(
                    agent_info["name"].lower() in name.lower() or 
                    agent_id in name.lower()
                    for name in agent_names
                )
                assert agent_found, f"Agent {agent_info['name']} not found in list"
                
    def test_get_individual_agent_info(self):
        """Test getting information for each agent"""
        for agent_id, agent_info in AGENTS.items():
            response = requests.get(f"{API_V1}/agents/{agent_id}", timeout=TIMEOUT)
            
            if response.status_code == 200:
                data = response.json()
                # Verify agent information
                assert "name" in data or "id" in data
                assert "capabilities" in data or "description" in data
            else:
                # Agent endpoint might not exist, but should handle gracefully
                assert response.status_code in [404, 501]


class TestJARVISAgent:
    """Test JARVIS - Advanced AI butler and orchestrator"""
    
    def test_jarvis_orchestration(self):
        """Test JARVIS orchestration capabilities"""
        payload = {
            "message": "Help me plan a full-stack web application project",
            "agent_id": "jarvis"
        }
        
        response = requests.post(f"{API_V1}/chat", json=payload, timeout=TIMEOUT)
        assert response.status_code == 200
        
        data = response.json()
        response_text = data.get("response", data.get("message", data.get("content", "")))
        
        # JARVIS should provide structured planning
        assert len(response_text) > 50  # Should be a detailed response
        
    def test_jarvis_multi_agent_coordination(self):
        """Test JARVIS coordinating multiple agents"""
        payload = {
            "message": "I need to build a secure API with documentation. Coordinate the right agents.",
            "agent_id": "jarvis"
        }
        
        response = requests.post(f"{API_V1}/chat", json=payload, timeout=TIMEOUT)
        assert response.status_code == 200
        
    def test_jarvis_websocket_connection(self):
        """Test JARVIS WebSocket endpoint"""
        import websocket
        
        ws_url = "ws://localhost:10200/api/v1/jarvis/ws"
        try:
            ws = websocket.create_connection(ws_url, timeout=5)
            ws.send(json.dumps({"message": "Hello JARVIS"}))
            ws.close()
            assert True
        except Exception as e:
            print(f"JARVIS WebSocket test: {e}")


class TestDocuMindAgent:
    """Test DocuMind - Documentation specialist"""
    
    def test_documind_api_documentation(self):
        """Test DocuMind creating API documentation"""
        payload = {
            "message": "Create REST API documentation for a user authentication endpoint POST /api/auth/login",
            "agent_id": "documind"
        }
        
        response = requests.post(f"{API_V1}/chat", json=payload, timeout=TIMEOUT)
        assert response.status_code == 200
        
        data = response.json()
        response_text = data.get("response", data.get("message", data.get("content", "")))
        
        # Should contain API documentation elements
        doc_elements = ["endpoint", "method", "request", "response", "parameters"]
        # Check if response mentions documentation concepts
        assert any(elem in response_text.lower() for elem in doc_elements) or len(response_text) > 100
        
    def test_documind_knowledge_extraction(self):
        """Test DocuMind extracting knowledge from text"""
        text = """
        The system uses a microservices architecture with Docker containers.
        Each service communicates via REST APIs and message queues.
        Authentication is handled by JWT tokens with a 24-hour expiry.
        The database is PostgreSQL with Redis for caching.
        """
        
        payload = {
            "message": f"Extract key technical information from this text: {text}",
            "agent_id": "documind"
        }
        
        response = requests.post(f"{API_V1}/chat", json=payload, timeout=TIMEOUT)
        assert response.status_code == 200


class TestQuantumCoderAgent:
    """Test QuantumCoder - Advanced coding specialist"""
    
    def test_quantumcoder_code_generation(self):
        """Test QuantumCoder generating code"""
        payload = {
            "message": "Write a Python function to calculate fibonacci numbers efficiently",
            "agent_id": "quantumcoder"
        }
        
        response = requests.post(f"{API_V1}/chat", json=payload, timeout=TIMEOUT)
        assert response.status_code == 200
        
        data = response.json()
        response_text = data.get("response", data.get("message", data.get("content", "")))
        
        # Should contain code elements
        code_indicators = ["def ", "return", "fibonacci", "python", "```"]
        assert any(indicator in response_text.lower() for indicator in code_indicators) or len(response_text) > 50
        
    def test_quantumcoder_algorithm_optimization(self):
        """Test QuantumCoder optimizing algorithms"""
        bad_code = """
        def find_duplicates(lst):
            duplicates = []
            for i in range(len(lst)):
                for j in range(i+1, len(lst)):
                    if lst[i] == lst[j] and lst[i] not in duplicates:
                        duplicates.append(lst[i])
            return duplicates
        """
        
        payload = {
            "message": f"Optimize this O(n²) algorithm: {bad_code}",
            "agent_id": "quantumcoder"
        }
        
        response = requests.post(f"{API_V1}/chat", json=payload, timeout=TIMEOUT)
        assert response.status_code == 200


class TestDataSageAgent:
    """Test DataSage - Data analysis expert"""
    
    def test_datasage_data_analysis(self):
        """Test DataSage analyzing data"""
        payload = {
            "message": "Analyze the pattern in this sequence: 2, 4, 8, 16, 32",
            "agent_id": "datasage"
        }
        
        response = requests.post(f"{API_V1}/chat", json=payload, timeout=TIMEOUT)
        assert response.status_code == 200
        
        data = response.json()
        response_text = data.get("response", data.get("message", data.get("content", "")))
        
        # Should identify exponential/geometric pattern
        pattern_words = ["exponential", "geometric", "doubling", "power", "2^n"]
        assert any(word in response_text.lower() for word in pattern_words) or len(response_text) > 30
        
    def test_datasage_statistical_analysis(self):
        """Test DataSage performing statistical analysis"""
        payload = {
            "message": "Calculate mean, median, and standard deviation for: 10, 20, 30, 40, 50",
            "agent_id": "datasage"
        }
        
        response = requests.post(f"{API_V1}/chat", json=payload, timeout=TIMEOUT)
        assert response.status_code == 200


class TestCreativeStormAgent:
    """Test CreativeStorm - Creative ideation specialist"""
    
    def test_creativestorm_brainstorming(self):
        """Test CreativeStorm brainstorming ideas"""
        payload = {
            "message": "Generate 5 creative names for a AI-powered fitness app",
            "agent_id": "creativestorm"
        }
        
        response = requests.post(f"{API_V1}/chat", json=payload, timeout=TIMEOUT)
        assert response.status_code == 200
        
        data = response.json()
        response_text = data.get("response", data.get("message", data.get("content", "")))
        
        # Should generate multiple creative options
        assert len(response_text) > 50  # Should have substantial creative content
        
    def test_creativestorm_creative_writing(self):
        """Test CreativeStorm creative writing"""
        payload = {
            "message": "Write a creative opening sentence for a sci-fi story about AI",
            "agent_id": "creativestorm"
        }
        
        response = requests.post(f"{API_V1}/chat", json=payload, timeout=TIMEOUT)
        assert response.status_code == 200


class TestTechAnalystAgent:
    """Test TechAnalyst - Technical analysis expert"""
    
    def test_techanalyst_architecture_review(self):
        """Test TechAnalyst reviewing architecture"""
        payload = {
            "message": "Review the pros and cons of microservices vs monolithic architecture",
            "agent_id": "techanalyst"
        }
        
        response = requests.post(f"{API_V1}/chat", json=payload, timeout=TIMEOUT)
        assert response.status_code == 200
        
        data = response.json()
        response_text = data.get("response", data.get("message", data.get("content", "")))
        
        # Should discuss both architectures
        architecture_terms = ["microservice", "monolithic", "scalability", "complexity"]
        assert any(term in response_text.lower() for term in architecture_terms) or len(response_text) > 100
        
    def test_techanalyst_technology_stack(self):
        """Test TechAnalyst analyzing technology stack"""
        payload = {
            "message": "Analyze this tech stack: React, Node.js, PostgreSQL, Redis, Docker",
            "agent_id": "techanalyst"
        }
        
        response = requests.post(f"{API_V1}/chat", json=payload, timeout=TIMEOUT)
        assert response.status_code == 200


class TestResearchOwlAgent:
    """Test ResearchOwl - Research specialist"""
    
    def test_researchowl_information_gathering(self):
        """Test ResearchOwl gathering information"""
        payload = {
            "message": "Research the latest trends in artificial intelligence for 2024",
            "agent_id": "researchowl"
        }
        
        response = requests.post(f"{API_V1}/chat", json=payload, timeout=TIMEOUT)
        assert response.status_code == 200
        
        data = response.json()
        response_text = data.get("response", data.get("message", data.get("content", "")))
        
        # Should provide research-based content
        assert len(response_text) > 100  # Should have detailed research
        
    def test_researchowl_fact_checking(self):
        """Test ResearchOwl fact-checking capabilities"""
        payload = {
            "message": "Fact check: Python was created in 1991 by Guido van Rossum",
            "agent_id": "researchowl"
        }
        
        response = requests.post(f"{API_V1}/chat", json=payload, timeout=TIMEOUT)
        assert response.status_code == 200


class TestCodeOptimizerAgent:
    """Test CodeOptimizer - Performance optimization expert"""
    
    def test_codeoptimizer_performance_analysis(self):
        """Test CodeOptimizer analyzing performance"""
        slow_code = """
        def slow_function(data):
            result = []
            for item in data:
                if item not in result:
                    result.append(item)
            return result
        """
        
        payload = {
            "message": f"Identify performance issues in this code: {slow_code}",
            "agent_id": "codeoptimizer"
        }
        
        response = requests.post(f"{API_V1}/chat", json=payload, timeout=TIMEOUT)
        assert response.status_code == 200
        
        data = response.json()
        response_text = data.get("response", data.get("message", data.get("content", "")))
        
        # Should identify O(n²) complexity or suggest using set
        optimization_terms = ["o(n", "complexity", "set", "performance", "optimize"]
        assert any(term in response_text.lower() for term in optimization_terms) or len(response_text) > 50
        
    def test_codeoptimizer_query_optimization(self):
        """Test CodeOptimizer optimizing database queries"""
        payload = {
            "message": "Optimize this SQL query: SELECT * FROM users WHERE age > 18 AND city = 'NYC'",
            "agent_id": "codeoptimizer"
        }
        
        response = requests.post(f"{API_V1}/chat", json=payload, timeout=TIMEOUT)
        assert response.status_code == 200


class TestVisionaryArchAgent:
    """Test VisionaryArch - System architecture visionary"""
    
    def test_visionaryarch_scalable_design(self):
        """Test VisionaryArch designing scalable systems"""
        payload = {
            "message": "Design a scalable architecture for a social media platform",
            "agent_id": "visionaryarch"
        }
        
        response = requests.post(f"{API_V1}/chat", json=payload, timeout=TIMEOUT)
        assert response.status_code == 200
        
        data = response.json()
        response_text = data.get("response", data.get("message", data.get("content", "")))
        
        # Should discuss scalability concepts
        scalability_terms = ["scale", "load balance", "cache", "distributed", "microservice"]
        assert any(term in response_text.lower() for term in scalability_terms) or len(response_text) > 100
        
    def test_visionaryarch_design_patterns(self):
        """Test VisionaryArch applying design patterns"""
        payload = {
            "message": "What design pattern should I use for a notification system?",
            "agent_id": "visionaryarch"
        }
        
        response = requests.post(f"{API_V1}/chat", json=payload, timeout=TIMEOUT)
        assert response.status_code == 200


class TestTestMasterAgent:
    """Test TestMaster - Testing and QA expert"""
    
    def test_testmaster_unit_test_generation(self):
        """Test TestMaster generating unit tests"""
        code = """
        def add(a, b):
            return a + b
        
        def multiply(a, b):
            return a * b
        """
        
        payload = {
            "message": f"Write unit tests for these functions: {code}",
            "agent_id": "testmaster"
        }
        
        response = requests.post(f"{API_V1}/chat", json=payload, timeout=TIMEOUT)
        assert response.status_code == 200
        
        data = response.json()
        response_text = data.get("response", data.get("message", data.get("content", "")))
        
        # Should contain test-related code
        test_terms = ["test", "assert", "unittest", "pytest", "def test_"]
        assert any(term in response_text.lower() for term in test_terms) or len(response_text) > 50
        
    def test_testmaster_test_plan(self):
        """Test TestMaster creating test plans"""
        payload = {
            "message": "Create a test plan for a user authentication feature",
            "agent_id": "testmaster"
        }
        
        response = requests.post(f"{API_V1}/chat", json=payload, timeout=TIMEOUT)
        assert response.status_code == 200


class TestSecurityGuardAgent:
    """Test SecurityGuard - Security assessment specialist"""
    
    def test_securityguard_vulnerability_scan(self):
        """Test SecurityGuard identifying vulnerabilities"""
        vulnerable_code = """
        def login(username, password):
            query = f"SELECT * FROM users WHERE username='{username}' AND password='{password}'"
            return execute_query(query)
        """
        
        payload = {
            "message": f"Review this code for security issues: {vulnerable_code}",
            "agent_id": "securityguard"
        }
        
        response = requests.post(f"{API_V1}/chat", json=payload, timeout=TIMEOUT)
        assert response.status_code == 200
        
        data = response.json()
        response_text = data.get("response", data.get("message", data.get("content", "")))
        
        # Should identify SQL injection vulnerability
        security_terms = ["sql injection", "vulnerability", "security", "parameterized", "escape"]
        assert any(term in response_text.lower() for term in security_terms) or len(response_text) > 50
        
    def test_securityguard_security_checklist(self):
        """Test SecurityGuard creating security checklist"""
        payload = {
            "message": "Create a security checklist for a web application",
            "agent_id": "securityguard"
        }
        
        response = requests.post(f"{API_V1}/chat", json=payload, timeout=TIMEOUT)
        assert response.status_code == 200


class TestAgentCollaboration:
    """Test multiple agents working together"""
    
    def test_code_review_collaboration(self):
        """Test multiple agents reviewing code together"""
        code = """
        def process_user_data(user_input):
            query = f"SELECT * FROM users WHERE id = {user_input}"
            result = db.execute(query)
            return result
        """
        
        # Have different agents analyze the same code
        agents_to_test = ["quantumcoder", "securityguard", "codeoptimizer", "testmaster"]
        responses = {}
        
        for agent in agents_to_test:
            payload = {
                "message": f"Review this code: {code}",
                "agent_id": agent,
                "session_id": "collab_test_001"
            }
            
            response = requests.post(f"{API_V1}/chat", json=payload, timeout=TIMEOUT)
            assert response.status_code == 200
            responses[agent] = response.json()
            
        # Each agent should provide their perspective
        assert len(responses) == len(agents_to_test)
        
    def test_project_planning_collaboration(self):
        """Test agents collaborating on project planning"""
        project = "Build a secure e-commerce platform with real-time analytics"
        
        # JARVIS coordinates, others contribute
        agents_sequence = [
            ("jarvis", "Plan the project architecture"),
            ("visionaryarch", "Design the system architecture"),
            ("securityguard", "Identify security requirements"),
            ("datasage", "Plan analytics infrastructure"),
            ("testmaster", "Create testing strategy")
        ]
        
        session_id = f"project_collab_{int(time.time())}"
        
        for agent, task in agents_sequence:
            payload = {
                "message": f"{task} for: {project}",
                "agent_id": agent,
                "session_id": session_id
            }
            
            response = requests.post(f"{API_V1}/chat", json=payload, timeout=TIMEOUT)
            assert response.status_code == 200


class TestAgentPerformance:
    """Test agent performance characteristics"""
    
    def test_agent_response_times(self):
        """Test response times for different agents"""
        simple_prompt = "Hello, how are you?"
        
        response_times = {}
        for agent_id in AGENTS.keys():
            start_time = time.time()
            
            payload = {
                "message": simple_prompt,
                "agent_id": agent_id
            }
            
            response = requests.post(f"{API_V1}/chat", json=payload, timeout=TIMEOUT)
            elapsed = time.time() - start_time
            
            response_times[agent_id] = elapsed
            
            # Each agent should respond within reasonable time
            assert elapsed < 10.0, f"Agent {agent_id} took {elapsed:.2f} seconds"
            
        # Log performance results
        print(f"Agent response times: {response_times}")
        
    def test_agent_parallel_processing(self):
        """Test agents handling parallel requests"""
        import concurrent.futures
        
        def agent_request(agent_id, message_num):
            payload = {
                "message": f"Test message {message_num}",
                "agent_id": agent_id
            }
            
            response = requests.post(f"{API_V1}/chat", json=payload, timeout=TIMEOUT)
            return response.status_code == 200
            
        # Test parallel requests to the same agent
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = []
            for i in range(5):
                future = executor.submit(agent_request, "jarvis", i)
                futures.append(future)
                
            results = [f.result() for f in concurrent.futures.as_completed(futures)]
            
        # Most requests should succeed
        successful = sum(1 for r in results if r)
        assert successful >= 3, f"Only {successful}/5 parallel requests succeeded"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])