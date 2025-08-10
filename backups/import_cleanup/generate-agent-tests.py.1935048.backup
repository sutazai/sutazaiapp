#!/usr/bin/env python3
"""
Dynamic Agent Load Test Generator for SutazAI
Generates load tests for all 69+ agents dynamically from the agent registry.
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any

def load_agent_registry(registry_path: str) -> Dict[str, Any]:
    """Load the agent registry JSON file."""
    try:
        with open(registry_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Agent registry not found at {registry_path}")
        return {"agents": {}}
    except json.JSONDecodeError as e:
        print(f"Error parsing agent registry: {e}")
        return {"agents": {}}

def get_agent_port_mapping(agents: Dict[str, Any], start_port: int = 8080) -> Dict[str, int]:
    """Generate port mappings for all agents."""
    port_mapping = {}
    current_port = start_port
    
    for agent_name in sorted(agents.keys()):
        port_mapping[agent_name] = current_port
        current_port += 1
    
    return port_mapping

def generate_agent_categories(agents: Dict[str, Any]) -> Dict[str, List[str]]:
    """Categorize agents based on their capabilities and names."""
    categories = {
        "core_agents": [],
        "development_agents": [],
        "qa_testing_agents": [],
        "security_agents": [],
        "infrastructure_agents": [],
        "specialized_agents": []
    }
    
    for agent_name, agent_info in agents.items():
        capabilities = agent_info.get('capabilities', [])
        
        # Categorize based on name patterns and capabilities
        if any(keyword in agent_name.lower() for keyword in ['system', 'architect', 'orchestrator']):
            categories["core_agents"].append(agent_name)
        elif any(keyword in agent_name.lower() for keyword in ['developer', 'backend', 'frontend', 'fullstack', 'code']):
            categories["development_agents"].append(agent_name)
        elif any(keyword in agent_name.lower() for keyword in ['qa', 'test', 'quality']):
            categories["qa_testing_agents"].append(agent_name)
        elif any(keyword in agent_name.lower() for keyword in ['security', 'pentesting', 'audit', 'vulnerability']):
            categories["security_agents"].append(agent_name)
        elif any(keyword in agent_name.lower() for keyword in ['infrastructure', 'devops', 'deploy', 'container', 'k8s']):
            categories["infrastructure_agents"].append(agent_name)
        else:
            categories["specialized_agents"].append(agent_name)
    
    return categories

def generate_k6_test_script(agents: Dict[str, Any], port_mapping: Dict[str, int], categories: Dict[str, List[str]]) -> str:
    """Generate a comprehensive K6 test script for all agents."""
    
    agent_list = json.dumps(port_mapping, indent=2)
    category_list = json.dumps(categories, indent=2)
    
    script_template = f'''// Dynamically Generated Agent Load Tests for SutazAI
// Generated for {len(agents)} agents across {len(categories)} categories

import {{ check, sleep }} from 'k6';
import http from 'k6/http';
import {{ config, httpParams, validateResponse, randomChoice, randomInt }} from '../k6-config.js';

export {{ options }} from '../k6-config.js';

// All agent endpoints with port mappings
const ALL_AGENTS = {agent_list};

// Agent categories for targeted testing
const AGENT_CATEGORIES = {category_list};

// Agent-specific test scenarios based on capabilities
const AGENT_TEST_SCENARIOS = {{
  "ai-system-architect": {{
    prompts: [
      "Design a scalable microservices architecture for an e-commerce platform",
      "Create system architecture for a real-time chat application",
      "Design a distributed data processing pipeline",
      "Architect a cloud-native application with high availability"
    ],
    expected_response_time: 8000,
    complexity: "high"
  }},
  "ai-qa-team-lead": {{
    prompts: [
      "Create comprehensive test strategy for a web application",
      "Design automated testing pipeline for microservices",
      "Develop performance testing plan for high-traffic system",
      "Create security testing checklist for APIs"
    ],
    expected_response_time: 5000,
    complexity: "medium"
  }},
  "ai-senior-backend-developer": {{
    prompts: [
      "Implement REST API with authentication and authorization",
      "Create database schema with proper indexing",
      "Design asynchronous message processing system",
      "Implement caching strategy for high-performance API"
    ],
    expected_response_time: 6000,
    complexity: "high"
  }},
  "ai-senior-frontend-developer": {{
    prompts: [
      "Create responsive React components with TypeScript",
      "Implement state management with Redux Toolkit",
      "Design accessible user interface components",
      "Create progressive web app with offline capabilities"
    ],
    expected_response_time: 4000,
    complexity: "medium"
  }},
  "security-pentesting-specialist": {{
    prompts: [
      "Perform security assessment of web application",
      "Identify OWASP Top 10 vulnerabilities",
      "Create penetration testing methodology",
      "Analyze API security implementation"
    ],
    expected_response_time: 7000,
    complexity: "high"
  }},
  "deployment-automation-master": {{
    prompts: [
      "Create CI/CD pipeline for containerized application",
      "Design blue-green deployment strategy",
      "Implement infrastructure as code with Terraform",
      "Create monitoring and alerting for production deployment"
    ],
    expected_response_time: 5000,
    complexity: "medium"
  }}
}};

// Default test scenario for agents without specific configuration
const DEFAULT_SCENARIO = {{
  prompts: [
    "Analyze the current system requirements",
    "Provide recommendations for improvement",
    "Create implementation plan",
    "Generate best practices documentation"
  ],
  expected_response_time: 3000,
  complexity: "medium"
}};

export default function() {{
  const testType = randomChoice(['single_agent', 'category_test', 'cross_agent', 'stress_specific']);
  
  switch(testType) {{
    case 'single_agent':
      testSingleAgent();
      break;
    case 'category_test':
      testAgentCategory();
      break;
    case 'cross_agent':
      testCrossAgentInteraction();
      break;
    case 'stress_specific':
      stressTestSpecificAgent();
      break;
  }}
  
  sleep(randomInt(1, 3));
}}

function testSingleAgent() {{
  const agentNames = Object.keys(ALL_AGENTS);
  const agentName = randomChoice(agentNames);
  const port = ALL_AGENTS[agentName];
  
  const scenario = AGENT_TEST_SCENARIOS[agentName] || DEFAULT_SCENARIO;
  const prompt = randomChoice(scenario.prompts);
  
  const payload = {{
    prompt: prompt,
    max_tokens: scenario.complexity === 'high' ? 2000 : 1000,
    temperature: 0.7,
    stream: false,
    agent_context: {{
      test_type: 'single_agent_load_test',
      complexity: scenario.complexity
    }}
  }};
  
  const response = http.post(`${{config.baseUrl}}:${{port}}/api/chat`, JSON.stringify(payload), {{
    ...httpParams,
    timeout: `${{scenario.expected_response_time + 10000}}ms`,
    tags: {{ 
      ...httpParams.tags, 
      test_scenario: 'single_agent',
      agent: agentName,
      complexity: scenario.complexity
    }}
  }});
  
  validateResponse(response, 200);
  
  check(response, {{
    [`${{agentName}} responds within SLA`]: (r) => r.timings.duration < scenario.expected_response_time,
    [`${{agentName}} provides meaningful response`]: (r) => {{
      try {{
        const body = JSON.parse(r.body);
        return body.response && body.response.length > 50;
      }} catch (e) {{
        return false;
      }}
    }}
  }});
}}

function testAgentCategory() {{
  const categories = Object.keys(AGENT_CATEGORIES);
  const category = randomChoice(categories);
  const agentsInCategory = AGENT_CATEGORIES[category];
  
  if (agentsInCategory.length === 0) return;
  
  const agentName = randomChoice(agentsInCategory);
  const port = ALL_AGENTS[agentName];
  
  const categoryPrompts = {{
    "core_agents": [
      "Analyze system architecture and provide optimization recommendations",
      "Coordinate between different system components",
      "Design scalable system infrastructure"
    ],
    "development_agents": [
      "Implement feature with proper error handling and testing",
      "Refactor code for better maintainability",
      "Create API documentation and examples"
    ],
    "qa_testing_agents": [
      "Create comprehensive test coverage analysis",
      "Design automated testing strategy",
      "Perform quality assurance review"
    ],
    "security_agents": [
      "Conduct security vulnerability assessment",
      "Implement security best practices",
      "Create security compliance checklist"
    ],
    "infrastructure_agents": [
      "Design cloud infrastructure architecture",
      "Create deployment automation scripts",
      "Implement monitoring and logging strategy"
    ],
    "specialized_agents": [
      "Provide specialized domain expertise",
      "Create domain-specific recommendations",
      "Analyze requirements from specialized perspective"
    ]
  }};
  
  const prompt = randomChoice(categoryPrompts[category] || categoryPrompts["specialized_agents"]);
  
  const payload = {{
    prompt: prompt,
    max_tokens: 1500,
    temperature: 0.6,
    category_context: category
  }};
  
  const response = http.post(`${{config.baseUrl}}:${{port}}/api/chat`, JSON.stringify(payload), {{
    ...httpParams,
    tags: {{ 
      ...httpParams.tags, 
      test_scenario: 'category_test',
      agent: agentName,
      category: category
    }}
  }});
  
  validateResponse(response, 200);
  
  check(response, {{
    'category-appropriate response': (r) => r.status === 200,
    'response time reasonable for category': (r) => r.timings.duration < 10000
  }});
}}

function testCrossAgentInteraction() {{
  // Test interaction between agents in the same category
  const category = randomChoice(Object.keys(AGENT_CATEGORIES));
  const agentsInCategory = AGENT_CATEGORIES[category];
  
  if (agentsInCategory.length < 2) {{
    // Fall back to single agent test
    testSingleAgent();
    return;
  }}
  
  const agent1 = randomChoice(agentsInCategory);
  const agent2 = randomChoice(agentsInCategory.filter(a => a !== agent1));
  
  const collaborationPrompt = `Collaborate with ${{agent2}} to create a comprehensive solution for: ${{randomChoice([
    "building a scalable web application",
    "implementing security best practices",
    "creating automated testing pipeline",
    "designing cloud infrastructure"
  ])}}`;
  
  const payload = {{
    prompt: collaborationPrompt,
    max_tokens: 1000,
    collaboration_context: {{
      partner_agent: agent2,
      collaboration_type: "cross_agent_interaction"
    }}
  }};
  
  const response = http.post(`${{config.baseUrl}}:${{ALL_AGENTS[agent1]}}/api/chat`, JSON.stringify(payload), {{
    ...httpParams,
    timeout: '45s',
    tags: {{ 
      ...httpParams.tags, 
      test_scenario: 'cross_agent_interaction',
      primary_agent: agent1,
      secondary_agent: agent2,
      category: category
    }}
  }});
  
  validateResponse(response, 200);
  
  check(response, {{
    'cross-agent collaboration successful': (r) => r.status === 200,
    'collaboration response comprehensive': (r) => {{
      try {{
        const body = JSON.parse(r.body);
        return body.response && body.response.length > 200;
      }} catch (e) {{
        return false;
      }}
    }}
  }});
}}

function stressTestSpecificAgent() {{
  // Focus stress testing on high-load agents
  const highLoadAgents = [
    'ai-system-architect',
    'ai-senior-backend-developer',
    'ai-qa-team-lead',
    'jarvis-voice-interface'
  ];
  
  const availableAgents = highLoadAgents.filter(agent => ALL_AGENTS[agent]);
  if (availableAgents.length === 0) return;
  
  const agent = randomChoice(availableAgents);
  const port = ALL_AGENTS[agent];
  
  // Send multiple rapid requests to stress test
  for (let i = 0; i < 3; i++) {{
    const payload = {{
      prompt: `Stress test request ${{i + 1}}: Provide quick analysis`,
      max_tokens: 100,
      temperature: 0.3,
      stress_test: true
    }};
    
    http.post(`${{config.baseUrl}}:${{port}}/api/chat`, JSON.stringify(payload), {{
      ...httpParams,
      timeout: '15s',
      tags: {{ 
        ...httpParams.tags, 
        test_scenario: 'stress_specific',
        agent: agent,
        stress_iteration: i + 1
      }}
    }});
    
    sleep(0.5); // Small delay between stress requests
  }}
}}

// Agent availability test
export function testAllAgentsAvailability() {{
  const agentNames = Object.keys(ALL_AGENTS);
  const results = {{}};
  
  agentNames.forEach(agentName => {{
    const port = ALL_AGENTS[agentName];
    const response = http.get(`${{config.baseUrl}}:${{port}}/health`, {{
      ...httpParams,
      timeout: '10s',
      tags: {{ 
        ...httpParams.tags, 
        test_scenario: 'availability_check',
        agent: agentName
      }}
    }});
    
    results[agentName] = response.status === 200;
    
    check(response, {{
      [`${{agentName}} is available`]: (r) => r.status === 200
    }});
  }});
  
  const availableCount = Object.values(results).filter(Boolean).length;
  const totalCount = agentNames.length;
  
  check(null, {{
    'majority of agents available': () => availableCount > totalCount * 0.8, // 80% availability
    'critical agents available': () => {{
      const criticalAgents = ['ai-system-architect', 'ai-qa-team-lead', 'deployment-automation-master'];
      return criticalAgents.every(agent => results[agent] === true);
    }}
  }});
  
  console.log(`Agent Availability: ${{availableCount}}/${{totalCount}} (${{Math.round(availableCount/totalCount*100)}}%)`);
}}

// Performance benchmark for all agent categories
export function benchmarkAgentCategories() {{
  const categoryBenchmarks = {{}};
  
  Object.keys(AGENT_CATEGORIES).forEach(category => {{
    const agentsInCategory = AGENT_CATEGORIES[category];
    if (agentsInCategory.length === 0) return;
    
    const sampleAgent = agentsInCategory[0];
    const port = ALL_AGENTS[sampleAgent];
    
    const startTime = Date.now();
    
    const response = http.post(`${{config.baseUrl}}:${{port}}/api/chat`, JSON.stringify({{
      prompt: "Quick benchmark test",
      max_tokens: 50
    }}), {{
      ...httpParams,
      tags: {{ 
        ...httpParams.tags, 
        test_scenario: 'category_benchmark',
        category: category,
        agent: sampleAgent
      }}
    }});
    
    const endTime = Date.now();
    const responseTime = endTime - startTime;
    
    categoryBenchmarks[category] = {{
      agent: sampleAgent,
      responseTime: responseTime,
      success: response.status === 200
    }};
    
    check(response, {{
      [`${{category}} category responsive`]: (r) => r.status === 200 && responseTime < 5000
    }});
  }});
  
  console.log('Category Benchmarks:', JSON.stringify(categoryBenchmarks, null, 2));
}}
'''
    
    return script_template

def main():
    """Main function to generate agent load tests."""
    script_dir = Path(__file__).parent
    registry_path = script_dir.parent / "agents" / "agent_registry.json"
    output_path = script_dir / "tests" / "all-agents-load.js"
    
    print(f"Loading agent registry from: {registry_path}")
    registry_data = load_agent_registry(str(registry_path))
    
    agents = registry_data.get("agents", {})
    if not agents:
        print("No agents found in registry, using minimal test configuration")
        agents = {
            "ai-system-architect": {"capabilities": ["system_design"]},
            "ai-qa-team-lead": {"capabilities": ["testing", "quality_assurance"]},
            "ai-senior-backend-developer": {"capabilities": ["code_generation", "backend"]}
        }
    
    print(f"Found {len(agents)} agents in registry")
    
    # Generate port mappings and categories
    port_mapping = get_agent_port_mapping(agents)
    categories = generate_agent_categories(agents)
    
    print("Agent categories:")
    for category, agent_list in categories.items():
        print(f"  {category}: {len(agent_list)} agents")
    
    # Generate K6 test script
    test_script = generate_k6_test_script(agents, port_mapping, categories)
    
    # Ensure output directory exists
    output_path.parent.mkdir(exist_ok=True)
    
    # Write the test script
    with open(output_path, 'w') as f:
        f.write(test_script)
    
    print(f"Generated comprehensive agent load test: {output_path}")
    print(f"Test covers {len(agents)} agents across {len(categories)} categories")
    
    # Generate agent configuration file for reference
    config_path = script_dir / "agent-ports.json"
    with open(config_path, 'w') as f:
        json.dump(port_mapping, f, indent=2)
    
    print(f"Agent port configuration saved: {config_path}")

if __name__ == "__main__":
    main()