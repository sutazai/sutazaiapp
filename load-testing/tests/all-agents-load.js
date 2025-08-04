// Dynamically Generated Agent Load Tests for SutazAI
// Generated for 105 agents across 6 categories

import { check, sleep } from 'k6';
import http from 'k6/http';
import { config, httpParams, validateResponse, randomChoice, randomInt } from '../k6-config.js';

export { options } from '../k6-config.js';

// All agent endpoints with port mappings
const ALL_AGENTS = {
  "adversarial-attack-detector": 8080,
  "agent-creator": 8081,
  "agent-debugger": 8082,
  "agent-orchestrator": 8083,
  "agentgpt-autonomous-executor": 8084,
  "agentzero-coordinator": 8085,
  "agi-system-architect": 8086,
  "ai-agent-creator": 8087,
  "ai-agent-orchestrator": 8088,
  "ai-product-manager": 8089,
  "ai-qa-team-lead": 8090,
  "ai-scrum-master": 8091,
  "ai-senior-backend-developer": 8092,
  "ai-senior-engineer": 8093,
  "ai-senior-frontend-developer": 8094,
  "ai-senior-full-stack-developer": 8095,
  "ai-system-architect": 8096,
  "ai-system-validator": 8097,
  "ai-testing-qa-validator": 8098,
  "automated-incident-responder": 8099,
  "autonomous-system-controller": 8100,
  "autonomous-task-executor": 8101,
  "bias-and-fairness-auditor": 8102,
  "bigagi-system-manager": 8103,
  "browser-automation-orchestrator": 8104,
  "cicd-pipeline-orchestrator": 8105,
  "code-generation-improver": 8106,
  "code-quality-gateway-sonarqube": 8107,
  "codebase-team-lead": 8108,
  "cognitive-load-monitor": 8109,
  "complex-problem-solver": 8110,
  "compute-scheduler-and-optimizer": 8111,
  "container-orchestrator-k3s": 8112,
  "container-vulnerability-scanner-trivy": 8113,
  "context-optimization-engineer": 8114,
  "cpu-only-hardware-optimizer": 8115,
  "data-drift-detector": 8116,
  "data-lifecycle-manager": 8117,
  "data-version-controller-dvc": 8118,
  "deep-learning-brain-architect": 8119,
  "deep-learning-brain-manager": 8120,
  "deep-learning-coordinator-manager": 8121,
  "deep-local-brain-builder": 8122,
  "deploy-automation-master": 8123,
  "deployment-automation-master": 8124,
  "dify-automation-specialist": 8125,
  "distributed-tracing-analyzer-jaeger": 8126,
  "document-knowledge-manager": 8127,
  "edge-inference-proxy": 8128,
  "emergency-shutdown-coordinator": 8129,
  "energy-consumption-optimize": 8130,
  "ethical-governor": 8131,
  "evolution-strategy-trainer": 8132,
  "experiment-tracker": 8133,
  "explainability-and-transparency-agent": 8134,
  "financial-analysis-specialist": 8135,
  "flowiseai-flow-manager": 8136,
  "garbage-collector": 8137,
  "genetic-algorithm-tuner": 8138,
  "goal-setting-and-planning-agent": 8139,
  "gpu-hardware-optimizer": 8140,
  "hardware-resource-optimizer": 8141,
  "honeypot-deployment-agent": 8142,
  "human-oversight-interface-agent": 8143,
  "infrastructure-devops-manager": 8144,
  "jarvis-voice-interface": 8145,
  "kali-hacker": 8146,
  "kali-security-specialist": 8147,
  "langflow-workflow-designer": 8148,
  "localagi-orchestration-manager": 8149,
  "log-aggregator-loki": 8150,
  "mega-code-auditor": 8151,
  "metrics-collector-prometheus": 8152,
  "ml-experiment-tracker-mlflow": 8153,
  "neural-architecture-search": 8154,
  "observability-dashboard-manager-grafana": 8155,
  "ollama-integration-specialist": 8156,
  "opendevin-code-generator": 8157,
  "private-data-analyst": 8158,
  "private-registry-manager-harbor": 8159,
  "product-manager": 8160,
  "prompt-injection-guard": 8161,
  "qa-team-lead": 8162,
  "quantum-ai-researcher": 8163,
  "ram-hardware-optimizer": 8164,
  "resource-arbitration-agent": 8165,
  "resource-visualiser": 8166,
  "runtime-behavior-anomaly-detector": 8167,
  "scrum-master": 8168,
  "secrets-vault-manager-vault": 8169,
  "security-pentesting-specialist": 8170,
  "semgrep-security-analyzer": 8171,
  "senior-ai-engineer": 8172,
  "senior-backend-developer": 8173,
  "senior-engineer": 8174,
  "senior-frontend-developer": 8175,
  "senior-full-stack-developer": 8176,
  "shell-automation-specialist": 8177,
  "system-knowledge-curator": 8178,
  "system-optimizer-reorganizer": 8179,
  "system-performance-forecaster": 8180,
  "system-validator": 8181,
  "task-assignment-coordinator": 8182,
  "testing-qa-team-lead": 8183,
  "testing-qa-validator": 8184
};

// Agent categories for targeted testing
const AGENT_CATEGORIES = {
  "core_agents": [
    "browser-automation-orchestrator",
    "agi-system-architect",
    "autonomous-system-controller",
    "bigagi-system-manager",
    "system-optimizer-reorganizer",
    "ai-agent-orchestrator",
    "ai-system-architect",
    "cicd-pipeline-orchestrator",
    "container-orchestrator-k3s",
    "deep-learning-brain-architect",
    "neural-architecture-search",
    "agent-orchestrator",
    "ai-system-validator",
    "system-knowledge-curator",
    "system-performance-forecaster",
    "system-validator"
  ],
  "development_agents": [
    "code-generation-improver",
    "opendevin-code-generator",
    "senior-frontend-developer",
    "senior-backend-developer",
    "ai-senior-full-stack-developer",
    "code-quality-gateway-sonarqube",
    "senior-full-stack-developer",
    "ai-senior-backend-developer",
    "ai-senior-frontend-developer",
    "codebase-team-lead",
    "mega-code-auditor"
  ],
  "qa_testing_agents": [
    "security-pentesting-specialist",
    "testing-qa-validator",
    "ai-qa-team-lead",
    "ai-testing-qa-validator",
    "qa-team-lead",
    "testing-qa-team-lead"
  ],
  "security_agents": [
    "semgrep-security-analyzer",
    "kali-security-specialist",
    "bias-and-fairness-auditor",
    "container-vulnerability-scanner-trivy"
  ],
  "infrastructure_agents": [
    "deployment-automation-master",
    "infrastructure-devops-manager",
    "deploy-automation-master",
    "honeypot-deployment-agent"
  ],
  "specialized_agents": [
    "document-knowledge-manager",
    "ollama-integration-specialist",
    "senior-ai-engineer",
    "hardware-resource-optimizer",
    "jarvis-voice-interface",
    "shell-automation-specialist",
    "dify-automation-specialist",
    "agentgpt-autonomous-executor",
    "task-assignment-coordinator",
    "ai-scrum-master",
    "complex-problem-solver",
    "context-optimization-engineer",
    "langflow-workflow-designer",
    "localagi-orchestration-manager",
    "ai-agent-creator",
    "flowiseai-flow-manager",
    "agentzero-coordinator",
    "financial-analysis-specialist",
    "ai-product-manager",
    "private-data-analyst",
    "deep-learning-coordinator-manager",
    "adversarial-attack-detector",
    "agent-creator",
    "deep-learning-brain-manager",
    "deep-local-brain-builder",
    "distributed-tracing-analyzer-jaeger",
    "ethical-governor",
    "evolution-strategy-trainer",
    "genetic-algorithm-tuner",
    "goal-setting-and-planning-agent",
    "quantum-ai-researcher",
    "resource-arbitration-agent",
    "runtime-behavior-anomaly-detector",
    "agent-debugger",
    "ai-senior-engineer",
    "automated-incident-responder",
    "autonomous-task-executor",
    "cognitive-load-monitor",
    "compute-scheduler-and-optimizer",
    "cpu-only-hardware-optimizer",
    "data-drift-detector",
    "data-lifecycle-manager",
    "data-version-controller-dvc",
    "edge-inference-proxy",
    "emergency-shutdown-coordinator",
    "energy-consumption-optimize",
    "experiment-tracker",
    "explainability-and-transparency-agent",
    "garbage-collector",
    "gpu-hardware-optimizer",
    "human-oversight-interface-agent",
    "kali-hacker",
    "log-aggregator-loki",
    "metrics-collector-prometheus",
    "ml-experiment-tracker-mlflow",
    "observability-dashboard-manager-grafana",
    "private-registry-manager-harbor",
    "product-manager",
    "prompt-injection-guard",
    "ram-hardware-optimizer",
    "resource-visualiser",
    "scrum-master",
    "secrets-vault-manager-vault",
    "senior-engineer"
  ]
};

// Agent-specific test scenarios based on capabilities
const AGENT_TEST_SCENARIOS = {
  "ai-system-architect": {
    prompts: [
      "Design a scalable microservices architecture for an e-commerce platform",
      "Create system architecture for a real-time chat application",
      "Design a distributed data processing pipeline",
      "Architect a cloud-native application with high availability"
    ],
    expected_response_time: 8000,
    complexity: "high"
  },
  "ai-qa-team-lead": {
    prompts: [
      "Create comprehensive test strategy for a web application",
      "Design automated testing pipeline for microservices",
      "Develop performance testing plan for high-traffic system",
      "Create security testing checklist for APIs"
    ],
    expected_response_time: 5000,
    complexity: "medium"
  },
  "ai-senior-backend-developer": {
    prompts: [
      "Implement REST API with authentication and authorization",
      "Create database schema with proper indexing",
      "Design asynchronous message processing system",
      "Implement caching strategy for high-performance API"
    ],
    expected_response_time: 6000,
    complexity: "high"
  },
  "ai-senior-frontend-developer": {
    prompts: [
      "Create responsive React components with TypeScript",
      "Implement state management with Redux Toolkit",
      "Design accessible user interface components",
      "Create progressive web app with offline capabilities"
    ],
    expected_response_time: 4000,
    complexity: "medium"
  },
  "security-pentesting-specialist": {
    prompts: [
      "Perform security assessment of web application",
      "Identify OWASP Top 10 vulnerabilities",
      "Create penetration testing methodology",
      "Analyze API security implementation"
    ],
    expected_response_time: 7000,
    complexity: "high"
  },
  "deployment-automation-master": {
    prompts: [
      "Create CI/CD pipeline for containerized application",
      "Design blue-green deployment strategy",
      "Implement infrastructure as code with Terraform",
      "Create monitoring and alerting for production deployment"
    ],
    expected_response_time: 5000,
    complexity: "medium"
  }
};

// Default test scenario for agents without specific configuration
const DEFAULT_SCENARIO = {
  prompts: [
    "Analyze the current system requirements",
    "Provide recommendations for improvement",
    "Create implementation plan",
    "Generate best practices documentation"
  ],
  expected_response_time: 3000,
  complexity: "medium"
};

export default function() {
  const testType = randomChoice(['single_agent', 'category_test', 'cross_agent', 'stress_specific']);
  
  switch(testType) {
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
  }
  
  sleep(randomInt(1, 3));
}

function testSingleAgent() {
  const agentNames = Object.keys(ALL_AGENTS);
  const agentName = randomChoice(agentNames);
  const port = ALL_AGENTS[agentName];
  
  const scenario = AGENT_TEST_SCENARIOS[agentName] || DEFAULT_SCENARIO;
  const prompt = randomChoice(scenario.prompts);
  
  const payload = {
    prompt: prompt,
    max_tokens: scenario.complexity === 'high' ? 2000 : 1000,
    temperature: 0.7,
    stream: false,
    agent_context: {
      test_type: 'single_agent_load_test',
      complexity: scenario.complexity
    }
  };
  
  const response = http.post(`${config.baseUrl}:${port}/api/chat`, JSON.stringify(payload), {
    ...httpParams,
    timeout: `${scenario.expected_response_time + 10000}ms`,
    tags: { 
      ...httpParams.tags, 
      test_scenario: 'single_agent',
      agent: agentName,
      complexity: scenario.complexity
    }
  });
  
  validateResponse(response, 200);
  
  check(response, {
    [`${agentName} responds within SLA`]: (r) => r.timings.duration < scenario.expected_response_time,
    [`${agentName} provides meaningful response`]: (r) => {
      try {
        const body = JSON.parse(r.body);
        return body.response && body.response.length > 50;
      } catch (e) {
        return false;
      }
    }
  });
}

function testAgentCategory() {
  const categories = Object.keys(AGENT_CATEGORIES);
  const category = randomChoice(categories);
  const agentsInCategory = AGENT_CATEGORIES[category];
  
  if (agentsInCategory.length === 0) return;
  
  const agentName = randomChoice(agentsInCategory);
  const port = ALL_AGENTS[agentName];
  
  const categoryPrompts = {
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
  };
  
  const prompt = randomChoice(categoryPrompts[category] || categoryPrompts["specialized_agents"]);
  
  const payload = {
    prompt: prompt,
    max_tokens: 1500,
    temperature: 0.6,
    category_context: category
  };
  
  const response = http.post(`${config.baseUrl}:${port}/api/chat`, JSON.stringify(payload), {
    ...httpParams,
    tags: { 
      ...httpParams.tags, 
      test_scenario: 'category_test',
      agent: agentName,
      category: category
    }
  });
  
  validateResponse(response, 200);
  
  check(response, {
    'category-appropriate response': (r) => r.status === 200,
    'response time reasonable for category': (r) => r.timings.duration < 10000
  });
}

function testCrossAgentInteraction() {
  // Test interaction between agents in the same category
  const category = randomChoice(Object.keys(AGENT_CATEGORIES));
  const agentsInCategory = AGENT_CATEGORIES[category];
  
  if (agentsInCategory.length < 2) {
    // Fall back to single agent test
    testSingleAgent();
    return;
  }
  
  const agent1 = randomChoice(agentsInCategory);
  const agent2 = randomChoice(agentsInCategory.filter(a => a !== agent1));
  
  const collaborationPrompt = `Collaborate with ${agent2} to create a comprehensive solution for: ${randomChoice([
    "building a scalable web application",
    "implementing security best practices",
    "creating automated testing pipeline",
    "designing cloud infrastructure"
  ])}`;
  
  const payload = {
    prompt: collaborationPrompt,
    max_tokens: 1000,
    collaboration_context: {
      partner_agent: agent2,
      collaboration_type: "cross_agent_interaction"
    }
  };
  
  const response = http.post(`${config.baseUrl}:${ALL_AGENTS[agent1]}/api/chat`, JSON.stringify(payload), {
    ...httpParams,
    timeout: '45s',
    tags: { 
      ...httpParams.tags, 
      test_scenario: 'cross_agent_interaction',
      primary_agent: agent1,
      secondary_agent: agent2,
      category: category
    }
  });
  
  validateResponse(response, 200);
  
  check(response, {
    'cross-agent collaboration successful': (r) => r.status === 200,
    'collaboration response comprehensive': (r) => {
      try {
        const body = JSON.parse(r.body);
        return body.response && body.response.length > 200;
      } catch (e) {
        return false;
      }
    }
  });
}

function stressTestSpecificAgent() {
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
  for (let i = 0; i < 3; i++) {
    const payload = {
      prompt: `Stress test request ${i + 1}: Provide quick analysis`,
      max_tokens: 100,
      temperature: 0.3,
      stress_test: true
    };
    
    http.post(`${config.baseUrl}:${port}/api/chat`, JSON.stringify(payload), {
      ...httpParams,
      timeout: '15s',
      tags: { 
        ...httpParams.tags, 
        test_scenario: 'stress_specific',
        agent: agent,
        stress_iteration: i + 1
      }
    });
    
    sleep(0.5); // Small delay between stress requests
  }
}

// Agent availability test
export function testAllAgentsAvailability() {
  const agentNames = Object.keys(ALL_AGENTS);
  const results = {};
  
  agentNames.forEach(agentName => {
    const port = ALL_AGENTS[agentName];
    const response = http.get(`${config.baseUrl}:${port}/health`, {
      ...httpParams,
      timeout: '10s',
      tags: { 
        ...httpParams.tags, 
        test_scenario: 'availability_check',
        agent: agentName
      }
    });
    
    results[agentName] = response.status === 200;
    
    check(response, {
      [`${agentName} is available`]: (r) => r.status === 200
    });
  });
  
  const availableCount = Object.values(results).filter(Boolean).length;
  const totalCount = agentNames.length;
  
  check(null, {
    'majority of agents available': () => availableCount > totalCount * 0.8, // 80% availability
    'critical agents available': () => {
      const criticalAgents = ['ai-system-architect', 'ai-qa-team-lead', 'deployment-automation-master'];
      return criticalAgents.every(agent => results[agent] === true);
    }
  });
  
  console.log(`Agent Availability: ${availableCount}/${totalCount} (${Math.round(availableCount/totalCount*100)}%)`);
}

// Performance benchmark for all agent categories
export function benchmarkAgentCategories() {
  const categoryBenchmarks = {};
  
  Object.keys(AGENT_CATEGORIES).forEach(category => {
    const agentsInCategory = AGENT_CATEGORIES[category];
    if (agentsInCategory.length === 0) return;
    
    const sampleAgent = agentsInCategory[0];
    const port = ALL_AGENTS[sampleAgent];
    
    const startTime = Date.now();
    
    const response = http.post(`${config.baseUrl}:${port}/api/chat`, JSON.stringify({
      prompt: "Quick benchmark test",
      max_tokens: 50
    }), {
      ...httpParams,
      tags: { 
        ...httpParams.tags, 
        test_scenario: 'category_benchmark',
        category: category,
        agent: sampleAgent
      }
    });
    
    const endTime = Date.now();
    const responseTime = endTime - startTime;
    
    categoryBenchmarks[category] = {
      agent: sampleAgent,
      responseTime: responseTime,
      success: response.status === 200
    };
    
    check(response, {
      [`${category} category responsive`]: (r) => r.status === 200 && responseTime < 5000
    });
  });
  
  console.log('Category Benchmarks:', JSON.stringify(categoryBenchmarks, null, 2));
}
