# Rule Combination Testing Report
        
## Test Suite Summary
- **Suite ID**: comprehensive_test_20250803_214345
- **Start Time**: 2025-08-03 21:43:45.900906
- **End Time**: 2025-08-03 21:43:46.525964
- **Duration**: 0.63 seconds
- **Total Combinations**: 5
- **Completed Tests**: 5
- **Successful Tests**: 3
- **Failed Tests**: 2
- **Success Rate**: 60.0%

## Performance Summary
{
  "impact_statistics": {
    "min": 0.0,
    "max": 0.0,
    "mean": 0.0,
    "median": 0.0
  },
  "duration_statistics": {
    "min": 0.203924,
    "max": 0.204456,
    "mean": 0.20424733333333334,
    "median": 0.204362
  },
  "high_impact_combinations": [
    {
      "combination_id": "baseline_000000",
      "enabled_rules": [],
      "performance_impact": 0.0
    },
    {
      "combination_id": "individual_01",
      "enabled_rules": [
        1
      ],
      "performance_impact": 0.0
    },
    {
      "combination_id": "individual_03",
      "enabled_rules": [
        3
      ],
      "performance_impact": 0.0
    }
  ],
  "fastest_combinations": [
    {
      "combination_id": "individual_03",
      "enabled_rules": [
        3
      ],
      "duration": 0.203924
    },
    {
      "combination_id": "individual_01",
      "enabled_rules": [
        1
      ],
      "duration": 0.204362
    },
    {
      "combination_id": "baseline_000000",
      "enabled_rules": [],
      "duration": 0.204456
    }
  ]
}

## Conflict Analysis
{}

## Recommendations
1. Success rate is 60.0%. Investigate and fix failing test scenarios to improve system reliability.

## Detailed Results
- Database location: /opt/sutazaiapp/logs/rule_test_results.db
- Log files: /opt/sutazaiapp/logs/rule-combination-testing.log
- Test suite completed at: 2025-08-03 21:43:46.526382

## System Information
- Python version: 3.12.3 (main, Jun 18 2025, 17:59:45) [GCC 13.3.0]
- Platform: linux
- CPU count: 12
- Memory limit: 8.0 GB
