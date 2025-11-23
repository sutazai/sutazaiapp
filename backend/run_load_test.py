"""
Run load tests with correct backend URL
"""
import asyncio
import sys
sys.path.insert(0, '/opt/sutazaiapp/backend')

# Import after path setup
from tests.load_test import LoadTester, test_10_concurrent_users, test_50_concurrent_users, test_100_concurrent_users
import json
from datetime import datetime

# Monkey patch the base URL
original_init = LoadTester.__init__

def patched_init(self, base_url: str = "http://localhost:10200"):
    original_init(self, base_url)

LoadTester.__init__ = patched_init


async def run_all_tests():
    """Run all load tests sequentially"""
    print("\n" + "="*60)
    print("SUTAZAI BACKEND LOAD TESTING SUITE")
    print("Backend URL: http://localhost:10200")
    print("="*60)
    
    all_results = []
    
    # Test 1: 10 users
    print("\n[1/3] Starting 10 concurrent users test...")
    results_10 = await test_10_concurrent_users()
    all_results.append(results_10)
    await asyncio.sleep(5)  # Cool down between tests
    
    # Test 2: 50 users
    print("\n[2/3] Starting 50 concurrent users test...")
    results_50 = await test_50_concurrent_users()
    all_results.append(results_50)
    await asyncio.sleep(5)
    
    # Test 3: 100 users
    print("\n[3/3] Starting 100 concurrent users test...")
    results_100 = await test_100_concurrent_users()
    all_results.append(results_100)
    
    # Save results to file
    output_file = f"load_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n✓ Results saved to {output_file}")
    
    # Final summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    for result in all_results:
        status = "✓ PASSED" if result["summary"]["passed"] else "✗ FAILED"
        print(f"{result['test_name']}: {status} ({result['summary']['overall_success_rate']:.1f}% success)")
    print("="*60 + "\n")
    
    return all_results


if __name__ == "__main__":
    results = asyncio.run(run_all_tests())
    
    # Exit with appropriate code
    all_passed = all(r["summary"]["passed"] for r in results)
    sys.exit(0 if all_passed else 1)
