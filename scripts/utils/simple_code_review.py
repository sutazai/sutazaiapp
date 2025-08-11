#!/usr/bin/env python3
"""
Simple Code Review Workflow
A practical example of using SutazAI for automated code review
"""

import asyncio
import httpx
import os
from pathlib import Path
from typing import List, Dict, Any

# API configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")


class CodeReviewWorkflow:
    """Simple code review workflow using local AI"""
    
    def __init__(self):
        self.api_url = f"{API_BASE_URL}/api/v1"
        self.client = httpx.AsyncClient(timeout=30.0)
    
    async def review_file(self, file_path: str) -> Dict[str, Any]:
        """Review a single code file"""
        print(f"📄 Reviewing: {file_path}")
        
        # Read the file
        try:
            with open(file_path, 'r') as f:
                code_content = f.read()
        except Exception as e:
            return {"error": f"Failed to read file: {e}"}
        
        # Prepare review request
        review_request = {
            "agent": "code-generation-improver",
            "task": "review",
            "data": {
                "file_path": file_path,
                "code": code_content,
                "checks": [
                    "code_quality",
                    "best_practices", 
                    "potential_bugs",
                    "performance",
                    "security"
                ]
            }
        }
        
        # Send to API
        try:
            response = await self.client.post(
                f"{self.api_url}/agents/execute",
                json=review_request
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": f"API request failed: {e}"}
    
    async def review_directory(self, directory: str, extensions: List[str] = [".py"]) -> Dict[str, Any]:
        """Review all code files in a directory"""
        print(f"📁 Reviewing directory: {directory}")
        
        # Find all matching files
        path = Path(directory)
        files = []
        for ext in extensions:
            files.extend(path.rglob(f"*{ext}"))
        
        print(f"Found {len(files)} files to review")
        
        # Review each file
        results = []
        for file_path in files:
            result = await self.review_file(str(file_path))
            results.append({
                "file": str(file_path),
                "review": result
            })
        
        return {
            "directory": directory,
            "files_reviewed": len(files),
            "results": results
        }
    
    async def generate_report(self, review_results: Dict[str, Any]) -> str:
        """Generate a markdown report from review results"""
        report = ["# Code Review Report", ""]
        report.append(f"**Directory**: `{review_results['directory']}`")
        report.append(f"**Files Reviewed**: {review_results['files_reviewed']}")
        report.append("")
        
        # Summary statistics
        total_issues = 0
        issues_by_type = {}
        
        for file_result in review_results['results']:
            if 'error' not in file_result['review']:
                issues = file_result['review'].get('issues', [])
                total_issues += len(issues)
                
                for issue in issues:
                    issue_type = issue.get('type', 'other')
                    issues_by_type[issue_type] = issues_by_type.get(issue_type, 0) + 1
        
        report.append("## Summary")
        report.append(f"- Total Issues Found: {total_issues}")
        report.append("- Issues by Type:")
        for issue_type, count in issues_by_type.items():
            report.append(f"  - {issue_type}: {count}")
        report.append("")
        
        # Detailed findings
        report.append("## Detailed Findings")
        
        for file_result in review_results['results']:
            report.append(f"\n### `{file_result['file']}`")
            
            if 'error' in file_result['review']:
                report.append(f"❌ Error: {file_result['review']['error']}")
            else:
                review = file_result['review']
                issues = review.get('issues', [])
                
                if not issues:
                    report.append("✅ No issues found")
                else:
                    for i, issue in enumerate(issues, 1):
                        report.append(f"\n{i}. **{issue.get('severity', 'INFO')}**: {issue.get('message', 'No message')}")
                        if 'line' in issue:
                            report.append(f"   - Line: {issue['line']}")
                        if 'suggestion' in issue:
                            report.append(f"   - Suggestion: {issue['suggestion']}")
        
        return "\n".join(report)
    
    async def close(self):
        """Clean up resources"""
        await self.client.aclose()


async def main():
    """Example usage"""
    workflow = CodeReviewWorkflow()
    
    # Review a specific directory
    directory = "./backend/app"  # Change this to your target directory
    
    print("🚀 Starting code review workflow...")
    
    try:
        # Perform review
        results = await workflow.review_directory(directory, extensions=[".py"])
        
        # Generate report
        report = await workflow.generate_report(results)
        
        # Save report
        report_path = "code_review_report.md"
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"\n✅ Review complete! Report saved to: {report_path}")
        
        # Print summary
        print("\n📊 Summary:")
        print(f"- Files reviewed: {results['files_reviewed']}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    
    finally:
        await workflow.close()


if __name__ == "__main__":
    asyncio.run(main())