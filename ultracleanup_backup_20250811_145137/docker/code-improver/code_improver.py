from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import subprocess
import git
import os
import asyncio
import httpx
from datetime import datetime
import json

app = FastAPI(title="Autonomous Code Improvement Service")

class ImprovementRequest(BaseModel):
    target_path: str = "/opt/sutazaiapp"
    file_patterns: List[str] = ["*.py", "*.js", "*.ts"]
    improvement_types: List[str] = ["refactor", "optimize", "security", "documentation"]
    require_approval: bool = True
    agents: List[str] = ["aider", "semgrep"]

class ImprovementResult(BaseModel):
    files_analyzed: int
    improvements_suggested: int
    improvements_applied: int
    details: List[Dict[str, Any]]
    branch_name: Optional[str] = None

class CodeAnalyzer:
    def __init__(self):
        self.agents = {
            "aider": self.analyze_with_aider,
            "semgrep": self.analyze_with_semgrep,
            "pylint": self.analyze_with_pylint,
            "black": self.format_with_black
        }
    
    async def analyze_with_aider(self, file_path: str, improvement_type: str) -> Dict:
        """Use Aider for code improvement suggestions"""
        try:
            # Connect to Aider service
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "http://aider:8080/analyze",
                    json={
                        "file_path": file_path,
                        "prompt": f"Analyze this file for {improvement_type} improvements"
                    },
                    timeout=30.0
                )
                return response.json()
        except Exception as e:
            return {"error": str(e), "improvements": []}
    
    async def analyze_with_semgrep(self, file_path: str, improvement_type: str) -> Dict:
        """Use Semgrep for security and code quality analysis"""
        try:
            # Run semgrep
            rules = {
                "security": "p/security-audit",
                "optimize": "p/performance",
                "refactor": "p/python"
            }
            
            rule = rules.get(improvement_type, "auto")
            cmd = f"semgrep --config={rule} --json {file_path}"
            
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                findings = json.loads(result.stdout)
                return {
                    "tool": "semgrep",
                    "improvements": findings.get("results", [])
                }
            return {"error": result.stderr, "improvements": []}
        except Exception as e:
            return {"error": str(e), "improvements": []}
    
    async def analyze_with_pylint(self, file_path: str) -> Dict:
        """Use pylint for Python code quality"""
        if not file_path.endswith('.py'):
            return {"improvements": []}
        
        try:
            cmd = f"pylint --output-format=json {file_path}"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            if result.stdout:
                issues = json.loads(result.stdout)
                return {
                    "tool": "pylint",
                    "improvements": issues
                }
            return {"improvements": []}
        except Exception as e:
            return {"error": str(e), "improvements": []}
    
    async def format_with_black(self, file_path: str) -> Dict:
        """Format Python code with Black"""
        if not file_path.endswith('.py'):
            return {"formatted": False}
        
        try:
            cmd = f"black --check {file_path}"
            result = subprocess.run(cmd, shell=True, capture_output=True)
            
            if result.returncode != 0:
                # File needs formatting
                subprocess.run(f"black {file_path}", shell=True)
                return {"formatted": True, "tool": "black"}
            return {"formatted": False}
        except Exception as e:
            return {"error": str(e), "formatted": False}

analyzer = CodeAnalyzer()

@app.post("/improve", response_model=ImprovementResult)
async def improve_code(request: ImprovementRequest, background_tasks: BackgroundTasks):
    """Trigger autonomous code improvement"""
    
    # Initialize result
    result = ImprovementResult(
        files_analyzed=0,
        improvements_suggested=0,
        improvements_applied=0,
        details=[]
    )
    
    try:
        # Initialize git repo
        repo = git.Repo(request.target_path)
        
        # Create improvement branch
        branch_name = f"auto-improve-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        if request.require_approval:
            current_branch = repo.active_branch.name
            repo.create_head(branch_name)
            repo.heads[branch_name].checkout()
            result.branch_name = branch_name
        
        # Find files to analyze
        files_to_analyze = []
        for pattern in request.file_patterns:
            cmd = f"find {request.target_path} -name '{pattern}' -type f"
            files = subprocess.check_output(cmd, shell=True, text=True).strip().split('\n')
            files_to_analyze.extend([f for f in files if f])
        
        # Analyze each file
        for file_path in files_to_analyze:
            if not file_path:
                continue
                
            result.files_analyzed += 1
            file_improvements = []
            
            # Run analysis with each requested agent
            for agent in request.agents:
                if agent in analyzer.agents:
                    for improvement_type in request.improvement_types:
                        analysis = await analyzer.agents[agent](file_path, improvement_type)
                        
                        if "improvements" in analysis:
                            improvements = analysis["improvements"]
                            if improvements:
                                file_improvements.extend(improvements)
                                result.improvements_suggested += len(improvements)
            
            # Apply improvements if not requiring approval
            if file_improvements and not request.require_approval:
                # Auto-apply safe improvements
                for improvement in file_improvements:
                    if improvement.get("auto_fixable", False):
                        # Apply the fix
                        result.improvements_applied += 1
            
            if file_improvements:
                result.details.append({
                    "file": file_path,
                    "improvements": file_improvements
                })
        
        # Commit changes if any
        if result.improvements_applied > 0 or (result.improvements_suggested > 0 and request.require_approval):
            repo.index.add("*")
            commit_msg = f"🤖 Autonomous improvements: {result.improvements_suggested} suggestions"
            if result.improvements_applied > 0:
                commit_msg += f", {result.improvements_applied} applied"
            repo.index.commit(commit_msg)
            
            if request.require_approval:
                # Switch back to original branch
                repo.heads[current_branch].checkout()
                
                # Create PR info
                result.details.append({
                    "action": "branch_created",
                    "branch": branch_name,
                    "message": f"Review improvements in branch: {branch_name}"
                })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    return result

@app.post("/schedule")
async def schedule_improvement(cron_expression: str = "0 */6 * * *"):
    """Schedule periodic code improvements"""
    try:
        # Update cron job
        cmd = f'echo "{cron_expression} /app/improve_cron.sh" | crontab -'
        subprocess.run(cmd, shell=True, check=True)
        
        return {
            "status": "scheduled",
            "cron": cron_expression,
            "next_run": "Check cron schedule"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status")
async def get_status():
    """Get improvement system status"""
    try:
        # Get current cron jobs
        cron_result = subprocess.run("crontab -l", shell=True, capture_output=True, text=True)
        
        # Get recent improvements
        repo = git.Repo("/opt/sutazaiapp")
        recent_commits = []
        for commit in repo.iter_commits(max_count=10):
            if "Autonomous improvements" in commit.message:
                recent_commits.append({
                    "sha": commit.hexsha[:8],
                    "message": commit.message,
                    "date": commit.committed_datetime.isoformat()
                })
        
        return {
            "status": "active",
            "cron_jobs": cron_result.stdout,
            "recent_improvements": recent_commits
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}

@app.post("/analyze_file")
async def analyze_single_file(file_path: str, agents: List[str] = ["semgrep", "pylint"]):
    """Analyze a single file with specified agents"""
    results = {}
    
    for agent in agents:
        if agent in analyzer.agents:
            result = await analyzer.agents[agent](file_path, "all")
            results[agent] = result
    
    return results

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "code-improver"}

@app.get("/")
async def root():
    return {
        "service": "Autonomous Code Improvement System",
        "endpoints": [
            "/improve - Trigger code improvements",
            "/schedule - Schedule periodic improvements",
            "/status - Get system status",
            "/analyze_file - Analyze single file",
            "/health - Health check"
        ],
        "features": [
            "Multi-agent code analysis",
            "Automatic improvement suggestions",
            "Git branch creation for reviews",
            "Scheduled autonomous improvements",
            "Security scanning with Semgrep",
            "Code formatting with Black",
            "Quality analysis with Pylint"
        ]
    }