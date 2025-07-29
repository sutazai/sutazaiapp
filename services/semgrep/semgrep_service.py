from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import subprocess
import json
import os
import tempfile
from typing import List, Dict, Any, Optional

app = FastAPI(title="Semgrep Code Analysis Service")

class CodeAnalysisRequest(BaseModel):
    code: str
    language: Optional[str] = None
    rules: Optional[List[str]] = ["auto"]
    config: Optional[str] = "auto"

class FileAnalysisRequest(BaseModel):
    file_path: str
    rules: Optional[List[str]] = ["auto"]
    config: Optional[str] = "auto"

class ProjectAnalysisRequest(BaseModel):
    project_path: str
    rules: Optional[List[str]] = ["auto"]
    config: Optional[str] = "auto"
    exclude: Optional[List[str]] = []

def run_semgrep(target: str, config: str = "auto", extra_args: List[str] = None) -> Dict[str, Any]:
    """Run semgrep and return results"""
    cmd = ["semgrep", "--json", f"--config={config}", target]
    
    if extra_args:
        cmd.extend(extra_args)
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode == 0:
            return json.loads(result.stdout)
        else:
            error_msg = result.stderr or "Unknown error"
            raise Exception(f"Semgrep failed: {error_msg}")
    
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=408, detail="Analysis timeout")
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Failed to parse semgrep output")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze/code")
async def analyze_code(request: CodeAnalysisRequest):
    """Analyze code snippet"""
    try:
        # Create temporary file with code
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py' if request.language == 'python' else '.txt', delete=False) as f:
            f.write(request.code)
            temp_file = f.name
        
        try:
            # Determine config
            config = request.config
            if config == "auto":
                if request.language == "python":
                    config = "p/python"
                elif request.language == "javascript":
                    config = "p/javascript"
                elif request.language == "go":
                    config = "p/golang"
                else:
                    config = "auto"
            
            # Run analysis
            results = run_semgrep(temp_file, config)
            
            # Process results
            findings = []
            for result in results.get("results", []):
                findings.append({
                    "rule_id": result.get("check_id"),
                    "severity": result.get("extra", {}).get("severity", "INFO"),
                    "message": result.get("extra", {}).get("message", ""),
                    "line": result.get("start", {}).get("line"),
                    "column": result.get("start", {}).get("col"),
                    "end_line": result.get("end", {}).get("line"),
                    "end_column": result.get("end", {}).get("col"),
                    "code": result.get("lines", ""),
                    "fix": result.get("extra", {}).get("fix", None)
                })
            
            return {
                "status": "completed",
                "findings": findings,
                "total_findings": len(findings),
                "summary": {
                    "errors": len([f for f in findings if f["severity"] == "ERROR"]),
                    "warnings": len([f for f in findings if f["severity"] == "WARNING"]),
                    "info": len([f for f in findings if f["severity"] == "INFO"])
                }
            }
        
        finally:
            # Clean up temp file
            if os.path.exists(temp_file):
                os.unlink(temp_file)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze/file")
async def analyze_file(request: FileAnalysisRequest):
    """Analyze a single file"""
    try:
        # Verify file exists and is within workspace
        file_path = request.file_path
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
        
        # Ensure file is within /workspace
        if not os.path.abspath(file_path).startswith("/workspace"):
            raise HTTPException(status_code=403, detail="Access denied: file outside workspace")
        
        # Run analysis
        results = run_semgrep(file_path, request.config)
        
        # Process results (similar to code analysis)
        findings = []
        for result in results.get("results", []):
            findings.append({
                "rule_id": result.get("check_id"),
                "severity": result.get("extra", {}).get("severity", "INFO"),
                "message": result.get("extra", {}).get("message", ""),
                "file": result.get("path"),
                "line": result.get("start", {}).get("line"),
                "column": result.get("start", {}).get("col"),
                "end_line": result.get("end", {}).get("line"),
                "end_column": result.get("end", {}).get("col"),
                "code": result.get("lines", ""),
                "fix": result.get("extra", {}).get("fix", None)
            })
        
        return {
            "status": "completed",
            "file": file_path,
            "findings": findings,
            "total_findings": len(findings),
            "summary": {
                "errors": len([f for f in findings if f["severity"] == "ERROR"]),
                "warnings": len([f for f in findings if f["severity"] == "WARNING"]),
                "info": len([f for f in findings if f["severity"] == "INFO"])
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze/project")
async def analyze_project(request: ProjectAnalysisRequest):
    """Analyze entire project"""
    try:
        # Verify project path exists
        project_path = request.project_path
        if not os.path.exists(project_path):
            raise HTTPException(status_code=404, detail=f"Project not found: {project_path}")
        
        # Ensure path is within /workspace
        if not os.path.abspath(project_path).startswith("/workspace"):
            raise HTTPException(status_code=403, detail="Access denied: project outside workspace")
        
        # Build exclude args
        extra_args = []
        for exclude in request.exclude:
            extra_args.extend(["--exclude", exclude])
        
        # Run analysis
        results = run_semgrep(project_path, request.config, extra_args)
        
        # Process results by file
        findings_by_file = {}
        all_findings = []
        
        for result in results.get("results", []):
            file_path = result.get("path", "unknown")
            finding = {
                "rule_id": result.get("check_id"),
                "severity": result.get("extra", {}).get("severity", "INFO"),
                "message": result.get("extra", {}).get("message", ""),
                "file": file_path,
                "line": result.get("start", {}).get("line"),
                "column": result.get("start", {}).get("col"),
                "end_line": result.get("end", {}).get("line"),
                "end_column": result.get("end", {}).get("col"),
                "code": result.get("lines", ""),
                "fix": result.get("extra", {}).get("fix", None)
            }
            
            all_findings.append(finding)
            
            if file_path not in findings_by_file:
                findings_by_file[file_path] = []
            findings_by_file[file_path].append(finding)
        
        return {
            "status": "completed",
            "project": project_path,
            "total_findings": len(all_findings),
            "files_analyzed": len(findings_by_file),
            "findings_by_file": findings_by_file,
            "summary": {
                "errors": len([f for f in all_findings if f["severity"] == "ERROR"]),
                "warnings": len([f for f in all_findings if f["severity"] == "WARNING"]),
                "info": len([f for f in all_findings if f["severity"] == "INFO"])
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/rules")
async def list_rules():
    """List available rule sets"""
    return {
        "default_configs": [
            "auto",
            "p/python",
            "p/javascript", 
            "p/typescript",
            "p/go",
            "p/java",
            "p/ruby",
            "p/generic",
            "p/security-audit",
            "p/owasp-top-ten",
            "p/cwe-top-25"
        ],
        "categories": {
            "security": ["p/security-audit", "p/owasp-top-ten", "p/cwe-top-25"],
            "language": ["p/python", "p/javascript", "p/typescript", "p/go", "p/java", "p/ruby"],
            "quality": ["p/ci", "p/correctness", "p/performance"]
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Test semgrep is available
        result = subprocess.run(
            ["semgrep", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        version = result.stdout.strip() if result.returncode == 0 else "unknown"
        
        return {
            "status": "healthy",
            "service": "semgrep-analysis",
            "version": version
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }