"""
ULTRAORGANIZE Engine - Advanced Frontend Organization Intelligence

Provides comprehensive file and component organization optimization with intelligent
structure analysis, dependency mapping, and automated reorganization capabilities.
"""

import os
import ast
import json
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import logging
import asyncio

logger = logging.getLogger(__name__)

@dataclass
class OrganizationRule:
    """Rule for file organization optimization"""
    pattern: str
    target_directory: str
    organization_type: str
    priority: int
    description: str

@dataclass
class FileAnalysis:
    """Analysis results for a single file"""
    file_path: str
    file_type: str
    current_location: str
    optimal_location: str
    dependencies: List[str]
    complexity_score: float
    organization_score: float
    recommended_actions: List[str]

@dataclass
class OrganizationResults:
    """Results from ULTRAORGANIZE optimization"""
    organization_score: float
    files_analyzed: int
    files_moved: int
    directories_created: int
    patterns_discovered: Dict[str, Any]
    performance_improvement: float
    recommendations: List[str]
    detailed_analysis: List[FileAnalysis]

class UltraOrganizeEngine:
    """Advanced frontend organization intelligence engine"""
    
    def __init__(self, config):
        self.config = config
        self.organization_rules = []
        self.dependency_graph = {}
        self.patterns_discovered = {}
        self.performance_baseline = {}
        self.initialized = False
    
    async def initialize(self):
        """Initialize ULTRAORGANIZE engine with rules and patterns"""
        try:
            logger.info("🔧 Initializing ULTRAORGANIZE engine...")
            
            # Load organization rules
            await self._load_organization_rules()
            
            # Initialize dependency analyzer
            await self._initialize_dependency_analyzer()
            
            # Load pattern recognition models
            await self._load_pattern_models()
            
            self.initialized = True
            logger.info("✅ ULTRAORGANIZE engine initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize ULTRAORGANIZE engine: {e}")
            raise
    
    async def optimize_frontend_structure(self, target_path: str, validate_only: bool = False) -> OrganizationResults:
        """Apply ULTRAORGANIZE optimization to frontend structure"""
        if not self.initialized:
            raise RuntimeError("ULTRAORGANIZE engine not initialized")
        
        logger.info(f"🎯 Starting ULTRAORGANIZE optimization: {target_path}")
        
        try:
            # Phase 1: Deep structure analysis
            analysis_results = await self._analyze_frontend_structure(target_path)
            
            # Phase 2: Generate optimization plan
            optimization_plan = await self._generate_optimization_plan(analysis_results)
            
            # Phase 3: Execute organization (if not validate_only)
            execution_results = {}
            if not validate_only:
                execution_results = await self._execute_organization(optimization_plan, target_path)
            
            # Phase 4: Calculate improvements and patterns
            patterns = await self._discover_organization_patterns(analysis_results, optimization_plan)
            performance_improvement = await self._calculate_performance_improvement(analysis_results, execution_results)
            
            # Phase 5: Generate comprehensive results
            results = OrganizationResults(
                organization_score=await self._calculate_organization_score(analysis_results, execution_results),
                files_analyzed=analysis_results.get('total_files', 0),
                files_moved=execution_results.get('files_moved', 0),
                directories_created=execution_results.get('directories_created', 0),
                patterns_discovered=patterns,
                performance_improvement=performance_improvement,
                recommendations=await self._generate_organization_recommendations(analysis_results),
                detailed_analysis=analysis_results.get('file_analyses', [])
            )
            
            logger.info(f"✅ ULTRAORGANIZE optimization completed. Score: {results.organization_score:.2f}")
            return results
            
        except Exception as e:
            logger.error(f"❌ ULTRAORGANIZE optimization failed: {e}")
            raise
    
    async def _analyze_frontend_structure(self, target_path: str) -> Dict[str, Any]:
        """Deep analysis of current frontend structure"""
        logger.info("🔍 Analyzing frontend structure...")
        
        analysis = {
            'total_files': 0,
            'directory_structure': {},
            'file_analyses': [],
            'dependency_graph': {},
            'component_hierarchy': {},
            'asset_distribution': {},
            'configuration_files': [],
            'documentation_coverage': {},
            'organizational_issues': [],
            'optimization_opportunities': []
        }
        
        # Walk through all files in target directory
        for root, dirs, files in os.walk(target_path):
            # Skip hidden directories and common ignore patterns
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'node_modules']]
            
            for file in files:
                if file.startswith('.'):
                    continue
                
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, target_path)
                
                analysis['total_files'] += 1
                
                # Analyze individual file
                file_analysis = await self._analyze_file(file_path, relative_path, target_path)
                analysis['file_analyses'].append(file_analysis)
                
                # Update dependency graph
                if file_analysis.dependencies:
                    analysis['dependency_graph'][relative_path] = file_analysis.dependencies
                
                # Analyze directory structure patterns
                await self._analyze_directory_pattern(file_path, analysis['directory_structure'])
        
        # Analyze component hierarchy
        analysis['component_hierarchy'] = await self._analyze_component_hierarchy(analysis['file_analyses'])
        
        # Analyze asset distribution
        analysis['asset_distribution'] = await self._analyze_asset_distribution(analysis['file_analyses'])
        
        # Identify organizational issues
        analysis['organizational_issues'] = await self._identify_organizational_issues(analysis)
        
        # Find optimization opportunities
        analysis['optimization_opportunities'] = await self._find_optimization_opportunities(analysis)
        
        logger.info(f"📊 Structure analysis completed: {analysis['total_files']} files analyzed")
        return analysis
    
    async def _analyze_file(self, file_path: str, relative_path: str, target_path: str) -> FileAnalysis:
        """Analyze individual file for organization optimization"""
        file_type = self._determine_file_type(file_path)
        current_location = os.path.dirname(relative_path)
        
        # Analyze dependencies
        dependencies = await self._extract_dependencies(file_path, file_type)
        
        # Calculate complexity score
        complexity_score = await self._calculate_file_complexity(file_path, file_type)
        
        # Determine optimal location
        optimal_location = await self._determine_optimal_location(file_path, relative_path, dependencies, file_type)
        
        # Calculate current organization score
        organization_score = await self._calculate_file_organization_score(
            current_location, optimal_location, dependencies, file_type
        )
        
        # Generate recommendations
        recommendations = await self._generate_file_recommendations(
            file_path, current_location, optimal_location, dependencies, complexity_score
        )
        
        return FileAnalysis(
            file_path=relative_path,
            file_type=file_type,
            current_location=current_location,
            optimal_location=optimal_location,
            dependencies=dependencies,
            complexity_score=complexity_score,
            organization_score=organization_score,
            recommended_actions=recommendations
        )
    
    def _determine_file_type(self, file_path: str) -> str:
        """Determine file type for organization purposes"""
        extension = Path(file_path).suffix.lower()
        filename = Path(file_path).name.lower()
        
        # Component files
        if extension == '.py' and any(keyword in filename for keyword in ['component', 'widget', 'ui']):
            return 'component'
        
        # Page files
        if extension == '.py' and 'page' in filename:
            return 'page'
        
        # Utility files
        if extension == '.py' and any(keyword in filename for keyword in ['util', 'helper', 'tool']):
            return 'utility'
        
        # Service files
        if extension == '.py' and any(keyword in filename for keyword in ['service', 'client', 'api']):
            return 'service'
        
        # Configuration files
        if filename in ['config.py', 'settings.py', 'constants.py'] or 'config' in filename:
            return 'configuration'
        
        # Test files
        if filename.startswith('test_') or '_test' in filename or extension in ['.test']:
            return 'test'
        
        # Documentation files
        if extension in ['.md', '.rst', '.txt'] and any(keyword in filename for keyword in ['readme', 'doc', 'guide']):
            return 'documentation'
        
        # Asset files
        if extension in ['.css', '.scss', '.sass', '.less']:
            return 'style'
        if extension in ['.js', '.ts', '.jsx', '.tsx']:
            return 'script'
        if extension in ['.png', '.jpg', '.jpeg', '.gif', '.svg', '.ico']:
            return 'image'
        if extension in ['.ttf', '.woff', '.woff2', '.eot']:
            return 'font'
        
        # Requirements and package files
        if filename in ['requirements.txt', 'requirements_optimized.txt', 'package.json', 'yarn.lock']:
            return 'dependency'
        
        # Python source files
        if extension == '.py':
            return 'python'
        
        # Other files
        return 'other'
    
    async def _extract_dependencies(self, file_path: str, file_type: str) -> List[str]:
        """Extract dependencies from file"""
        dependencies = []
        
        if file_type in ['python', 'component', 'page', 'utility', 'service', 'configuration']:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Parse Python AST to extract imports
                try:
                    tree = ast.parse(content)
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Import):
                            for alias in node.names:
                                dependencies.append(alias.name)
                        elif isinstance(node, ast.ImportFrom):
                            if node.module:
                                dependencies.append(node.module)
                except SyntaxError:
                    # File might not be valid Python, extract imports with regex
                    import re
                    import_pattern = r'^\s*(?:from\s+(\S+)\s+)?import\s+([^\n]+)'
                    matches = re.findall(import_pattern, content, re.MULTILINE)
                    for match in matches:
                        if match[0]:  # from X import Y
                            dependencies.append(match[0])
                        else:  # import X
                            deps = [dep.strip() for dep in match[1].split(',')]
                            dependencies.extend(deps)
            
            except Exception as e:
                logger.debug(f"Could not extract dependencies from {file_path}: {e}")
        
        return list(set(dependencies))  # Remove duplicates
    
    async def _calculate_file_complexity(self, file_path: str, file_type: str) -> float:
        """Calculate complexity score for file"""
        if file_type not in ['python', 'component', 'page', 'utility', 'service']:
            return 0.0
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Simple complexity metrics
            lines_of_code = len([line for line in content.split('\n') if line.strip() and not line.strip().startswith('#')])
            functions_count = content.count('def ')
            classes_count = content.count('class ')
            imports_count = content.count('import ')
            
            # Normalize complexity score (0.0 to 1.0)
            complexity = min(1.0, (lines_of_code * 0.01 + functions_count * 0.1 + classes_count * 0.2 + imports_count * 0.05) / 10)
            return complexity
            
        except Exception:
            return 0.0
    
    async def _determine_optimal_location(self, file_path: str, relative_path: str, dependencies: List[str], file_type: str) -> str:
        """Determine optimal location for file based on type and dependencies"""
        
        # Atomic design structure mapping
        organization_map = {
            'component': {
                'simple': 'components/atoms',
                'medium': 'components/molecules',
                'complex': 'components/organisms'
            },
            'page': 'pages',
            'utility': 'utils',
            'service': 'services',
            'configuration': 'config',
            'test': 'tests',
            'documentation': 'docs',
            'style': 'styles',
            'script': 'assets/scripts',
            'image': 'assets/images',
            'font': 'assets/fonts',
            'dependency': '.',  # Root level
            'python': 'src'
        }
        
        # Get base location for file type
        if file_type == 'component':
            # Determine component complexity for atomic design placement
            complexity = await self._calculate_file_complexity(file_path, file_type)
            if complexity < 0.3:
                return organization_map['component']['simple']
            elif complexity < 0.7:
                return organization_map['component']['medium']
            else:
                return organization_map['component']['complex']
        
        return organization_map.get(file_type, 'misc')
    
    async def _calculate_file_organization_score(self, current_location: str, optimal_location: str, 
                                               dependencies: List[str], file_type: str) -> float:
        """Calculate organization score for file (0.0 to 1.0)"""
        score = 1.0
        
        # Penalty for being in wrong location
        if current_location != optimal_location:
            score -= 0.4
        
        # Penalty for deep nesting
        depth = len(current_location.split('/')) if current_location else 0
        if depth > 3:
            score -= 0.2 * (depth - 3)
        
        # Bonus for following naming conventions
        filename = Path(current_location).name if current_location else ''
        if file_type in filename.lower():
            score += 0.1
        
        return max(0.0, min(1.0, score))
    
    async def _generate_file_recommendations(self, file_path: str, current_location: str, 
                                           optimal_location: str, dependencies: List[str], 
                                           complexity_score: float) -> List[str]:
        """Generate recommendations for file organization"""
        recommendations = []
        
        if current_location != optimal_location:
            recommendations.append(f"Move to {optimal_location} for better organization")
        
        if complexity_score > 0.8:
            recommendations.append("Consider breaking down into smaller components")
        
        if len(dependencies) > 10:
            recommendations.append("High dependency count - consider refactoring")
        
        return recommendations
    
    async def _generate_optimization_plan(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive optimization plan"""
        plan = {
            'file_moves': [],
            'directory_creation': [],
            'component_consolidation': [],
            'asset_reorganization': [],
            'configuration_centralization': [],
            'documentation_enhancement': [],
            'import_optimization': []
        }
        
        # Generate file move operations
        for file_analysis in analysis_results.get('file_analyses', []):
            if file_analysis.current_location != file_analysis.optimal_location:
                plan['file_moves'].append({
                    'source': file_analysis.file_path,
                    'target': os.path.join(file_analysis.optimal_location, os.path.basename(file_analysis.file_path)),
                    'reason': f"Optimize {file_analysis.file_type} organization",
                    'priority': 1.0 - file_analysis.organization_score
                })
        
        # Generate directory structure optimizations
        required_directories = set()
        for move in plan['file_moves']:
            required_directories.add(os.path.dirname(move['target']))
        
        for directory in required_directories:
            plan['directory_creation'].append({
                'path': directory,
                'purpose': 'Optimized file organization'
            })
        
        return plan
    
    async def _execute_organization(self, optimization_plan: Dict[str, Any], target_path: str) -> Dict[str, Any]:
        """Execute organization optimization plan"""
        logger.info("🚀 Executing ULTRAORGANIZE optimization plan...")
        
        results = {
            'files_moved': 0,
            'directories_created': 0,
            'errors': [],
            'completed_operations': []
        }
        
        try:
            # Create required directories
            for dir_op in optimization_plan.get('directory_creation', []):
                dir_path = os.path.join(target_path, dir_op['path'])
                try:
                    os.makedirs(dir_path, exist_ok=True)
                    results['directories_created'] += 1
                    results['completed_operations'].append(f"Created directory: {dir_op['path']}")
                except Exception as e:
                    results['errors'].append(f"Failed to create directory {dir_op['path']}: {e}")
            
            # Execute file moves (sorted by priority)
            file_moves = sorted(optimization_plan.get('file_moves', []), 
                              key=lambda x: x.get('priority', 0), reverse=True)
            
            for move_op in file_moves:
                source_path = os.path.join(target_path, move_op['source'])
                target_file_path = os.path.join(target_path, move_op['target'])
                
                try:
                    # Ensure target directory exists
                    os.makedirs(os.path.dirname(target_file_path), exist_ok=True)
                    
                    # Move file
                    shutil.move(source_path, target_file_path)
                    results['files_moved'] += 1
                    results['completed_operations'].append(f"Moved: {move_op['source']} → {move_op['target']}")
                    
                except Exception as e:
                    results['errors'].append(f"Failed to move {move_op['source']}: {e}")
            
            logger.info(f"✅ Organization execution completed: {results['files_moved']} files moved, {results['directories_created']} directories created")
            
        except Exception as e:
            logger.error(f"❌ Organization execution failed: {e}")
            results['errors'].append(f"Execution failed: {e}")
        
        return results
    
    async def _calculate_organization_score(self, analysis_results: Dict[str, Any], 
                                          execution_results: Dict[str, Any]) -> float:
        """Calculate overall organization score"""
        if not analysis_results.get('file_analyses'):
            return 0.0
        
        # Calculate average organization score across all files
        total_score = sum(file_analysis.organization_score for file_analysis in analysis_results['file_analyses'])
        base_score = total_score / len(analysis_results['file_analyses'])
        
        # Bonus for successful moves
        if execution_results:
            move_bonus = min(0.2, execution_results.get('files_moved', 0) * 0.01)
            base_score += move_bonus
        
        return min(1.0, base_score)
    
    async def _discover_organization_patterns(self, analysis_results: Dict[str, Any], 
                                            optimization_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Discover organization patterns for system intelligence"""
        patterns = {
            'file_type_distribution': {},
            'directory_depth_patterns': {},
            'dependency_patterns': {},
            'naming_conventions': {},
            'optimization_impact': {}
        }
        
        # Analyze file type distribution
        file_types = {}
        for file_analysis in analysis_results.get('file_analyses', []):
            file_type = file_analysis.file_type
            file_types[file_type] = file_types.get(file_type, 0) + 1
        patterns['file_type_distribution'] = file_types
        
        # Analyze directory depth patterns
        depth_distribution = {}
        for file_analysis in analysis_results.get('file_analyses', []):
            depth = len(file_analysis.current_location.split('/')) if file_analysis.current_location else 0
            depth_distribution[depth] = depth_distribution.get(depth, 0) + 1
        patterns['directory_depth_patterns'] = depth_distribution
        
        # Store patterns for system intelligence
        self.patterns_discovered.update(patterns)
        
        return patterns
    
    async def _calculate_performance_improvement(self, analysis_results: Dict[str, Any], 
                                               execution_results: Dict[str, Any]) -> float:
        """Calculate expected performance improvement from organization"""
        if not execution_results:
            return 0.0
        
        # Simple heuristic: organization improvements lead to performance gains
        files_moved = execution_results.get('files_moved', 0)
        total_files = analysis_results.get('total_files', 1)
        
        # Estimate performance improvement based on organization improvements
        improvement_ratio = files_moved / total_files
        performance_improvement = improvement_ratio * 0.15  # Up to 15% improvement
        
        return min(0.15, performance_improvement)
    
    async def _generate_organization_recommendations(self, analysis_results: Dict[str, Any]) -> List[str]:
        """Generate high-level organization recommendations"""
        recommendations = []
        
        total_files = analysis_results.get('total_files', 0)
        issues = analysis_results.get('organizational_issues', [])
        
        if total_files > 100:
            recommendations.append("Consider implementing atomic design structure for better component organization")
        
        if len(issues) > 10:
            recommendations.append("Multiple organizational issues detected - implement comprehensive reorganization")
        
        if analysis_results.get('asset_distribution', {}).get('scattered', False):
            recommendations.append("Consolidate assets into dedicated assets/ directory structure")
        
        return recommendations
    
    async def get_discovered_patterns(self) -> Dict[str, Any]:
        """Get patterns discovered by ULTRAORGANIZE"""
        return self.patterns_discovered.copy()
    
    # Additional helper methods for initialization and pattern analysis
    
    async def _load_organization_rules(self):
        """Load organization rules and patterns"""
        # Default organization rules for frontend optimization
        self.organization_rules = [
            OrganizationRule("*.py", "components/atoms", "component", 1, "Simple Python components"),
            OrganizationRule("*page*.py", "pages", "page", 2, "Page components"),
            OrganizationRule("*util*.py", "utils", "utility", 3, "Utility functions"),
            OrganizationRule("*service*.py", "services", "service", 4, "Service layer"),
            OrganizationRule("*config*.py", "config", "configuration", 5, "Configuration files"),
            OrganizationRule("test_*.py", "tests", "test", 6, "Test files"),
            OrganizationRule("*.css", "styles", "style", 7, "Style files"),
            OrganizationRule("*.md", "docs", "documentation", 8, "Documentation"),
        ]
    
    async def _initialize_dependency_analyzer(self):
        """Initialize dependency analysis capabilities"""
        self.dependency_graph = {}
    
    async def _load_pattern_models(self):
        """Load pattern recognition models"""
        self.patterns_discovered = {}
    
    async def _analyze_directory_pattern(self, file_path: str, directory_structure: Dict):
        """Analyze directory patterns for optimization"""
        # Implementation for directory pattern analysis
        pass
    
    async def _analyze_component_hierarchy(self, file_analyses: List[FileAnalysis]) -> Dict[str, Any]:
        """Analyze component hierarchy for atomic design optimization"""
        hierarchy = {
            'atoms': [],
            'molecules': [],
            'organisms': [],
            'templates': [],
            'pages': []
        }
        
        for analysis in file_analyses:
            if analysis.file_type == 'component':
                if analysis.complexity_score < 0.3:
                    hierarchy['atoms'].append(analysis.file_path)
                elif analysis.complexity_score < 0.7:
                    hierarchy['molecules'].append(analysis.file_path)
                else:
                    hierarchy['organisms'].append(analysis.file_path)
            elif analysis.file_type == 'page':
                hierarchy['pages'].append(analysis.file_path)
        
        return hierarchy
    
    async def _analyze_asset_distribution(self, file_analyses: List[FileAnalysis]) -> Dict[str, Any]:
        """Analyze asset distribution for optimization"""
        distribution = {
            'styles': [],
            'scripts': [],
            'images': [],
            'fonts': [],
            'scattered': False
        }
        
        asset_locations = set()
        for analysis in file_analyses:
            if analysis.file_type in ['style', 'script', 'image', 'font']:
                distribution[analysis.file_type + 's'].append(analysis.file_path)
                asset_locations.add(analysis.current_location)
        
        # Check if assets are scattered across multiple locations
        distribution['scattered'] = len(asset_locations) > 3
        
        return distribution
    
    async def _identify_organizational_issues(self, analysis: Dict[str, Any]) -> List[str]:
        """Identify organizational issues in current structure"""
        issues = []
        
        # Check for files in wrong locations
        misplaced_count = sum(1 for fa in analysis.get('file_analyses', []) 
                             if fa.current_location != fa.optimal_location)
        if misplaced_count > 0:
            issues.append(f"{misplaced_count} files in suboptimal locations")
        
        # Check for deep nesting
        deep_files = sum(1 for fa in analysis.get('file_analyses', []) 
                        if len(fa.current_location.split('/')) > 4)
        if deep_files > 0:
            issues.append(f"{deep_files} files with excessive nesting")
        
        return issues
    
    async def _find_optimization_opportunities(self, analysis: Dict[str, Any]) -> List[str]:
        """Find specific optimization opportunities"""
        opportunities = []
        
        # Component consolidation opportunities
        components = [fa for fa in analysis.get('file_analyses', []) if fa.file_type == 'component']
        if len(components) > 20:
            opportunities.append("Implement atomic design structure for component organization")
        
        # Asset optimization opportunities
        assets = [fa for fa in analysis.get('file_analyses', []) if fa.file_type in ['style', 'script', 'image']]
        if len(assets) > 10:
            opportunities.append("Implement asset optimization and bundling strategy")
        
        return opportunities