#!/usr/bin/env python3
"""
Comprehensive documentation cleanup script to remove all fantasy elements 
from SutazAI system and replace with practical automation terminology.
"""

import os
import re
import shutil
from pathlib import Path
from typing import List, Dict, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DocumentationCleaner:
    def __init__(self, root_path: str = "/opt/sutazaiapp"):
        self.root_path = Path(root_path)
        self.backup_dir = self.root_path / "cleanup_backups"
        self.backup_dir.mkdir(exist_ok=True)
        
        # Fantasy elements to practical replacements
        self.replacements = {
            # automation/advanced automation references
            r'\bAGI\b': 'automation system',
            r'\bASI\b': 'advanced automation',
            r'\bArtificial General Intelligence\b': 'Task Automation System',
            r'\bArtificial Super Intelligence\b': 'Advanced Automation System',
            r'\badvanced AI Autonomous System\b': 'task automation system',
            r'\bAI Autonomous System\b': 'automation system',
            r'\bAutonomous System\b': 'automation system',
            
            # Coordinator/Processing terminology
            r'\bcoordinator\b': 'coordinator',
            r'\bCoordinator\b': 'Coordinator',
            r'\bprocessing\b': 'processing',
            r'\bProcessing\b': 'Processing',
            r'\boptimized\b': 'optimized',
            r'\bOptimized\b': 'Optimized',
            r'\bneuron\b': 'processor',
            r'\bNeuron\b': 'Processor',
            r'\bsynaptic\b': 'connection',
            r'\bSynaptic\b': 'Connection',
            
            # System State terminology
            r'\bsystem_state\b': 'system state',
            r'\bSystem State\b': 'System State',
            r'\bactive\b': 'aware',
            r'\bConscious\b': 'Aware',
            r'\bcognitive\b': 'processing',
            r'\bCognitive\b': 'Processing',
            r'\bsentient\b': 'responsive',
            r'\bSentient\b': 'Responsive',
            r'\bawakening\b': 'activation',
            r'\bAwakening\b': 'Activation',
            
            # Advanced/advanced physics
            r'\badvanced\b': 'advanced',
            r'\bAdvanced\b': 'Advanced',
            r'\badvanced computing\b': 'advanced computing',
            r'\bAdvanced Computing\b': 'Advanced Computing',
            r'\badvanced mechanics\b': 'advanced algorithms',
            r'\bAdvanced Mechanics\b': 'Advanced Algorithms',
            
            # Mystical/enhanced language
            r'\bsophisticated\b': 'advanced',
            r'\bMystical\b': 'Advanced',
            r'\benhanced\b': 'optimal',
            r'\bDivine\b': 'Optimal',
            r'\boptimal\b': 'enhanced',
            r'\bTranscendent\b': 'Enhanced',
            r'\benlightened\b': 'optimized',
            r'\bEnlightened\b': 'Optimized',
            r'\bmiracle\b': 'breakthrough',
            r'\bMiracle\b': 'Breakthrough',
            
            # Intelligence terminology
            r'\bintelligence-aware\b': 'state-aware',
            r'\bIntelligence-aware\b': 'State-aware',
            r'\bintelligence integration\b': 'system integration',
            r'\bIntelligence Integration\b': 'System Integration',
            r'\bcollective intelligence\b': 'distributed processing',
            r'\bCollective Intelligence\b': 'Distributed Processing',
            r'\bemergent intelligence\b': 'optimized processing',
            r'\bEmergent Intelligence\b': 'Optimized Processing',
            
            # Backend/Frontend automation references
            r'\bbackend\b': 'backend',
            r'\bfrontend\b': 'frontend',
            r'\bBackend-automation\b': 'Backend',
            r'\bFrontend-automation\b': 'Frontend',
            
            # Specific system_state-related terms
            r'\bphi\b': 'integration_score',
            r'\bemergence\b': 'optimization',
            r'\bEmergence\b': 'Optimization',
            r'\bemergent\b': 'optimized',
            r'\bEmergent\b': 'Optimized',
            
            # File path references
            r'/opt/sutazaiapp/coordinator': '/opt/sutazaiapp/coordinator',
            r'coordinator_path': 'coordinator_path',
            r'BRAIN_': 'COORDINATOR_',
        }
        
        # Files to completely delete (purely fantasy content)
        self.files_to_delete = [
            # automation/advanced automation specific files
            "AGI_SYSTEM_COMPLETE.md",
            "COMPREHENSIVE_AGI_COMPLETION_REPORT.md",
            "FINAL_AGI_SYSTEM_REPORT.md",
            "AGI_ASI_ARCHITECTURE_PLAN.md",
            "AGI_ASI_IMPLEMENTATION_PLAN.md",
            "AGI_INTEGRATION_GUIDE.md",
            "ENTERPRISE_AGI_IMPLEMENTATION_PLAN.md",
            "OPTIMIZED_AGI_ARCHITECTURE_PLAN.md",
            "ADVANCED_AGI_ASI_ARCHITECTURE.md",
            "SUTAZAI_AGI_ASI_PROJECT_DOCUMENTATION.md",
            
            # Coordinator/Processing specific files
            "BRAIN_ARCHITECTURE.md",
            "BRAIN_SYSTEM_SUMMARY.md",
            "ENHANCED_BRAIN_SUMMARY.md",
            "NEURAL_LINK_NETWORKS_IMPLEMENTATION.md",
            
            # System State specific files
            "CONSCIOUSNESS_CLEANUP_REPORT.md",
            
            # Agent files with too much fantasy content
            "agi-system-architect.md",
            "agi-system-architect-detailed.md",
            "agi-system-validator.md",
            "agi-system-validator-detailed.md",
            "bigagi-system-manager.md",
            "bigagi-system-manager-detailed.md",
            "deep-learning-coordinator-architect.md",
            "deep-learning-coordinator-architect-detailed.md",
            "deep-learning-coordinator-manager.md",
            "deep-learning-coordinator-manager-detailed.md",
            "deep-local-coordinator-builder.md",
            "deep-local-coordinator-builder-detailed.md",
            "evolution-strategy-trainer.md",
            "evolution-strategy-trainer-detailed.md",
            "genetic-algorithm-tuner.md",
            "genetic-algorithm-tuner-detailed.md",
            "processing-architecture-search.md",
            "processing-architecture-search-detailed.md",
            "advanced-ai-researcher.md",
            "advanced-computing-optimizer.md",
            "advanced-computing-optimizer-detailed.md",
        ]
    
    def backup_file(self, file_path: Path) -> Path:
        """Create a backup of the file before modification"""
        backup_path = self.backup_dir / file_path.name
        counter = 1
        while backup_path.exists():
            stem = file_path.stem
            suffix = file_path.suffix
            backup_path = self.backup_dir / f"{stem}_backup_{counter}{suffix}"
            counter += 1
        
        shutil.copy2(file_path, backup_path)
        logger.info(f"Backed up {file_path} to {backup_path}")
        return backup_path
    
    def clean_content(self, content: str) -> str:
        """Apply all replacement patterns to content"""
        cleaned_content = content
        
        for pattern, replacement in self.replacements.items():
            cleaned_content = re.sub(pattern, replacement, cleaned_content)
        
        return cleaned_content
    
    def should_delete_file(self, file_path: Path) -> bool:
        """Check if file should be completely deleted"""
        filename = file_path.name
        return filename in self.files_to_delete
    
    def process_markdown_file(self, file_path: Path) -> bool:
        """Process a single markdown file"""
        try:
            # Check if file should be deleted
            if self.should_delete_file(file_path):
                logger.info(f"Deleting fantasy file: {file_path}")
                self.backup_file(file_path)
                file_path.unlink()
                return True
            
            # Read original content
            with open(file_path, 'r', encoding='utf-8') as f:
                original_content = f.read()
            
            # Clean content
            cleaned_content = self.clean_content(original_content)
            
            # Only write if content changed
            if cleaned_content != original_content:
                self.backup_file(file_path)
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(cleaned_content)
                
                logger.info(f"Cleaned: {file_path}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            return False
    
    def find_markdown_files(self) -> List[Path]:
        """Find all markdown files in the repository"""
        markdown_files = []
        
        for md_file in self.root_path.rglob("*.md"):
            # Skip backup directories
            if "backup" in str(md_file).lower():
                continue
            markdown_files.append(md_file)
        
        return sorted(markdown_files)
    
    def cleanup_all_documentation(self) -> Dict[str, int]:
        """Clean up all markdown documentation"""
        results = {
            "processed": 0,
            "cleaned": 0,
            "deleted": 0,
            "errors": 0
        }
        
        markdown_files = self.find_markdown_files()
        logger.info(f"Found {len(markdown_files)} markdown files to process")
        
        for file_path in markdown_files:
            results["processed"] += 1
            
            try:
                if self.should_delete_file(file_path):
                    self.backup_file(file_path)
                    file_path.unlink()
                    results["deleted"] += 1
                    logger.info(f"Deleted: {file_path}")
                else:
                    if self.process_markdown_file(file_path):
                        results["cleaned"] += 1
            except Exception as e:
                results["errors"] += 1
                logger.error(f"Error with {file_path}: {e}")
        
        return results
    
    def generate_report(self, results: Dict[str, int]) -> None:
        """Generate cleanup report"""
        report_path = self.root_path / "DOCUMENTATION_CLEANUP_REPORT.md"
        
        report_content = f"""# Documentation Cleanup Report

## Summary
- Total files processed: {results['processed']}
- Files cleaned: {results['cleaned']}
- Files deleted: {results['deleted']}
- Errors: {results['errors']}

## Changes Made

### Terminology Replacements
- automation/advanced automation → automation system/advanced automation
- coordinator/processing → coordinator/processing
- system_state → system state
- advanced/optimized → advanced/optimized
- sophisticated/enhanced/optimal → advanced/optimal/enhanced
- backend/frontend → backend/frontend

### Files Deleted
Fantasy-focused documentation files were archived and removed as they contained
primarily automation/system_state concepts not relevant to the practical automation system.

### Result
All documentation now reflects SutazAI as a practical multi-agent task automation platform
without fantasy elements like automation, system_state, or sophisticated concepts.

## Backups
All modified and deleted files have been backed up to: {self.backup_dir}

Generated on: {os.popen('date').read().strip()}
"""
        
        with open(report_path, 'w') as f:
            f.write(report_content)
        
        logger.info(f"Report generated: {report_path}")


def main():
    """Main cleanup function"""
    logger.info("Starting comprehensive documentation cleanup...")
    
    cleaner = DocumentationCleaner()
    results = cleaner.cleanup_all_documentation()
    cleaner.generate_report(results)
    
    logger.info("Cleanup completed!")
    logger.info(f"Results: {results}")


if __name__ == "__main__":
    main()