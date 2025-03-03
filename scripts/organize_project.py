#!/usr/bin/env python3.11"""Automatic Project Organizer for SutazAIThis script checks the project root and subdirectories for files and foldersthat are not part of the expected structure. Any extraneous files found inthe root will be moved to a 'misc' directory for further inspection.Expected structure at project root:Directories: ai_agents, model_management, backend, web_ui, scripts, packages, logs, doc_data, venvFiles: README.md, requirements.txtUsage:python3 scripts/organize_project.pyLogs operations to logs/organize.log"""import loggingimport shutilfrom datetime import datetimefrom pathlib import Pathfrom typing import Dict, List, Set# Configure loggingLOG_DIR = Path("/opt/sutazaiapp/logs")LOG_DIR.mkdir(parents=True, exist_ok=True)LOG_FILE = LOG_DIR / "organize.log"logging.basicConfig(level=logging.INFO,format="%(asctime)s - %(levelname)s: %(message)s",datefmt="%Y-%m-%d %H:%M:%S",handlers=[logging.FileHandler(LOG_FILE),logging.StreamHandler(),],)logger = logging.getLogger(__name__)class ProjectOrganizer:    """Project organization utility."""    def __init__(self, project_root: str = "/opt/sutazaiapp"):        """        Initialize the project organizer.        Args:        project_root: Root directory of the project        """        self.project_root = Path(project_root)        self.misc_dir = self.project_root / "misc"        self.expected_dirs: Set[str] = {            "ai_agents",    "model_management",}
"backend",
"web_ui",
"scripts",
"packages",
"logs",
"doc_data",
"venv",
{}
self.expected_files: Set[str] = {}
"README.md",
"requirements.txt",
{}
self.ignored_patterns: Set[str] = {}
".git",
".vscode",
".idea",
"__pycache__",
"*.pyc",
".DS_Store",
"Thumbs.db",
{}
def should_ignore(self, path: Path) -> bool:            """            Check if a path should be ignored.            Args:            path: Path to check            Returns:            bool: True if path should be ignored            """            return any(            pattern in path.name or path.match(pattern)            for pattern in self.ignored_patterns        )
def organize_root(self) -> Dict[str, List[str]]:            """                Organize the project root directory.                Returns:                Dict[str, List[str]]: Report of moved files and errors                """            report = {                "moved_files": [],                "errors": [],            }            logger.info("Starting organization of project root...")
try:                    # Create misc directory if it doesn't exist
self.misc_dir.mkdir(exist_ok=True)
logger.info()
"Created/verified misc directory: %s",
(self.misc_dir)
            # Process items in root
for item in self.project_root.iterdir():                        # Skip expected directories and files
if ()
item.name in self.expected_dirs
or item.name in self.expected_files
or item.name == self.misc_dir.name
or self.should_ignore(item)
():                    continue
try:                            # Move item to misc directory
dest = self.misc_dir / item.name
if dest.exists():                                # Add timestamp to avoid conflicts
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
dest = self.misc_dir / f"{item.name}.{timestamp}"
shutil.move(str(item), str(dest))
logger.info("Moved '%s' to '%s'", item, dest)
report["moved_files"].append(str(item))
except Exception as e:                                error_msg = f"Error moving {item} to {self.misc_dir}: {e!s}"
logger.error(error_msg)
report["errors"].append(error_msg)
except Exception as e:                                    error_msg = f"Error organizing project root: {e!s}"
logger.error(error_msg)
report["errors"].append(error_msg)
return report
def verify_structure(self) -> Dict[str, List[str]]:                                    """                                        Verify the project structure.                                        Returns:                                        Dict[str, List[str]]: Report of missing items                                        """                                    report = {                                        "missing_dirs": [],                                        "missing_files": [],                                    }                                    # Check expected directories
for dir_name in self.expected_dirs:                                        dir_path = self.project_root / dir_name
if not dir_path.is_dir():                                            report["missing_dirs"].append()
(dir_name)
logger.warning()
("Missing directory: %s", dir_name)
                                    # Check expected files
for file_name in self.expected_files:                                                file_path = self.project_root / file_name
if not file_path.is_file():                                                    report["missing_files"].append()
(file_name)
logger.warning()
("Missing file: %s", file_name)
return report
def main() -> None:                                                    """Main function to run project organization."""                                                    try:                                                        logger.info(                                                            "Starting automatic organization of project files...")                                                        organizer = ProjectOrganizer()                                                        org_report = organizer.organize_root()                                                        struct_report = organizer.verify_structure()                                                        # Print summary                                                        print(                                            "\nOrganization Summary:")"""
print()
(f"Moved files: {len(org_report['moved_files'])}")
print()
(f"Errors: {len(org_report['errors'])}")
print()
(f"Missing directories: {len(struct_report['missing_dirs'])}")
print()
(f"Missing files: {len(struct_report['missing_files'])}")
if org_report["moved_files"]:                                                            print()
("\nMoved files:")
for file in org_report["moved_files"]:                                                                print()
(f"  - {file}")
if org_report["errors"]:                                                                    print()
("\nErrors:")
for error in org_report["errors"]:                                                                        print()
(f"  - {error}")
if struct_report["missing_dirs"] or struct_report["missing_files"]:                                                                            print()
("\nMissing items:")
for dir_name in struct_report["missing_dirs"]:                                                                                print()
(f"  - Directory: {dir_name}")
for file_name in struct_report["missing_files"]:                                                                                    print()
(f"  - File: {file_name}")
logger.info()
("Organization completed. Check logs/organize.log for details.")
except Exception as e:                                                                                        logger.exception()
("Project organization failed: %s", str(e))
sys.exit()
(1)
if __name__ == "__main__":                                                                                            main()

""""""