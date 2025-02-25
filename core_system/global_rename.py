import logging
import os
import re
from typing import List


class SutazAiRenamer:
    def __init__(self, old_name="Quantum", new_name="SutazAi"):
        self.old_name = old_name
        self.new_name = new_name
        self.logger = self._setup_logger()

    def _setup_logger(self):
        """Configure logging for renaming process"""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - SutazAi Renamer - %(levelname)s: %(message)s",
            handlers=[
                logging.FileHandler("sutazai_rename.log"),
                logging.StreamHandler(),
            ],
        )
        return logging.getLogger(__name__)

    def global_rename(self, root_dir="."):
        """
        Comprehensive renaming utility for entire codebase

        Args:
            root_dir (str): Root directory to start renaming. Defaults to current directory.
        """
        try:
            renamed_files = 0
            renamed_dirs = 0

            # Rename directories first
            for root, dirs, files in os.walk(root_dir, topdown=False):
                for dir_name in dirs:
                    if self.old_name.lower() in dir_name.lower():
                        old_path = os.path.join(root, dir_name)
                        new_path = os.path.join(
                            root,
                            dir_name.replace(
                                self.old_name.lower(), self.new_name.lower()
                            ),
                        )
                        os.rename(old_path, new_path)
                        renamed_dirs += 1
                        self.logger.info(
                            f"Renamed directory: {old_path} -> {new_path}"
                        )

            # Rename and modify files
            for root, _, files in os.walk(root_dir):
                for file in files:
                    file_path = os.path.join(root, file)

                    # Rename files containing old name
                    if self.old_name.lower() in file.lower():
                        new_file_name = file.replace(
                            self.old_name.lower(), self.new_name.lower()
                        )
                        new_file_path = os.path.join(root, new_file_name)
                        os.rename(file_path, new_file_path)
                        file_path = new_file_path
                        renamed_files += 1
                        self.logger.info(
                            f"Renamed file: {file} -> {new_file_name}"
                        )

                    # Modify file contents
                    if file.endswith(
                        (".py", ".md", ".txt", ".json", ".yaml", ".yml")
                    ):
                        self._rename_in_file(file_path)

            self.logger.info(
                f"Renaming complete. Renamed {renamed_dirs} directories and {renamed_files} files."
            )
        except Exception as e:
            self.logger.error(f"Renaming process failed: {e}")

    def _rename_in_file(self, filepath):
        """Rename instances within file contents"""
        try:
            with open(filepath, "r") as file:
                content = file.read()

            # Case-insensitive replacement
            content = re.sub(
                r"\b{}\b".format(self.old_name),
                self.new_name,
                content,
                flags=re.IGNORECASE,
            )

            with open(filepath, "w") as file:
                file.write(content)
        except Exception as e:
            self.logger.error(f"Error processing {filepath}: {e}")

    def rename_entities(self, entities: List[str]) -> List[str]:
        """
        Rename entities from 'Quantum' to 'SutazAi'.

        Args:
            entities (List[str]): List of entity names to rename.

        Returns:
            List[str]: Renamed entities.
        """
        renamed_entities = [
            entity.replace(self.old_name, self.new_name) for entity in entities
        ]
        return renamed_entities


def rename_files_and_folders(root_dir, old_name, new_name):
    """Rename files and folders globally."""
    try:
        for dirpath, dirnames, filenames in os.walk(root_dir):
            for dirname in dirnames:
                if old_name in dirname:
                    new_dirname = dirname.replace(old_name, new_name)
                    os.rename(
                        os.path.join(dirpath, dirname),
                        os.path.join(dirpath, new_dirname),
                    )
            for filename in filenames:
                if old_name in filename:
                    new_filename = filename.replace(old_name, new_name)
                    os.rename(
                        os.path.join(dirpath, filename),
                        os.path.join(dirpath, new_filename),
                    )
    except PermissionError as e:
        logger.error(f"Permission denied: {e}")
        raise


def main():
    renamer = SutazAiRenamer()
    renamer.global_rename()


def rename_file(old_path, new_path):
    try:
        os.rename(old_path, new_path)
    except Exception as e:
        logging.error(
            f"Failed to rename file from {old_path} to {new_path}: {e}"
        )
        raise


if __name__ == "__main__":
    main()
