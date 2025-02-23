#!/usr/bin/env python3
# cSpell:ignore sutazai Sutaz levelname automac ctaches semgrep deepseek docstrings

import ast
import difflib
import json
import logging
import os
import re
import sys
import tokenize
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import spellchecker
from rich.console import Console
from rich.table import Table


class SpellCheckManager:
    def __init__(self, base_path: str = "/opt/sutazai_project/SutazAI"):
        """
        Initialize comprehensive spell-checking system

        Args:
            base_path (str): Base path of the project
        """
        self.base_path = base_path
        self.console = Console()
        self.spell_checker = spellchecker.SpellChecker()

        # Logging setup
        self.log_dir = os.path.join(base_path, "logs", "spell_check")
        os.makedirs(self.log_dir, exist_ok=True)

        self.log_file = os.path.join(
            self.log_dir, f"spell_check_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s: %(message)s",
            handlers=[
                logging.FileHandler(self.log_file),
                logging.StreamHandler(sys.stdout),
            ],
        )

        # Custom dictionary for project-specific terms
        self.custom_dictionary = self._load_custom_dictionary()

    def _load_custom_dictionary(self) -> set:
        """
        Load project-specific custom dictionary

        Returns:
            Set of project-specific valid words
        """
        custom_dict_path = os.path.join(
            self.base_path, "config", "custom_dictionary.json"
        )
        try:
            with open(custom_dict_path, "r") as f:
                return set(json.load(f))
        except FileNotFoundError:
            return {
                "sutazai",
                "automatic",
                "optimized",
                "ctaches",
                "networkx",
                "semgrep",
                "deepseek",
                "gpt4all",
            }

    def _is_valid_identifier(self, word: str) -> bool:
        """
        Check if a word is a valid Python identifier

        Args:
            word (str): Word to check

        Returns:
            Boolean indicating if word is a valid identifier
        """
        try:
            ast.parse(word)
            return True
        except SyntaxError:
            return False

    def check_file_spelling(self, file_path: str) -> Dict[str, List[Tuple[str, str]]]:
        """
        Perform comprehensive spell checking on a file

        Args:
            file_path (str): Path to the file to check

        Returns:
            Dictionary of spelling errors with suggested corrections
        """
        spelling_errors = {"comments": [], "docstrings": [], "string_literals": []}

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                file_content = f.read()

            # Parse the file as an AST
            tree = ast.parse(file_content)

            # Extract comments, docstrings, and string literals
            for node in ast.walk(tree):
                # Check docstrings
                if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Module)):
                    docstring = ast.get_docstring(node)
                    if docstring:
                        doc_errors = self._check_text_spelling(docstring)
                        if doc_errors:
                            spelling_errors["docstrings"].extend(
                                [(word, correction) for word, correction in doc_errors]
                            )

                # Check string literals
                if isinstance(node, ast.Str):
                    str_errors = self._check_text_spelling(node.s)
                    if str_errors:
                        spelling_errors["string_literals"].extend(
                            [(word, correction) for word, correction in str_errors]
                        )

            # Extract comments using tokenize
            with open(file_path, "rb") as f:
                tokens = list(tokenize.tokenize(f.readline))
                comments = [
                    token.string for token in tokens if token.type == tokenize.COMMENT
                ]

                for comment in comments:
                    comment_errors = self._check_text_spelling(comment)
                    if comment_errors:
                        spelling_errors["comments"].extend(
                            [(word, correction) for word, correction in comment_errors]
                        )

        except Exception as e:
            logging.error(f"Error checking spelling in {file_path}: {e}")

        return spelling_errors

    def _check_text_spelling(self, text: str) -> List[Tuple[str, str]]:
        """
        Check spelling of a given text

        Args:
            text (str): Text to check

        Returns:
            List of misspelled words with corrections
        """
        # Remove code-specific patterns
        text = re.sub(r"[{}()[\]:;]", " ", text)

        # Split into words, removing punctuation
        words = re.findall(r"\b\w+\b", text.lower())

        errors = []
        for word in words:
            # Skip if word is in custom dictionary or is a valid identifier
            if (
                word in self.custom_dictionary
                or self._is_valid_identifier(word)
                or len(word) <= 2
            ):
                continue

            # Check spelling
            if word not in self.spell_checker:
                # Get correction
                correction = self.spell_checker.correction(word)
                if correction != word:
                    errors.append((word, correction))

        return errors

    def auto_correct_file(self, file_path: str) -> bool:
        """
        Automatically correct spelling errors in a file

        Args:
            file_path (str): Path to the file to correct

        Returns:
            Boolean indicating if corrections were made
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Find spelling errors
            spelling_errors = self.check_file_spelling(file_path)

            # Apply corrections
            for error_type, errors in spelling_errors.items():
                for original, correction in errors:
                    # Use regex to replace whole words
                    content = re.sub(
                        r"\b{}\b".format(re.escape(original)),
                        correction,
                        content,
                        flags=re.IGNORECASE,
                    )

            # Write corrected content
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)

            # Log corrections
            if any(spelling_errors.values()):
                logging.info(f"Corrected spelling in {file_path}: {spelling_errors}")
                return True

            return False

        except Exception as e:
            logging.error(f"Error auto-correcting {file_path}: {e}")
            return False

    def scan_project(self) -> Dict[str, List[Tuple[str, str]]]:
        """
        Scan entire project for spelling errors

        Returns:
            Dictionary of spelling errors by file
        """
        project_spelling_errors = {}

        # Walk through project files
        for root, _, files in os.walk(self.base_path):
            for file in files:
                if file.endswith((".py", ".md", ".txt", ".rst")):
                    file_path = os.path.join(root, file)

                    # Check spelling
                    errors = self.check_file_spelling(file_path)

                    # Only add files with errors
                    if any(errors.values()):
                        project_spelling_errors[file_path] = errors

        return project_spelling_errors

    def auto_correct_project(self) -> int:
        """
        Automatically correct spelling errors across the entire project

        Returns:
            Number of files corrected
        """
        files_corrected = 0

        # Walk through project files
        for root, _, files in os.walk(self.base_path):
            for file in files:
                if file.endswith((".py", ".md", ".txt", ".rst")):
                    file_path = os.path.join(root, file)

                    # Auto-correct file
                    if self.auto_correct_file(file_path):
                        files_corrected += 1

        # Visualize results
        self._visualize_correction_results(files_corrected)

        return files_corrected

    def _visualize_correction_results(self, files_corrected: int):
        """
        Visualize spell-checking and correction results

        Args:
            files_corrected (int): Number of files corrected
        """
        self.console.rule("[bold blue]SutazAI Spell Check Results[/bold blue]")

        # Create results table
        results_table = Table(title="Spelling Correction Summary")
        results_table.add_column("Metric", style="cyan")
        results_table.add_column("Value", style="magenta")

        results_table.add_row("Files Corrected", str(files_corrected))

        self.console.print(results_table)


def main():
    spell_check_manager = SpellCheckManager()

    # Perform project-wide spell check and auto-correction
    files_corrected = spell_check_manager.auto_correct_project()

    print(f"Spell check complete. {files_corrected} files corrected.")


if __name__ == "__main__":
    main()
