import re


    class MarkdownFormatter:
    @staticmethod
        def format_markdown(content: str) -> str:
        """Comprehensive markdown formatting"""
        lines = content.split("\n")
        formatted_lines = []
        
            for i, line in enumerate(lines):
            # Add blank lines around headings
                if re.match(r"^#+\s", line):
                    if i > 0 and formatted_lines[-1].strip():
                    formatted_lines.append("")
                    formatted_lines.append(line)
                        if i < len(lines) - 1 and lines[i + 1].strip():
                        formatted_lines.append("")
                        
                        # Ensure lists have blank lines
                            elif re.match(r"^[*+-]\s", line) or re.match(r"^\d+\.\s", line):
                                if i > 0 and not re.match(r"^[*+-\d]\s", formatted_lines[-1]):
                                formatted_lines.append("")
                                formatted_lines.append(line)
                                    if i < len(lines) - 1 and not re.match(r"^[*+-\d]\s", lines[i + 1]):
                                    formatted_lines.append("")
                                    
                                        else:
                                        formatted_lines.append(line)
                                        
                                        # Ensure single trailing newline
                                            while formatted_lines and formatted_lines[-1].strip() == "":
                                            formatted_lines.pop()
                                            formatted_lines.append("")
                                            
                                            return "\n".join(formatted_lines)
                                            