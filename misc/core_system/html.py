import logging
from typing import Dict, List, Optional, Tuple

from safety.formatter import FormatterAPI
from safety.formatters.json import build_json_report
from safety.output_utils import parse_html

LOG = logging.getLogger(__name__)


class HTMLReport(FormatterAPI):
    """
    HTML report formatter for when the output is input for something else.
    """

    def render_vulnerabilities(
        self,
        announcements: List[Dict],
        vulnerabilities: List[Dict],
        remediations: Dict,
        full: bool,
        packages: List[Dict],
        fixes: Tuple = (),
    ) -> Optional[str]:
        """
        Renders vulnerabilities in HTML format.

        Args:
            announcements (List[Dict]): List of announcements.
            vulnerabilities (List[Dict]): List of vulnerabilities.
            remediations (Dict): Remediation data.
            full (bool): Flag indicating full output.
            packages (List[Dict]): List of packages.
            fixes (Tuple, optional): Tuple of fixes.

        Returns:
            str: Rendered HTML vulnerabilities report.
        """
        LOG.debug(
            f"HTML Output, Rendering {len(vulnerabilities)} vulnerabilities, {len(remediations)} package "
            f"remediations with full_report: {full}"
        )
        report = build_json_report(
            announcements, vulnerabilities, remediations, packages
        )

        return parse_html(kwargs={"json_data": report})

    def render_licenses(
        self, announcements: List[Dict], licenses: List[Dict]
    ) -> None:
        """
        Renders licenses in HTML format.

        Args:
            announcements (List[Dict]): List of announcements.
            licenses (List[Dict]): List of licenses.
        """

    def render_announcements(self, announcements: List[Dict]) -> None:
        """
        Renders announcements in HTML format.

        Args:
            announcements (List[Dict]): List of announcements.
        """
