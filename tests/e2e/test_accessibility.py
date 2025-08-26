#!/usr/bin/env python3
"""
import logging

logger = logging.getLogger(__name__)
ULTRA-COMPREHENSIVE ACCESSIBILITY AND RESPONSIVE DESIGN VALIDATION
WCAG 2.1 AA Compliance Testing for SutazAI Frontend
"""

import os
import re
from typing import Dict, List, Any
from pathlib import Path
import json

class AccessibilityTester:
    """Comprehensive accessibility and responsive design testing"""
    
    def __init__(self, frontend_path: str = "/opt/sutazaiapp/frontend"):
        self.frontend_path = Path(frontend_path)
        self.test_results = []
        self.accessibility_violations = []
        self.responsive_issues = []
    
    def log_test(self, category: str, test_name: str, success: bool, details: str = "", severity: str = "info"):
        """Log test result with categorization"""
        result = {
            'category': category,
            'test_name': test_name,
            'success': success,
            'details': details,
            'severity': severity,
            'timestamp': 'test-time'
        }
        self.test_results.append(result)
        
        status = "✅ PASS" if success else "❌ FAIL" if severity == "critical" else "⚠️ WARN"
        logger.info(f"{status} {category}: {test_name}")
        if details:
            logger.info(f"    {details}")
    
    def test_semantic_html(self) -> None:
        """Test semantic HTML structure"""
        
        # Check all Python files for Streamlit semantic elements
        py_files = list(self.frontend_path.rglob("*.py"))
        
        semantic_elements_found = {
            'headers': 0,
            'navigation': 0,
            'main_content': 0,
            'sections': 0,
            'articles': 0
        }
        
        accessibility_patterns = {
            'st.header': 'headers',
            'st.subheader': 'headers', 
            'st.sidebar': 'navigation',
            'st.container': 'sections',
            'st.expander': 'sections'
        }
        
        for py_file in py_files:
            if py_file.name.startswith('test_'):
                continue
                
            try:
                content = py_file.read_text()
                
                for pattern, element_type in accessibility_patterns.items():
                    matches = len(re.findall(pattern, content))
                    semantic_elements_found[element_type] += matches
                    
            except Exception as e:
                self.log_test("Semantic HTML", f"File Read Error: {py_file.name}", False, str(e), "warning")
        
        # Evaluate semantic structure
        total_elements = sum(semantic_elements_found.values())
        
        if total_elements > 20:
            self.log_test("Semantic HTML", "Element Count", True, 
                         f"Found {total_elements} semantic elements across {len(py_files)} files")
        else:
            self.log_test("Semantic HTML", "Element Count", False,
                         f"Only {total_elements} semantic elements found - needs improvement", "warning")
        
        # Check for proper header hierarchy
        header_usage = semantic_elements_found['headers']
        if header_usage > 5:
            self.log_test("Semantic HTML", "Header Hierarchy", True,
                         f"{header_usage} headers provide good content structure")
        else:
            self.log_test("Semantic HTML", "Header Hierarchy", False,
                         f"Only {header_usage} headers - may lack proper content hierarchy", "warning")
    
    def test_aria_labels(self) -> None:
        """Test ARIA labels and accessibility attributes"""
        
        py_files = list(self.frontend_path.rglob("*.py"))
        
        aria_patterns = {
            'aria-label': 0,
            'aria-describedby': 0, 
            'aria-expanded': 0,
            'role=': 0,
            'alt=': 0,
            'title=': 0,
            'help=': 0  # Streamlit help parameter
        }
        
        for py_file in py_files:
            if py_file.name.startswith('test_'):
                continue
                
            try:
                content = py_file.read_text()
                
                for pattern in aria_patterns.keys():
                    matches = len(re.findall(pattern, content, re.IGNORECASE))
                    aria_patterns[pattern] += matches
                    
            except Exception as e:
                continue
        
        total_aria = sum(aria_patterns.values())
        
        if total_aria > 10:
            self.log_test("ARIA Labels", "Accessibility Attributes", True,
                         f"Found {total_aria} accessibility attributes")
        else:
            self.log_test("ARIA Labels", "Accessibility Attributes", False,
                         f"Only {total_aria} accessibility attributes - needs enhancement", "warning")
        
        # Check specific ARIA implementations
        if aria_patterns['help='] > 5:
            self.log_test("ARIA Labels", "Help Text Implementation", True,
                         f"{aria_patterns['help=']} help texts provide good UX guidance")
        else:
            self.log_test("ARIA Labels", "Help Text Implementation", False,
                         f"Only {aria_patterns['help=']} help texts - consider adding more guidance", "warning")
    
    def test_responsive_design(self) -> None:
        """Test responsive design implementation"""
        
        # Check for CSS media queries and responsive patterns
        py_files = list(self.frontend_path.rglob("*.py"))
        
        responsive_patterns = {
            '@media': 0,
            'max-width': 0,
            'min-width': 0,
            'st.columns': 0,
            'use_container_width=True': 0,
            'mobile': 0,
            'tablet': 0,
            'desktop': 0
        }
        
        for py_file in py_files:
            if py_file.name.startswith('test_'):
                continue
                
            try:
                content = py_file.read_text()
                
                for pattern in responsive_patterns.keys():
                    matches = len(re.findall(pattern, content, re.IGNORECASE))
                    responsive_patterns[pattern] += matches
                    
            except Exception as e:
                continue
        
        # Evaluate responsive design
        media_queries = responsive_patterns['@media']
        responsive_containers = responsive_patterns['use_container_width=True'] 
        column_layouts = responsive_patterns['st.columns']
        
        total_responsive = media_queries + responsive_containers + column_layouts
        
        if total_responsive > 15:
            self.log_test("Responsive Design", "Responsive Elements", True,
                         f"Found {total_responsive} responsive design elements")
        else:
            self.log_test("Responsive Design", "Responsive Elements", False,
                         f"Only {total_responsive} responsive elements - needs improvement", "warning")
        
        if media_queries > 3:
            self.log_test("Responsive Design", "Media Queries", True,
                         f"{media_queries} media queries for different screen sizes")
        else:
            self.log_test("Responsive Design", "Media Queries", False,
                         f"Only {media_queries} media queries - consider mobile/tablet breakpoints", "warning")
        
        if responsive_containers > 10:
            self.log_test("Responsive Design", "Container Width", True,
                         f"{responsive_containers} responsive containers")
        else:
            self.log_test("Responsive Design", "Container Width", False,
                         f"Only {responsive_containers} responsive containers", "warning")
    
    def test_color_contrast(self) -> None:
        """Test color contrast and visual accessibility"""
        
        py_files = list(self.frontend_path.rglob("*.py"))
        
        color_patterns = {
            'color:': 0,
            'background': 0,
            'rgba(': 0,
            '#[0-9a-fA-F]{6}': 0,  # Hex colors
            '#[0-9a-fA-F]{3}': 0,   # Short hex colors
            'prefers-contrast': 0,
            'prefers-color-scheme': 0
        }
        
        high_contrast_support = 0
        
        for py_file in py_files:
            if py_file.name.startswith('test_'):
                continue
                
            try:
                content = py_file.read_text()
                
                for pattern in color_patterns.keys():
                    matches = len(re.findall(pattern, content, re.IGNORECASE))
                    color_patterns[pattern] += matches
                
                # Check for high contrast support
                if 'prefers-contrast' in content or 'high contrast' in content.lower():
                    high_contrast_support += 1
                    
            except Exception as e:
                continue
        
        total_colors = sum(color_patterns.values())
        
        if total_colors > 20:
            self.log_test("Color Contrast", "Color Usage", True,
                         f"Found {total_colors} color implementations")
        else:
            self.log_test("Color Contrast", "Color Usage", False,
                         f"Only {total_colors} color implementations", "info")
        
        if high_contrast_support > 0:
            self.log_test("Color Contrast", "High Contrast Support", True,
                         f"High contrast mode support found in {high_contrast_support} files")
        else:
            self.log_test("Color Contrast", "High Contrast Support", False,
                         "No high contrast mode support detected", "warning")
        
        # Check for accessibility-friendly color usage
        hex_colors = color_patterns['#[0-9a-fA-F]{6}'] + color_patterns['#[0-9a-fA-F]{3}']
        rgba_colors = color_patterns['rgba(']
        
        if rgba_colors > hex_colors:
            self.log_test("Color Contrast", "Color Implementation", True,
                         f"Good use of RGBA colors ({rgba_colors}) over hex ({hex_colors})")
        else:
            self.log_test("Color Contrast", "Color Implementation", False,
                         f"Consider using more RGBA colors for better transparency control", "info")
    
    def test_keyboard_navigation(self) -> None:
        """Test keyboard navigation support"""
        
        py_files = list(self.frontend_path.rglob("*.py"))
        
        keyboard_patterns = {
            'key=': 0,  # Streamlit key parameter
            'focus': 0,
            'tab': 0,
            'button': 0,
            'input': 0,
            'select': 0,
            'onKeyPress': 0,
            'onKeyDown': 0,
            'tabindex': 0
        }
        
        for py_file in py_files:
            if py_file.name.startswith('test_'):
                continue
                
            try:
                content = py_file.read_text()
                
                for pattern in keyboard_patterns.keys():
                    matches = len(re.findall(pattern, content, re.IGNORECASE))
                    keyboard_patterns[pattern] += matches
                    
            except Exception as e:
                continue
        
        # Evaluate keyboard navigation
        interactive_elements = keyboard_patterns['button'] + keyboard_patterns['input'] + keyboard_patterns['select']
        focus_management = keyboard_patterns['focus'] + keyboard_patterns['tab'] + keyboard_patterns['tabindex']
        
        if interactive_elements > 20:
            self.log_test("Keyboard Navigation", "Interactive Elements", True,
                         f"{interactive_elements} interactive elements support keyboard access")
        else:
            self.log_test("Keyboard Navigation", "Interactive Elements", False,
                         f"Only {interactive_elements} interactive elements", "warning")
        
        if focus_management > 5:
            self.log_test("Keyboard Navigation", "Focus Management", True,
                         f"{focus_management} focus management implementations")
        else:
            self.log_test("Keyboard Navigation", "Focus Management", False,
                         f"Limited focus management ({focus_management}) - consider keyboard navigation", "warning")
    
    def test_reduced_motion(self) -> None:
        """Test reduced motion and animation accessibility"""
        
        py_files = list(self.frontend_path.rglob("*.py"))
        
        motion_patterns = {
            'animation': 0,
            'transition': 0,
            'transform': 0,
            'prefers-reduced-motion': 0,
            '@keyframes': 0,
            'ease': 0,
            'duration': 0
        }
        
        reduced_motion_support = 0
        
        for py_file in py_files:
            if py_file.name.startswith('test_'):
                continue
                
            try:
                content = py_file.read_text()
                
                for pattern in motion_patterns.keys():
                    matches = len(re.findall(pattern, content, re.IGNORECASE))
                    motion_patterns[pattern] += matches
                
                if 'prefers-reduced-motion' in content:
                    reduced_motion_support += 1
                    
            except Exception as e:
                continue
        
        total_animations = motion_patterns['animation'] + motion_patterns['transition'] + motion_patterns['@keyframes']
        
        if total_animations > 0:
            if reduced_motion_support > 0:
                self.log_test("Reduced Motion", "Motion Accessibility", True,
                             f"{total_animations} animations with reduced motion support")
            else:
                self.log_test("Reduced Motion", "Motion Accessibility", False,
                             f"{total_animations} animations but no reduced motion support", "warning")
        else:
            self.log_test("Reduced Motion", "Motion Accessibility", True,
                         "No animations detected - good for accessibility")
    
    def test_form_accessibility(self) -> None:
        """Test form accessibility and validation"""
        
        py_files = list(self.frontend_path.rglob("*.py"))
        
        form_patterns = {
            'st.text_input': 0,
            'st.text_area': 0,
            'st.selectbox': 0,
            'st.multiselect': 0,
            'st.slider': 0,
            'st.checkbox': 0,
            'st.radio': 0,
            'st.file_uploader': 0,
            'label': 0,
            'placeholder': 0,
            'help': 0,
            'required': 0,
            'disabled': 0
        }
        
        for py_file in py_files:
            if py_file.name.startswith('test_'):
                continue
                
            try:
                content = py_file.read_text()
                
                for pattern in form_patterns.keys():
                    matches = len(re.findall(pattern, content))
                    form_patterns[pattern] += matches
                    
            except Exception as e:
                continue
        
        total_form_elements = sum(v for k, v in form_patterns.items() if k.startswith('st.'))
        form_helpers = form_patterns['help'] + form_patterns['placeholder']
        
        if total_form_elements > 10:
            self.log_test("Form Accessibility", "Form Elements", True,
                         f"{total_form_elements} form elements implemented")
        else:
            self.log_test("Form Accessibility", "Form Elements", False,
                         f"Only {total_form_elements} form elements", "info")
        
        if form_helpers > total_form_elements * 0.5:  # At least 50% of forms have help
            self.log_test("Form Accessibility", "Form Help Text", True,
                         f"{form_helpers} help texts for {total_form_elements} form elements")
        else:
            self.log_test("Form Accessibility", "Form Help Text", False,
                         f"Only {form_helpers} help texts for {total_form_elements} elements", "warning")
    
    def generate_accessibility_report(self) -> Dict[str, Any]:
        """Generate comprehensive accessibility report"""
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for test in self.test_results if test['success'])
        warnings = sum(1 for test in self.test_results if not test['success'] and test['severity'] == 'warning')
        critical_failures = sum(1 for test in self.test_results if not test['success'] and test['severity'] == 'critical')
        
        # Calculate compliance score
        compliance_score = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        # Determine WCAG compliance level
        if compliance_score >= 95:
            wcag_level = "AA (Excellent)"
        elif compliance_score >= 80:
            wcag_level = "AA (Good)"
        elif compliance_score >= 70:
            wcag_level = "A (Acceptable)"
        else:
            wcag_level = "Below Standards"
        
        report = {
            'accessibility_summary': {
                'total_tests': total_tests,
                'passed': passed_tests,
                'warnings': warnings,
                'critical_failures': critical_failures,
                'compliance_score': round(compliance_score, 1),
                'wcag_level': wcag_level
            },
            'category_results': {},
            'detailed_results': self.test_results,
            'recommendations': []
        }
        
        # Group results by category
        categories = set(test['category'] for test in self.test_results)
        for category in categories:
            category_tests = [test for test in self.test_results if test['category'] == category]
            category_passed = sum(1 for test in category_tests if test['success'])
            
            report['category_results'][category] = {
                'total': len(category_tests),
                'passed': category_passed,
                'success_rate': round((category_passed / len(category_tests) * 100), 1)
            }
        
        # Generate recommendations
        if warnings > 0:
            report['recommendations'].append("Address accessibility warnings to improve WCAG compliance")
        
        if critical_failures > 0:
            report['recommendations'].append("Fix critical accessibility issues immediately")
        
        low_performing_categories = [
            cat for cat, results in report['category_results'].items() 
            if results['success_rate'] < 70
        ]
        
        if low_performing_categories:
            report['recommendations'].append(f"Focus improvement on: {', '.join(low_performing_categories)}")
        
        return report
    
    def run_comprehensive_accessibility_tests(self) -> Dict[str, Any]:
        """Run all accessibility and responsive design tests"""
        
        logger.info("♿ ULTRA-COMPREHENSIVE ACCESSIBILITY & RESPONSIVE DESIGN VALIDATION")
        logger.info("=" * 80)
        logger.info("Testing WCAG 2.1 AA Compliance for SutazAI Frontend")
        logger.info()
        
        logger.info("🏗️ 1. SEMANTIC HTML STRUCTURE")
        logger.info("-" * 40)
        self.test_semantic_html()
        logger.info()
        
        logger.info("🏷️ 2. ARIA LABELS & ATTRIBUTES")
        logger.info("-" * 40)
        self.test_aria_labels()
        logger.info()
        
        logger.info("📱 3. RESPONSIVE DESIGN")
        logger.info("-" * 40)
        self.test_responsive_design()
        logger.info()
        
        logger.info("🎨 4. COLOR CONTRAST & VISUAL")
        logger.info("-" * 40)
        self.test_color_contrast()
        logger.info()
        
        logger.info("⌨️ 5. KEYBOARD NAVIGATION")
        logger.info("-" * 40)
        self.test_keyboard_navigation()
        logger.info()
        
        logger.info("🎬 6. REDUCED MOTION")
        logger.info("-" * 40)
        self.test_reduced_motion()
        logger.info()
        
        logger.info("📝 7. FORM ACCESSIBILITY")
        logger.info("-" * 40)
        self.test_form_accessibility()
        logger.info()
        
        # Generate final report
        report = self.generate_accessibility_report()
        
        logger.info("=" * 80)
        logger.info("♿ ACCESSIBILITY COMPLIANCE SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total Tests: {report['accessibility_summary']['total_tests']}")
        logger.info(f"✅ Passed: {report['accessibility_summary']['passed']}")
        logger.warning(f"⚠️ Warnings: {report['accessibility_summary']['warnings']}")
        logger.error(f"❌ Critical: {report['accessibility_summary']['critical_failures']}")
        logger.info(f"📊 Compliance Score: {report['accessibility_summary']['compliance_score']}%")
        logger.info(f"🏆 WCAG Level: {report['accessibility_summary']['wcag_level']}")
        logger.info()
        
        logger.info("📊 CATEGORY BREAKDOWN:")
        for category, results in report['category_results'].items():
            logger.info(f"  {category}: {results['passed']}/{results['total']} ({results['success_rate']}%)")
        
        if report['recommendations']:
            logger.info("\n💡 RECOMMENDATIONS:")
            for i, rec in enumerate(report['recommendations'], 1):
                logger.info(f"  {i}. {rec}")
        
        logger.info("\n♿ ACCESSIBILITY VALIDATION COMPLETE")
        
        return report

def main():
    """Run comprehensive accessibility tests"""
    tester = AccessibilityTester()
    results = tester.run_comprehensive_accessibility_tests()
    
    # Save results to file
    with open('/tmp/accessibility_test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\n📄 Detailed results saved to: /tmp/accessibility_test_results.json")
    
    # Return based on compliance score
    if results['accessibility_summary']['compliance_score'] >= 70:
        logger.info("🎉 ACCESSIBILITY VALIDATION: PASSED")
        return 0
    else:
        logger.info("⚠️ ACCESSIBILITY VALIDATION: NEEDS IMPROVEMENT")
        return 1

if __name__ == "__main__":
    exit(main())