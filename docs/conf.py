# Configuration file for the Sphinx documentation builder.

import os
import sys
import datetime

# Add project root to Python path
sys.path.insert(0, os.path.abspath('..'))

# Project information
project = 'SutazAI'
copyright = f'{datetime.datetime.now().year}, Florin Cristian Suta'
author = 'Florin Cristian Suta'
release = '2.1.0'
version = '2.1.0'

# Extensions configuration
extensions = [
    'sphinx.ext.autodoc',      # Auto-generate documentation from docstrings
    'sphinx.ext.napoleon',     # Support for Google and NumPy docstring styles
    'sphinx.ext.viewcode',     # Add source code links
    'sphinx.ext.todo',         # Support for TODO notes
    'sphinx.ext.intersphinx',  # Link to other projects' documentation
    'sphinx.ext.coverage',     # Coverage reports
    'sphinx.ext.inheritance_diagram'  # Inheritance diagrams
]

# Source file parsing
source_suffix = '.rst'
source_encoding = 'utf-8-sig'

# Master document
master_doc = 'index'

# Exclude patterns
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# HTML output configuration
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_logo = '_static/logo.png'
html_favicon = '_static/favicon.ico'

# HTML theme options
html_theme_options = {
    'logo_only': False,
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': False,
    'style_nav_header_background': '#2980B9',
    'collapse_navigation': True,
    'sticky_navigation': True,
    'navigation_depth': 3,
    'includehidden': True,
    'titles_only': False
}

# Intersphinx configuration
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'torch': ('https://pytorch.org/docs/stable/', None)
}

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True

# Todo settings
todo_include_todos = True

# Autodoc settings
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}

# Additional configuration
add_module_names = True
pygments_style = 'sphinx'
highlight_language = 'python'

# Inheritance diagram settings
inheritance_graph_attrs = {
    'rankdir': 'TB',
    'size': '"8.0, 10.0"',
    'fontsize': 10,
    'ratio': 'compress'
}

# Latex configuration (optional)
latex_elements = {
    'papersize': 'letterpaper',
    'pointsize': '10pt',
    'preamble': r'\usepackage{multicol}'
}

# Latex documents
latex_documents = [
    (master_doc, 'SutazAI.tex', 'SutazAI Documentation', author, 'manual')
]