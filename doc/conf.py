# -*- coding: utf-8 -*-

import os
import sys

sys.path.append(os.path.abspath('../'))

release = os.getenv("PAGES_DEPLOY_PATH","dev").lstrip("refs/tags/")

print(release)

extensions = [
    "sphinx.ext.mathjax",
    "sphinx.ext.ifconfig",
    "sphinx.ext.autosummary",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    # "process-docstring",
    'm2r2'
]

autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "inherited-members": True,
    "member-order": "bysource",
}

source_suffix = ".rst"
master_doc = "index"

project = u"OpenMM ML Docs"
copyright = u"2023, Stanford University and the Authors"


exclude_patterns = ["_build", "_templates"]
html_static_path = ["_static"]
templates_path = ["_templates"]

pygments_style = "sphinx"

html_theme = "pydata_sphinx_theme"

html_theme_options = {
    "logo": {
        "text": "OpenMM-ML docs",
        "image_light": "_static/logo.png",
        "image_dark": "_static/logo.png",
    },

    "external_links": [
      {"name": "OpenMM.org", "url": "https://openmm.org/"},
      {"name": "OpenMM docs", "url": "https://openmm.org/documentation"},
      {"name": "GitHub", "url": "https://github.com/openmm"}
    ],

    "github_url": "https://github.com/openmm/openmm-ml",

    "switcher": {
        "json_url": "https://sef43.github.io/openmm-ml-docs-test/versions.json",
        "version_match": release,
    },

    "check_switcher": False,

    "navbar_start": ["navbar-logo", "version-switcher"],

    "show_version_warning_banner": True,
}
# html_theme_options = {
#     "github_button": False,
#     "github_user": "openmm",
#     "github_repo": "openmm",
#     "logo_name": True,
#     "logo": "logo.png",
#     "extra_nav_links": [
#         {
#             "title": "OpenMM.org",
#             "uri": "https://openmm.org",
#             "relative": False,
#         },
#         {
#             "title": "User's Manual",
#             "uri": "../userguide/",
#             "relative": True,
#         },
#         {
#             "title": "Developer Guide",
#             "uri": "../developerguide/",
#             "relative": True,
#         },
#         {
#             "title": "C++ API reference",
#             "uri": "../api-c++/",
#             "relative": True,
#         },
#         {
#             "title": "GitHub",
#             "uri": "https://github.com/openmm",
#             "relative": False,
#         },
#     ],
#     "show_relbar_bottom": True,
# }
# html_sidebars = {
#     "**": [
#         "about.html",
#         "searchbox.html",
#         "navigation.html",
#     ]
# }

# Napoleon settings
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
