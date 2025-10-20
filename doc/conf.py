# -*- coding: utf-8 -*-

import os
import sys
import openmmml
import pkg_resources
import git


sys.path.append(os.path.abspath("../"))


# version specified in ../setup.py
version = pkg_resources.require("openmmml")[0].version

repo = git.Repo(search_parent_directories=True)
short_sha = hash = repo.git.rev_parse(repo.head, short=True)

# get the the current tag if this commit has one
tag = next((tag for tag in repo.tags if tag.commit == repo.head.commit), None)

if tag is None:
    release = version + "dev_" + short_sha
    version_match = "dev"
    version = version_match
else:
    release = str(tag) + "_" + short_sha
    version_match = str(tag)
    version = version_match

print("version:", version)
print("git tag:", tag)
print("git sha:", short_sha)
print("release:", release)
print("version_match", version_match)


extensions = [
    "sphinx.ext.mathjax",
    "sphinx.ext.ifconfig",
    "sphinx.ext.autosummary",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "m2r2",
]

autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "inherited-members": True,
    "member-order": "bysource",
}

source_suffix = ".rst"
master_doc = "index"

project = "OpenMM ML"
copyright = "2025, Stanford University and the Authors"


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
        {"name": "GitHub", "url": "https://github.com/openmm"},
    ],
    "github_url": "https://github.com/openmm/openmm-ml",
}


# settings for version switcher and warning
html_theme_options["navbar_start"] = ["navbar-logo", "version-switcher"]
html_theme_options["switcher"] = {
    "json_url": "https://openmm.github.io/openmm-ml/dev/_static/versions.json",
    "version_match": version_match,
}

# https://github.com/pydata/pydata-sphinx-theme/issues/1552
html_theme_options["show_version_warning_banner"] = False
html_theme_options["check_switcher"] = False

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
