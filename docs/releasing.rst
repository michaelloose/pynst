Release procedure
=================

PyNST releases are built from immutable Git tags and published through PyPI
Trusted Publishing. The repository does not contain a long-lived PyPI token.

One-time setup
--------------

#. Create the GitHub environment ``pypi`` and protect it with a maintainer
   approval rule.
#. In the PyPI account's *Publishing* settings, register a pending GitHub
   Trusted Publisher with project name ``pynst``, owner ``michaelloose``,
   repository ``pynst``, workflow ``release.yml`` and environment ``pypi``.
   The first successful publication creates the PyPI project and converts the
   pending publisher into a normal one. A pending publisher does not reserve
   the project name before that first publication.
#. Import the GitHub repository into Read the Docs. The checked-in
   ``.readthedocs.yaml`` is discovered automatically.

For each release
----------------

#. Update ``__version__`` in ``src/pynst/_version.py`` and the version in
   ``CITATION.cff``.
#. Replace ``Unreleased`` in ``CHANGELOG.md`` with the release date.
#. Run tests, documentation and distribution checks locally.
#. Merge the release branch into ``main``.
#. Create and push the matching tag, for example ``v0.4.0``.
#. Verify the ``Publish release`` workflow and the new Read the Docs build.

The workflow rejects tags that do not exactly match the package version, runs
the complete tests, validates both distributions, installs the wheel once, and
only then requests the short-lived PyPI publishing credential.
