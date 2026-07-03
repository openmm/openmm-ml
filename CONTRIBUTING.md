## How to Contribute to OpenMM-ML Development

We welcome anyone who wants to contribute to the project, whether by adding a feature,
fixing a bug, or improving documentation.  The process is quite simple.

First, it is always best to begin by opening an issue on Github that describes the change you
want to make.  This gives everyone a chance to discuss it before you put in a lot of work.
For bug fixes, we will confirm that the behavior is actually a bug and that the proposed fix
is correct.  For new features, we will decide whether the proposed feature is something we
want and discuss possible designs for it.

Once everyone is in agreement, the next step is to
[create a pull request](https://help.github.com/en/articles/about-pull-requests) with the code changes.
For larger features, feel free to create the pull request even before the implementation is
finished so as to get early feedback on the code.  When doing this, put the letters "WIP" at
the start of the title of the pull request to indicate it is still a work in progress.

For new features, your PR should include documentation and tests along with the implementation, as needed and appropriate.
You may find it useful to review the OpenMM-ML [developer API documentation](https://openmm.github.io/openmm-ml/dev/api.html#developer-api).
If you are proposing to add a new potential or embedding method, be sure to include as necessary (in addition to the
module containing the implementation itself):

* An entry point or entry points in `setup.py`.
* A test module in `test/`.
* Any inputs/scripts used to generate reference values for test cases in `test/data/`.
* Updates to `devtools/requirements` and/or `.github/workflows/CI.yml` to specify any packages necessary for running the test cases.
* A subsection in `doc/userguide.md`.

The core developers will review the pull request and may suggest changes.  Simply push the
changes to the branch that is being pulled from, and they will automatically be added to the
pull request.  In addition, the full test suite is automatically run on every pull request,
and rerun every time a change is added.  Once the tests are passing and everyone is satisfied
with the code, the pull request will be merged.  Congratulations on a successful contribution!
