"""Task runner for pyFFTW developers

# Usage (see https://nox.thea.codes)

```
nox -l            # list of sessions.
nox -s <session>  # execute a session
nox -k <keyword>  # execute some session

# to build the documentation
nox -s doc -R

```

"""

import os

import nox


os.environ.update({"PDM_IGNORE_SAVED_PYTHON": "1"})
nox.options.reuse_existing_virtualenvs = True


@nox.session
def doc(session):
    """Build the documentation"""
    command = "pdm sync --clean -G doc"
    session.run_always(*command.split(), external=True)

    session.chdir("docs")
    session.run("make", "clean", external=True)
    session.run("make", "html", external=True)
