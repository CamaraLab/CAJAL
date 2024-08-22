This document is meant for developers of CAJAL rather than end users.

# Releasing a new version of CAJAL
You must follow a certain workflow in releasing a new version of CAJAL so that the Github Actions continuous integration workflow does certain mundane tasks automatically (uploading wheels as assets on the Github release page, and pushing the wheels to PyPI)

To release a new version of CAJAL,

1. Make the final git commit locally which is intended to be the publicly facing new release (or an alpha or beta version)
2. Run `git tag -a  v1.0.1-alpha.5 -m "Prerelease 5 of version 1.0.1-alpha of CAJAL"`, replacing  v1.0.1-alpha.5 with the desired tag number (and following semantic versioning conventions). The tag name *must* start with the letter `v` to trigger the CI.
3. Run `git push origin v1.0.1-alpha.5`, where origin is the name of the CAJAL github repository. You must include the name of the tag as the last part  in the push command, or it will not push the *tag* to Github, and the presence of the tag is what triggers the conditional branch in the CI script.
4. Wait and check to see that the wheels build correctly and are uploaded to PyPI.

N. b. - I estimate that around 80% of the compilation time for the wheels is spent compiling C++ code dealing with 64 bit floating point to run on 32 bit Linux platforms (i686), I assume this requires some kind of convoluted compilation technique. When making a new release, I suggest making an alpha or beta release of the new version first, in order to test and debug the continuous integration; and in the beta release, uncomment the line in the "Build wheels job" that filters out the i686 build targets (here)[https://github.com/CamaraLab/CAJAL/blob/b1521703c1b2d5dd79ed371fc9abff6fbeb46fe7/.github/workflows/python-package.yml#L38] 