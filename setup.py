#!/usr/bin/env python
# encoding: utf-8

import re
import os

try:
    from setuptools import setup, Extension
    from setuptools.command.build_ext import build_ext as _build_ext
except ImportError:
    from distutils.core import setup, Extension
    from distutils.command.build_ext import build_ext as _build_ext


def find_eigen(hint=None):
    """
    Find the location of the Eigen 3 include directory. This will return
    ``None`` on failure.

    """
    # List the standard locations including a user supplied hint.
    search_dirs = [] if hint is None else hint
    search_dirs += [
        "/usr/local/include/eigen3",
        "/usr/local/homebrew/include/eigen3",
        "/opt/local/var/macports/software/eigen3",
        "/opt/local/include/eigen3",
        "/usr/include/eigen3",
        "/usr/include/local",
        "/usr/include",
    ]

    # Loop over search paths and check for the existence of the Eigen/Dense
    # header.
    for d in search_dirs:
        path = os.path.join(d, "Eigen", "Dense")
        if os.path.exists(path):
            # Determine the version.
            vf = os.path.join(d, "Eigen", "src", "Core", "util", "Macros.h")
            if not os.path.exists(vf):
                continue
            src = open(vf, "r").read()
            v1 = re.findall("#define EIGEN_WORLD_VERSION (.+)", src)
            v2 = re.findall("#define EIGEN_MAJOR_VERSION (.+)", src)
            v3 = re.findall("#define EIGEN_MINOR_VERSION (.+)", src)
            if not len(v1) or not len(v2) or not len(v3):
                continue
            v = "{0}.{1}.{2}".format(v1[0], v2[0], v3[0])
            print("Found Eigen version {0} in: {1}".format(v, d))
            return d
    return None


class build_ext(_build_ext):
    """
    A custom extension builder that finds the include directories for Boost.

    """

    def build_extension(self, ext):
        dirs = ext.include_dirs + self.compiler.include_dirs

        # Look for the Eigen headers and make sure that we can find them.
        eigen_include = find_eigen(hint=dirs)
        if eigen_include is None:
            raise RuntimeError("Required library Eigen not found. "
                               "Check the documentation for solutions.")

        # Update the extension's include directories.
        ext.include_dirs += [eigen_include]
        ext.extra_compile_args += ["-Wno-unused-function",
                                   "-Wno-uninitialized"]

        # Run the standard build procedure.
        _build_ext.build_extension(self, ext)


if __name__ == "__main__":
    import sys
    import numpy

    # Publish the library to PyPI.
    if "publish" in sys.argv[-1]:
        os.system("python setup.py sdist upload")
        sys.exit()

    # Choose libraries to link.
    libraries = []
    if os.name == "posix":
        libraries.append("m")

    # Specify the include directories.
    include_dirs = [
        "carma",
        numpy.get_include(),
    ]

    # Check for the Cython source (development mode) and compile it if it
    # exists.
    src = os.path.join("carma", "_carma")
    if os.path.exists(src + ".pyx"):
        from Cython.Build import cythonize
        src = [src + ".pyx"]
    else:
        src = [src + ".cpp"]

        def cythonize(x):
            return x

    # Set up the extension.
    ext = Extension("carma._carma", sources=src,
                    libraries=libraries, include_dirs=include_dirs)

    # Hackishly inject a constant into builtins to enable importing of the
    # package before the library is built.
    if sys.version_info[0] < 3:
        import __builtin__ as builtins
    else:
        import builtins
    builtins.__CARMA_SETUP__ = True
    import carma

    # Execute the setup command.
    desc = open("README.rst").read()
    setup(
        name="carma",
        version=carma.__version__,
        author="Daniel Foreman-Mackey",
        author_email="foreman.mackey@gmail.com",
        packages=["carma"],
        # py_modules=["transit.tests"],
        ext_modules=cythonize([ext]),
        url="http://github.com/dfm/carma",
        license="MIT",
        description="",
        long_description=desc,
        package_data={"": ["README.rst", "LICENSE", "carma/*.h"]},
        include_package_data=True,
        cmdclass=dict(build_ext=build_ext),
        classifiers=[
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
            "Programming Language :: Python",
        ],
    )
