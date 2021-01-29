from setuptools import setup

v_temp = {}
with open("bw_processing/version.py") as fp:
    exec(fp.read(), v_temp)
version = ".".join((str(x) for x in v_temp["version"]))


setup(
    name="bw_processing",
    version=version,
    packages=["bw_processing", "bw_processing.examples"],
    author="Chris Mutel",
    author_email="cmutel@gmail.com",
    license="BSD 3-clause",
    install_requires=["numpy", "pandas", "fs"],
    url="https://github.com/brightway-lca/bw_processing",
    long_description_content_type="text/markdown",
    long_description=open("README.md").read(),
    package_data={"bw_processing": ["examples/*.zip"]},
    description="Tools to create structured arrays in a common format",
    classifiers=[
        "Intended Audience :: End Users/Desktop",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Visualization",
    ],
)
