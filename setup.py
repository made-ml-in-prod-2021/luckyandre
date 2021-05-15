from setuptools import find_packages, setup

setup(
    name="ml_project",
    packages=find_packages(),
    version="0.1.0",
    description="Example of ml project",
    author="Andrei Belenko",
    install_requires=[
        "notebook == 6.3.0",
        "pandas == 1.2.4",
        "pandas-profiling == 3.0.0",
        "click == 7.0",
        "scikit-learn==0.24.2 ",
        "dataclasses==0.6",
        "marshmallow==3.12.1",
        "marshmallow-dataclass==8.4.1",
        "pytest=6.2.4 "
    ]
)