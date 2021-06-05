from setuptools import find_packages, setup

setup(
    name="online_inference",
    packages=find_packages(),
    version="0.1.0",
    description="example docker usage",
    author="Andrei Belenko",
    install_requires=[
        "fastapi == 0.65.1",
        "uvicorn == 0.13.4",
        "scikit-learn == 0.24.2",
        "pandas == 1.2.4",
        "marshmallow-dataclass == 8.4.1",
        "pytest == 6.2.4",
        "requests == 2.25.1"
    ]
)