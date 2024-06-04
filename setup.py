import ast
import re

from setuptools import find_packages, setup


def fetch_requirements(path: str):
    with open(path, "r") as f:
        requirements = f.read().splitlines()
        return requirements


_version_re = re.compile(r"__version__\s+=\s+(.*)")

with open("logix/__init__.py", "rb") as f:
    version = str(
        ast.literal_eval(_version_re.search(f.read().decode("utf-8")).group(1))
    )

python_requires = ">=3.6.0,<3.11.0"

if __name__ == "__main__":
    setup(
        name="logix-ai",
        version=version,
        description="AI Logging for Interpretability and Explainability",
        license="Apache-2.0",
        packages=find_packages(
            exclude=[
                "examples",
                "tests",
                "tests.*",
                "*.tests.*",
            ]
        ),
        install_requires=fetch_requirements("requirements.txt"),
        extras_require={
            "test": fetch_requirements("test-requirements.txt"),
        },
        python_requires=python_requires,
    )
