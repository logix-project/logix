from setuptools import setup, find_packages


def fetch_requirements(path: str):
    with open(path, "r") as f:
        requirements = f.read().splitlines()
        return requirements


python_requires = ">=3.6.0,<3.11.0"

if __name__ == "__main__":
    setup(
        name="analog",
        version="0.0.0",
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
