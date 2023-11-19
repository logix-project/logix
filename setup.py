from setuptools import setup, find_packages


with open("requirements.txt", "r") as f:
    requirements = f.read().splitlines()

python_requires = ">=3.9.0,<3.11.0"

if __name__ == "__main__":
    setup(
        name="analog",
        version="0.0.0",
        packages=find_packages(
            exclude=[
                "examples",
                "tests",
            ]
        ),
        install_requires=requirements,
        python_requires=python_requires,
    )
