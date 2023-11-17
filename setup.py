from setuptools import find_packages, setup


def read_requirements():
    with open("requirements.txt") as req:
        content = req.read()
        requirements = content.split("\n")

    return requirements


setup(
    name="xllm-demo",
    version="0.0.1",
    packages=find_packages(),
    include_package_data=True,
    install_requires=read_requirements(),
)
