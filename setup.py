from setuptools import setup, find_packages

VERSION = "0.1.0"
DESCRIPTION = ""
with open("README.md", encoding="utf-8") as f:
    LONG_DESCRIPTION = f.read()

setup(
    name="carefree-learn-deploy",
    version=VERSION,
    packages=find_packages(exclude=("tests",)),
    install_requires=[],
    author="carefree0910",
    author_email="syameimaru.saki@gmail.com",
    url="https://github.com/carefree0910/carefree-learn-deploy",
    download_url=f"https://github.com/carefree0910/carefree-learn-deploy/archive/v{VERSION}.tar.gz",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    keywords="python carefree-learn PyTorch",
)
