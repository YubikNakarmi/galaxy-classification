from setuptools import setup,find_packages

setup (
    author = "Yubik Nakarmi",
    description=" A package which everyting to run galaxy classification",
    name = "galaxy-classification",
    version = "v0.6.0",
    packages= find_packages(include =["src","src.*"] ),
    install_requires=[
        "torch","torchvision","pandas","numpy","scikit-learn","loguru",
        "matplotlib","azure-storage-blob", "azure-identity" ]
    )