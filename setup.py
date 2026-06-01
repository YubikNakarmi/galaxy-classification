from setuptools import setup,find_packages

setup (
    author = "Yubik Nakarmi",
    description=" A package which everyting to run galaxy classification",
    name = "galaxy classification",
    version = "0.1.0",
    packages= find_packages(include =["src","src.*"] )
    )