from setuptools import setup, find_packages

setup(
    name="cml_unit_value",
    version="0.1",
    description="A package for trade analysis and preparation",
    packages=find_packages(),  # Automatically discovers all modules
    install_requires=["numpy", "pandas"],  # Include dependencies
)