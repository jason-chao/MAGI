from setuptools import setup, find_packages

setup(
    name="magi",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "litellm",
        "pyyaml",
        "python-dotenv",
        "termcolor"
    ],
    scripts=['magi-cli.py'],
)
