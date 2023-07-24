from setuptools import setup

setup(
    name="tekscope",
    version="0.1",
    description="Library for interacting with Tektronix oscilloscopes.",
    url="https://github.com/rohanku/tekscope",
    author="Rohan Kumar",
    author_email="rohankumar@berkeley.edu",
    license="BSD",
    packages=["tekscope"],
    entry_points={
        "console_scripts": ["tekscope=apps.cli:main"],
    },
    install_requires=["matplotlib", "argparse", "numpy"],
)
