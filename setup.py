import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="daltonize", # Replace with your own username
    version="0.1.0",
    author="Jörg Dietrich",
    author_email="joerg@joergdietrich.com",
    description="simulate and correct for color blindness in matplotlib figures and images",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/joergdietrich/daltonize",
    packages=setuptools.find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        "License :: OSI Approved :: GNU General Public License v2 (GPLv2)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',

    install_requires=['numpy>=1.9', 'Pillow'],

    extras_require={
        'dev': ['pytest'],
    },

    # To provide executable scripts, use entry points in preference to the
    # "scripts" keyword. Entry points provide cross-platform support and allow
    # pip to create the appropriate form of executable for the target
    # platform.
    entry_points={
        'console_scripts': ['daltonize=daltonize.daltonize:main'],
        },
)