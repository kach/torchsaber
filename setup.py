import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="torchsaber",
    version="1.0.2",
    author="Kartik Chandra",
    author_email="",
    description="Elegant dimensions for a more civilized age",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kach/torchsaber",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
