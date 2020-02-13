import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="VirusDB",
    version="0.0.1",
    author="Dense AI",
    author_email="DenseAI@outlook.com",
    description="Virus AI platform",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/denseai/virusdb",
    python_requires=">=3.6",
    packages=["denseai.virusdb"],
    package_dir={"denseai.virusdb": "denseai.virusdb"},
    extras_require={
        "tf": ["tensorflow"],
        "tfgpu": ["tensorflow-gpu"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache License 2.0",
        "Operating System :: OS Independent",
    ],
)