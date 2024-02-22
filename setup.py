import setuptools
with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="Hough-TMF",
    version="0.2.2",
    author="Hao Lv",
    author_email="lh21@apm.ac.cn",
    description="A package for template matching using Torch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    license='MIT',
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9',
)


