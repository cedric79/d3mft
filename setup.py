import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="laim", # Replace with your own username
    version="0.0.1",
    author="Evan Sheridan",
    author_email="sheridev@tcd.ie",
    description="laim uses machine learning to solve the anderson impurity model",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/evan1415/laim",
    packages=setuptools.find_packages(),
    #packages=setuptools.find_packages(where="laim"),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
