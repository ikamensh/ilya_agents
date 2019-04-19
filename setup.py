import setuptools



with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ilya_agents",
    version="0.0.5",
    author="Ilya Kamenshchikov",
    author_email="ikamenshchikov@gmail.com",
    description="Experimental reinforcement learning implementations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ikamensh/ddpg",
    packages=['ilya_agents'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)