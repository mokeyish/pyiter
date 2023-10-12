from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name='pyiter',
    version='0.7.2',
    keywords=['linq', 'iterator', 'typing', 'lazy evaluation', 'type inference'],
    description='PyIter is a Python package for iterative operations inspired by the Kotlin、CSharp(linq)、TypeSrcipt '
                'and Rust . Enables strong typing and type inference for iterative operations.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='YISH',
    author_email="mokeyish@hotmail.com",
    url= 'https://github.com/mokeyish/pyiter',
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    license='MIT',
    python_requires='>=3.8',
    install_requires = [
        'Deprecated'
    ]
)