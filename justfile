# list available commands
default:
    @just --list

# run tests arguments: -v to verbose
test args='':
    python src/pyiter/sequence.py {{args}}

# package as wheel
package:
    rm -rf dist
    python -m build

# publish package to pypi
publish: package
    python -m twine upload --repository pypi dist/*

# build docs
docs:
    python -m pdoc src/pyiter -t src/templates -o docs

# install dev requirements
dev:
    python -m pip install -r requirements-dev.txt
