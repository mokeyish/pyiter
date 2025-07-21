alias b := build
alias t := test
alias c := check
alias v := version
alias fmt := format


# list available commands
[private]
default:
  @just --list

#------------#
# versioning #
#------------#

# Increment manifest version: major, minor, patch
bump BUMP="patch":
  @uv version --bump {{BUMP}}

# Print current version
version:
  @uv version --short

#----------#
# building #
#----------#

# Build Python packages into source distributions and wheels
build:
  @rm -rf dist build
  uv build

# Upload distributions to an index
publish: build
  uvx twine upload --repository pypi dist/*

# Build documentation using pdoc
docs:
  uv run -m pdoc src/pyiter -t src/templates -o docs

#---------------#
# running tests #
#---------------#

# Run tests arguments: -v to verbose
test *args:
  uv run src/pyiter/sequence.py {{args}}

#-----------------------#
# code quality and misc #
#-----------------------#


# Check the code cleanliness
check: type-check
  @uvx ruff check
  @uvx ruff format --check

# Strict type checking
type-check:
  uv run pyright-python src/*

# Format the code
format:
  @uvx ruff format

