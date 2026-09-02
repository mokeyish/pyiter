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
build: clean
  uv build

# Upload distributions to PyPI
publish: build
  uvx twine upload --non-interactive --repository-url https://upload.pypi.org/legacy/ dist/*

# Generate type stubs
stubgen:
  rm -rf typings
  uv run pyright --createstub pyiter

# Build documentation using pdoc.
# pyiter-stubs is a symlink to pyright-generated stubs; pdoc would pick up
# their pyright header docstrings over the real ones, so hide the symlink
# for the duration of the render. The trap restores it even on failure.
# Stale generated files (e.g. from removed modules) are cleaned first,
# keeping CNAME (GitHub Pages) which pdoc does not regenerate.
docs:
  rm -f docs/index.html docs/pyiter.html docs/search.js
  rm -rf docs/pyiter
  trap '[ ! -e src/pyiter-stubs.hide ] || mv src/pyiter-stubs.hide src/pyiter-stubs' EXIT; mv src/pyiter-stubs src/pyiter-stubs.hide; uv run -m pdoc src/pyiter -t src/templates -o docs

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
  uv run pyright-python src/pyiter

# Format the code
format:
  @uv run ruff format

# Clean
clean:
  @rm -rf dist build
