# PyIter

[![Pypi version](https://img.shields.io/pypi/v/pyiter?style=for-the-badge)](https://pypi.org/project/pyiter/)

PyIter is a Python package for iterative operations inspired by the Kotlin、CSharp(linq)、TypeSrcipt and Rust .
Enables strong **typing** and type inference for iterative operations.

- Chain operations like map, reduce, filter, map
- Lazy evaluation
- parallel execution
- strong **typing**

## Install

```bash
pip install pyiter
```

## Quickstart

```python
from pyiter import iterate as it
from tqdm import tqdm

text = ["hello", "world"]
it(text).map(str.upper).to_list()
# ['HELLO', 'WORLD']

# use tqdm
it(range(10)).map(lambda x: str(x)).progress(lambda x: tqdm(x, total=x.len)).parallel_map(lambda x: x, max_workers=5).to_list()

```


**Type inference**
![.](https://github.com/mokeyish/pyiter/raw/master/screenshots/screenshot.png)

## API

See [API](https://pyiter.yish.org/pyiter/sequence.html) documention.

- You no need to read api documention. all api functions are listed by the code completion as follows.
  
   ![.](https://github.com/mokeyish/pyiter/raw/master/screenshots/apilist.png)

- All documentions are showed as follows.

   ![.](https://github.com/mokeyish/pyiter/raw/master/screenshots/apidoc.png)

## Similar libraries

Note that none of the following libraries are providing full strong typing for code completion.

- [Pyterator](https://github.com/remykarem/pyterator)
- [PyFunctional](https://github.com/EntilZha/PyFunctional)
- [fluent](https://github.com/dwt/fluent)
- [Simple Smart Pipe](https://github.com/sspipe/sspipe)
- [pyxtension](https://github.com/asuiu/pyxtension)