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


words = 'I dont want to believe I want to know'.split()
it(words).group_by(lambda x: x).map(lambda g: (g.key, g.values.count())).to_list()
# [('I', 2), ('dont', 1), ('want', 2), ('to', 2), ('believe', 1), ('know', 1)]


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


## License

Licensed under either of

- Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or <https://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or <https://opensource.org/licenses/MIT>)

at your option.


### Contribution

Unless you explicitly state otherwise, any contribution intentionally
submitted for inclusion in the work by you, as defined in the Apache-2.0
license, shall be dual licensed as above, without any additional terms or
conditions.