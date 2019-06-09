# ConTrie

[![Travis Build Status](https://api.travis-ci.org/vorner/contrie.png?branch=master)](https://travis-ci.org/vorner/contrie)
[![AppVeyor Build status](https://ci.appveyor.com/api/projects/status/9e0mmfqqp4o9ap5c/branch/master?svg=true)](https://ci.appveyor.com/project/vorner/contrie/branch/master)

A concurrent hash-trie map & set.

Still in somewhat experimental state, large parts are missing and some API
changes are to be expected (though it'll probably still stay being a concurrent
map & set).

Inspired by this [article] and [wikipedia entry], though simplified
significantly (at the cost of some features).

Read [the documentation](https://docs.rs/contrie) before using, there are some
quirks to be aware of.

## License

Licensed under either of

 * Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally
submitted for inclusion in the work by you, as defined in the Apache-2.0
license, shall be dual licensed as above, without any additional terms
or conditions.

[article]: https://www.researchgate.net/publication/221643801_Concurrent_Tries_with_Efficient_Non-Blocking_Snapshots
[wikipedia entry]: https://en.wikipedia.org/wiki/Ctrie
