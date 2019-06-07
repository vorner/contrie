//! A concurrent trie.
//!
//! The crate provides implementation of a hash map and a hash set. Unlike the versions from the
//! standard library, these allow concurrent accesses and *modifications* from multiple threads at
//! once, without locking. Actually, even the inner implementation is [lock-free] and the lookups
//! are [wait-free] (though even the lookup methods may invoke collecting garbage in
//! [`crossbeam-epoch`] which while unlikely might contain non-wait-free functions).
//!
//! # Downsides
//!
//! The concurrent access does not come for free, if you can, you should prefer using the usual
//! single-threaded alternatives or consider combining them with locks. In particular, there are
//! several downsides.
//!
//! * Under the hood, the [crossbeam-epoch] is used to manage memory. This has the effect that
//!   removed elements are not deleted at precisely known moment. While the [crossbeam-epoch] is
//!   usually reasonably fast in collecting garbage, this might be unsuitable for object with
//!   observable side effects in their destructors (like, containing open files that need to be
//!   flushed and closed).
//! * As even after removing an element this element might be still being accessed by another
//!   thread, there's no way to get an owned access to the original element once it is inserted.
//!   Depending on the flavour, the data structure either clones the data or returns [`Arc`]s to
//!   them.
//! * They are slower in single-threaded usage and use more memory than their standard library
//!   counter-parts. While the mileage may differ, 2-3 times slowdown was measured in trivial
//!   benchmarks.
//! * The order of modifications as seen by different threads may be different (if one thread
//!   inserts elements `a` and `b`, another thread may see `b` already inserted while `a` still not
//!   being present). In other words, the existence of values of different keys is considered
//!   independent on each other. This limitation might be lifted in the future by letting the user
//!   configure the desired sequentially consistent behaviour at the cost of further slowdown.
//! * Iteration doesn't take a snapshot at a given time. In other words, if the data structure is
//!   modified during the iteration (even if by the same thread that iterates), the changes may or
//!   may not be reflected in the list of iterated elements.
//!
//! # The gist of the data structure
//!
//! Internally, the data structure is an array-mapped trie. At each level there's an array of 16
//! pointers. They get indexed by 4 bits of the hash of the key. The branches are kept as short as
//! possible to still distinguish between different hashes (once a subtree contains only one value,
//! it contains only the data leaf without further intermediate levels). These pointers can be
//! updated atomically with compare-and-swap operations.
//!
//! The idea was inspired by this [article], but with severe simplifications. The simplifications
//! were done to lower the number of traversed pointers during operations and to gain more
//! confidence in correctness of the implementation, however at the cost of the snapshots for
//! iteration and higher memory consumption. Pruning of the trie of unneeded branches on removals
//! was preserved.
//!
//! For further technical details, arguments of correctness and similar, see the source code and
//! comments in there, especially the [`raw`] module.
//!
//! [wait-free]: https://en.wikipedia.org/wiki/Non-blocking_algorithm#Wait-freedom
//! [lock-free]: https://en.wikipedia.org/wiki/Non-blocking_algorithm#Lock-freedom
//! [crossbeam-epoch]: https://docs.rs/crossbeam-epoch
//! [`Arc`]: std::sync::Arc
//! [article]: https://www.researchgate.net/publication/221643801_Concurrent_Tries_with_Efficient_Non-Blocking_Snapshots

mod existing_or_new;
pub mod map;
pub mod raw;
// Some integration-like tests live here, instead of crate/tests. This is because this allows cargo
// to compile them in parallel with the crate and also run them more in parallel. And I like to get
// all the test failures at once.
//
// Interface is tested through doctests anyway.
#[cfg(test)]
mod tests;

pub use self::existing_or_new::ExistingOrNew;
pub use self::map::ConMap;
