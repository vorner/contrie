//! The [`ConSet`] and other related structures.

use std::borrow::Borrow;
use std::collections::hash_map::RandomState;
use std::hash::{BuildHasher, Hash};

use crossbeam_epoch;

use crate::raw::config::Trivial as TrivialConfig;
use crate::raw::Raw;

/// A concurrent lock-free set.
///
/// Note that due to the limitations described in the crate level docs, values returned by looking
/// up (or misplacing or removing) are always copied using the `Clone` trait. Therefore, the set is
/// more suitable for types that are cheap to copy (eg. `u64` or `IpAddr`).
///
/// If you intend to store types that are more expensive to make copies of or are not `Clone`, you
/// can wrap them in an `Arc` (eg. `Arc<str>`).
pub struct ConSet<T, S = RandomState>
where
    T: Clone + Hash + Eq + 'static,
{
    raw: Raw<TrivialConfig<T>, S>,
}

impl<T> ConSet<T, RandomState>
where
    T: Clone + Hash + Eq + 'static,
{
    /// Creates a new empty set.
    pub fn new() -> Self {
        Self::with_hasher(RandomState::default())
    }
}

impl<T, S> ConSet<T, S>
where
    T: Clone + Hash + Eq + 'static,
    S: BuildHasher,
{
    /// Creates a new empty set with the given hasher.
    pub fn with_hasher(hasher: S) -> Self {
        Self {
            raw: Raw::with_hasher(hasher),
        }
    }

    /// Inserts a new value into the set.
    ///
    /// It returns the previous value, if any was present.
    pub fn insert(&self, value: T) -> Option<T> {
        let pin = crossbeam_epoch::pin();
        self.raw.insert(value, &pin).cloned()
    }

    /// Looks up a value in the set.
    ///
    /// This creates a copy of the original value.
    pub fn get<Q>(&self, key: &Q) -> Option<T>
    where
        Q: ?Sized + Eq + Hash,
        T: Borrow<Q>,
    {
        let pin = crossbeam_epoch::pin();
        self.raw.get(key, &pin).cloned()
    }

    /// Checks if a value identified by the given key is present in the set.
    ///
    /// Note that by the time you can act on it, the presence of the value can change (eg. other
    /// thread can add or remove it in the meantime).
    pub fn contains<Q>(&self, key: &Q) -> bool
    where
        Q: ?Sized + Eq + Hash,
        T: Borrow<Q>,
    {
        let pin = crossbeam_epoch::pin();
        self.raw.get(key, &pin).is_some()
    }

    /// Removes an element identified by the given key, returning it.
    pub fn remove<Q>(&self, key: &Q) -> Option<T>
    where
        Q: ?Sized + Eq + Hash,
        T: Borrow<Q>,
    {
        let pin = crossbeam_epoch::pin();
        self.raw.remove(key, &pin).cloned()
    }

    /// Checks if the set is currently empty.
    ///
    /// Note that due to being concurrent, the use-case of this method is mostly for debugging
    /// purposes, because the state can change between reading the value and acting on it.
    pub fn is_empty(&self) -> bool {
        self.raw.is_empty()
    }
}

impl<T> Default for ConSet<T, RandomState>
where
    T: Clone + Hash + Eq + 'static,
{
    fn default() -> Self {
        Self::new()
    }
}

// TODO: Tests
// TODO: Iterators
