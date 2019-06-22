//! The [`ConSet`] and other related structures.

use std::borrow::Borrow;
use std::collections::hash_map::RandomState;
use std::fmt::{Debug, Formatter, Result as FmtResult};
use std::hash::{BuildHasher, Hash};
use std::iter::FromIterator;

use crossbeam_epoch;

use crate::raw::config::Trivial as TrivialConfig;
use crate::raw::{self, Raw};

/// A concurrent lock-free set.
///
/// Note that due to the limitations described in the crate level docs, values returned by looking
/// up (or misplacing or removing) are always copied using the `Clone` trait. Therefore, the set is
/// more suitable for types that are cheap to copy (eg. `u64` or `IpAddr`).
///
/// If you intend to store types that are more expensive to make copies of or are not `Clone`, you
/// can wrap them in an `Arc` (eg. `Arc<str>`).
///
/// ```rust
/// use contrie::ConSet;
/// use crossbeam_utils::thread;
///
/// let set = ConSet::new();
///
/// thread::scope(|s| {
///     s.spawn(|_| {
///         set.insert("hello");
///     });
///     s.spawn(|_| {
///         set.insert("world");
///     });
/// }).unwrap();
///
/// assert_eq!(Some("hello"), set.get("hello"));
/// assert_eq!(Some("world"), set.get("world"));
/// assert_eq!(None, set.get("universe"));
/// set.remove("world");
/// assert_eq!(None, set.get("world"));
/// ```
///
/// ```rust
/// use contrie::set::{ConSet};
/// let set: ConSet<usize> = ConSet::new();
///
/// set.insert(0);
/// set.insert(1);
///
/// assert!(set.contains(&1));
///
/// set.remove(&1);
/// assert!(!set.contains(&1));
///
/// set.remove(&0);
/// assert!(set.is_empty());
/// ```
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

impl<T, S> Debug for ConSet<T, S>
where
    T: Debug + Clone + Hash + Eq + 'static,
{
    fn fmt(&self, fmt: &mut Formatter) -> FmtResult {
        let mut d = fmt.debug_set();
        for n in self {
            d.entry(&n);
        }
        d.finish()
    }
}

impl<T, S> ConSet<T, S>
where
    T: Clone + Hash + Eq + 'static,
{
    /// Returns an iterator through the elements of the set.
    pub fn iter(&self) -> Iter<T, S> {
        Iter {
            inner: raw::iterator::Iter::new(&self.raw),
        }
    }
}

/// The iterator of the [`ConSet`].
///
/// See the [`iter`][ConSet::iter] method for details.
pub struct Iter<'a, T, S>
where
    T: Clone + Hash + Eq + 'static,
{
    inner: raw::iterator::Iter<'a, TrivialConfig<T>, S>,
}

impl<'a, T, S> Iterator for Iter<'a, T, S>
where
    T: Clone + Hash + Eq + 'static,
{
    type Item = T;

    fn next(&mut self) -> Option<T> {
        self.inner.next().cloned()
    }
}

impl<'a, T, S> IntoIterator for &'a ConSet<T, S>
where
    T: Clone + Hash + Eq + 'static,
{
    type Item = T;
    type IntoIter = Iter<'a, T, S>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, T, S> Extend<T> for &'a ConSet<T, S>
where
    T: Clone + Hash + Eq + 'static,
    S: BuildHasher,
{
    fn extend<I>(&mut self, iter: I)
    where
        I: IntoIterator<Item = T>,
    {
        for n in iter {
            self.insert(n);
        }
    }
}

impl<T, S> Extend<T> for ConSet<T, S>
where
    T: Clone + Hash + Eq + 'static,
    S: BuildHasher,
{
    fn extend<I>(&mut self, iter: I)
    where
        I: IntoIterator<Item = T>,
    {
        let mut me: &ConSet<_, _> = self;
        me.extend(iter);
    }
}

impl<T> FromIterator<T> for ConSet<T>
where
    T: Clone + Hash + Eq + 'static,
{
    fn from_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = T>,
    {
        let mut me = ConSet::new();
        me.extend(iter);
        me
    }
}

#[cfg(test)]
mod tests {
    use crossbeam_utils::thread;

    use super::*;
    use crate::raw::tests::NoHasher;
    use crate::raw::LEVEL_CELLS;

    const TEST_THREADS: usize = 4;
    const TEST_BATCH: usize = 10000;
    const TEST_BATCH_SMALL: usize = 100;
    const TEST_REP: usize = 20;

    #[test]
    fn debug_when_empty() {
        let set: ConSet<String> = ConSet::new();
        assert_eq!("{}".to_owned(), format!("{:?}", set));
    }

    #[test]
    fn debug_when_has_elements() {
        let set: ConSet<&str> = ConSet::new();
        assert!(set.insert("hello").is_none());
        assert!(set.insert("world").is_none());
        assert_eq!("{\"hello\", \"world\"}".to_owned(), format!("{:?}", set));
    }

    #[test]
    fn debug_when_elements_are_added_and_removed() {
        let set: ConSet<&str> = ConSet::new();
        assert_eq!("{}".to_owned(), format!("{:?}", set));
        assert!(set.insert("hello").is_none());
        assert!(set.insert("world").is_none());
        assert_eq!("{\"hello\", \"world\"}".to_owned(), format!("{:?}", set));
        assert!(set.remove("world").is_some());
        assert_eq!("{\"hello\"}".to_owned(), format!("{:?}", set));
        assert!(set.remove("hello").is_some());
        assert_eq!("{}".to_owned(), format!("{:?}", set));
    }

    #[test]
    fn create_destroy() {
        let set: ConSet<String> = ConSet::new();
        drop(set);
    }

    #[test]
    fn lookup_empty() {
        let set: ConSet<String> = ConSet::new();
        assert!(set.get("hello").is_none());
    }

    #[test]
    fn insert_lookup() {
        let set = ConSet::new();
        assert!(set.insert("hello").is_none());
        assert!(set.get("world").is_none());
        let found = set.get("hello").unwrap();
        assert_eq!("hello", found);
    }

    // Insert a lot of things, to make sure we have multiple levels.
    #[test]
    fn insert_many() {
        let set = ConSet::new();
        for i in 0..TEST_BATCH * LEVEL_CELLS {
            assert!(set.insert(i).is_none());
        }

        for i in 0..TEST_BATCH * LEVEL_CELLS {
            assert_eq!(i, set.get(&i).unwrap());
        }
    }

    #[test]
    fn par_insert_many() {
        for _ in 0..TEST_REP {
            let set: ConSet<usize> = ConSet::new();
            thread::scope(|s| {
                for t in 0..TEST_THREADS {
                    let set = &set;
                    s.spawn(move |_| {
                        for i in 0..TEST_BATCH {
                            let num = t * TEST_BATCH + i;
                            assert!(set.insert(num).is_none());
                        }
                    });
                }
            })
            .unwrap();

            for i in 0..TEST_BATCH * TEST_THREADS {
                assert_eq!(set.get(&i).unwrap(), i);
            }
        }
    }

    #[test]
    fn par_get_many() {
        for _ in 0..TEST_REP {
            let set = ConSet::new();
            for i in 0..TEST_BATCH * TEST_THREADS {
                assert!(set.insert(i).is_none());
            }
            thread::scope(|s| {
                for t in 0..TEST_THREADS {
                    let set = &set;
                    s.spawn(move |_| {
                        for i in 0..TEST_BATCH {
                            let num = t * TEST_BATCH + i;
                            assert_eq!(set.get(&num).unwrap(), num);
                        }
                    });
                }
            })
            .unwrap();
        }
    }

    #[test]
    fn no_collisions() {
        let set = ConSet::with_hasher(NoHasher);
        // While their hash is the same under the hasher, they don't kick each other out.
        for i in 0..TEST_BATCH_SMALL {
            assert!(set.insert(i).is_none());
        }
        // And all are present.
        for i in 0..TEST_BATCH_SMALL {
            assert_eq!(i, set.get(&i).unwrap());
        }
        // No key kicks another one out.
        for i in 0..TEST_BATCH_SMALL {
            assert_eq!(i, set.insert(i).unwrap());
        }
    }

    #[test]
    fn simple_remove() {
        let set = ConSet::new();
        assert!(set.remove(&42).is_none());
        assert!(set.insert(42).is_none());
        assert_eq!(42, set.get(&42).unwrap());
        assert_eq!(42, set.remove(&42).unwrap());
        assert!(set.get(&42).is_none());
        assert!(set.is_empty());
        assert!(set.remove(&42).is_none());
        assert!(set.is_empty());
    }

    fn remove_many_inner<H: BuildHasher>(mut set: ConSet<usize, H>, len: usize) {
        for i in 0..len {
            assert!(set.insert(i).is_none());
        }
        for i in 0..len {
            assert_eq!(i, set.get(&i).unwrap());
            assert_eq!(i, set.remove(&i).unwrap());
            assert!(set.get(&i).is_none());
            set.raw.assert_pruned();
        }

        assert!(set.is_empty());
    }

    #[test]
    fn remove_many() {
        remove_many_inner(ConSet::new(), TEST_BATCH);
    }

    #[test]
    fn remove_many_collision() {
        remove_many_inner(ConSet::with_hasher(NoHasher), TEST_BATCH_SMALL);
    }

    #[test]
    fn collision_remove_one_left() {
        let mut set = ConSet::with_hasher(NoHasher);
        set.insert(1);
        set.insert(2);

        set.raw.assert_pruned();

        assert!(set.remove(&2).is_some());
        set.raw.assert_pruned();

        assert!(set.remove(&1).is_some());

        set.raw.assert_pruned();
        assert!(set.is_empty());
    }

    #[test]
    fn collision_remove_one_left_with_str() {
        let mut set = ConSet::with_hasher(NoHasher);
        set.insert("hello");
        set.insert("world");

        set.raw.assert_pruned();

        assert!(set.remove("world").is_some());
        set.raw.assert_pruned();

        assert!(set.remove("hello").is_some());

        set.raw.assert_pruned();
        assert!(set.is_empty());
    }

    #[test]
    fn remove_par() {
        let mut set = ConSet::new();
        for i in 0..TEST_THREADS * TEST_BATCH {
            set.insert(i);
        }

        thread::scope(|s| {
            for t in 0..TEST_THREADS {
                let set = &set;
                s.spawn(move |_| {
                    for i in 0..TEST_BATCH {
                        let num = t * TEST_BATCH + i;
                        let val = set.remove(&num).unwrap();
                        assert_eq!(num, val);
                        assert_eq!(num, val);
                    }
                });
            }
        })
        .unwrap();

        set.raw.assert_pruned();
        assert!(set.is_empty());
    }

    fn iter_test_inner<S: BuildHasher>(set: ConSet<usize, S>) {
        for i in 0..TEST_BATCH_SMALL {
            assert!(set.insert(i).is_none());
        }

        let mut extracted = set.iter().collect::<Vec<_>>();

        extracted.sort();
        let expected = (0..TEST_BATCH_SMALL).into_iter().collect::<Vec<_>>();
        assert_eq!(expected, extracted);
    }

    #[test]
    fn iter() {
        let set = ConSet::new();
        iter_test_inner(set);
    }

    #[test]
    fn iter_collision() {
        let set = ConSet::with_hasher(NoHasher);
        iter_test_inner(set);
    }

    #[test]
    fn collect() {
        let set = (0..TEST_BATCH_SMALL).into_iter().collect::<ConSet<_>>();

        let mut extracted = set.iter().collect::<Vec<_>>();
        extracted.sort();
        let expected = (0..TEST_BATCH_SMALL).into_iter().collect::<Vec<_>>();
        assert_eq!(expected, extracted);
    }

    #[test]
    fn par_extend() {
        let set = ConSet::new();

        thread::scope(|s| {
            for t in 0..TEST_THREADS {
                let mut set = &set;
                s.spawn(move |_| {
                    let start = t * TEST_BATCH_SMALL;
                    let iter = (start..start + TEST_BATCH_SMALL).into_iter();
                    set.extend(iter);
                });
            }
        })
        .unwrap();

        let mut extracted = set.iter().collect::<Vec<_>>();

        extracted.sort();
        let expected = (0..TEST_THREADS * TEST_BATCH_SMALL)
            .into_iter()
            .collect::<Vec<_>>();

        assert_eq!(expected, extracted);
    }
}
