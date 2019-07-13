//! The [`CloneConMap`][crate::CloneConMap] type and its helpers.

use std::borrow::Borrow;
use std::collections::hash_map::RandomState;
use std::fmt::{Debug, Formatter, Result as FmtResult};
use std::hash::{BuildHasher, Hash};
use std::iter::FromIterator;
use std::marker::PhantomData;

#[cfg(feature = "rayon")]
use rayon::iter::{FromParallelIterator, IntoParallelIterator, ParallelExtend, ParallelIterator};

use crate::existing_or_new::ExistingOrNew;
use crate::raw::config::Config;
use crate::raw::{self, Raw};

#[derive(Clone)]
struct CloneMapPayload<K, V>((K, V));

impl<K, V> Borrow<K> for CloneMapPayload<K, V> {
    fn borrow(&self) -> &K {
        let &(ref k, _) = &self.0;
        k
    }
}

struct CloneMapConfig<K, V>(PhantomData<(K, V)>);

impl<K, V> Config for CloneMapConfig<K, V>
where
    K: Clone + Hash + Eq + 'static,
    V: Clone + 'static,
{
    type Payload = CloneMapPayload<K, V>;
    type Key = K;
}

/// The iterator of the [`CloneConMap`].
///
/// See the [`iter`][CloneConMap::iter] method for details.
pub struct Iter<'a, K, V, S>
where
    K: Clone + Hash + Eq + 'static,
    V: Clone + 'static,
{
    inner: raw::iterator::Iter<'a, CloneMapConfig<K, V>, S>,
}

impl<'a, K, V, S> Iterator for Iter<'a, K, V, S>
where
    K: Clone + Hash + Eq + 'static,
    V: Clone + 'static,
{
    type Item = (K, V);
    fn next(&mut self) -> Option<(K, V)> {
        self.inner.next().map(|p| (p.0).clone())
    }
}

/// A concurrent map that clones its elements.
///
/// This flavour stores the data as `(K, V)` tuples; it clones
/// independently both the elements (to be intended as the key and the
/// value) of the tuple. The return values of its functions are clones
/// of the stored values. This makes this data structure suitable for
/// types cheap to clone.
///
/// Iteration returns cloned copies of its elements. The
/// [`FromIterator`] and [`Extend`] traits accept tuples as arguments.
/// Furthermore, the [`Extend`] is also implemented for shared
/// references (to allow extending the same map concurrently from
/// multiple threads).
///
/// # Examples
///
/// ```rust
/// use contrie::CloneConMap;
/// use crossbeam_utils::thread;
///
/// let map = CloneConMap::new();
///
/// thread::scope(|s| {
///     s.spawn(|_| {
///         map.insert("hello", 1);
///     });
///     s.spawn(|_| {
///         map.insert("world", 2);
///     });
/// }).unwrap();
/// assert_eq!(1, map.get("hello").unwrap().1);
/// assert_eq!(2, map.get("world").unwrap().1);
/// ```
///
/// ```rust
/// use contrie::clonemap::{CloneConMap};
///
/// let map_1: CloneConMap<usize, Vec<usize>> = CloneConMap::new();
///
/// map_1.insert(42, vec![1, 2, 3]);
/// map_1.insert(43, vec![1, 2, 3, 4]);
///
/// assert_eq!(3, map_1.get(&42).unwrap().1.len());
/// assert_eq!(4, map_1.get(&43).unwrap().1.len());
/// assert_eq!(None, map_1.get(&44));
///
/// let map_2 = CloneConMap::new();
/// map_2.insert(44, map_1.get(&43).unwrap().1);
/// assert_eq!(4, map_2.get(&44).unwrap().1.len());
/// ```
pub struct CloneConMap<K, V, S = RandomState>
where
    K: Clone + Hash + Eq + 'static,
    V: Clone + 'static,
{
    raw: Raw<CloneMapConfig<K, V>, S>,
}

impl<K, V> CloneConMap<K, V>
where
    K: Clone + Hash + Eq + 'static,
    V: Clone + 'static,
{
    /// Creates a new empty map.
    pub fn new() -> Self {
        Self::with_hasher(RandomState::default())
    }
}

impl<K, V, S> CloneConMap<K, V, S>
where
    K: Clone + Hash + Eq + 'static,
    V: Clone + 'static,
    S: BuildHasher,
{
    /// Inserts a new element as a tuple `(key, value)`.
    ///
    /// Any previous element with the same key is replaced and returned.
    pub fn insert(&self, key: K, value: V) -> Option<(K, V)> {
        let pin = crossbeam_epoch::pin();
        self.raw
            .insert(CloneMapPayload((key, value)), &pin)
            .map(|p| p.0.clone())
    }

    /// Looks up or inserts an element as a tuple `(key, value)`.
    ///
    /// It looks up an element. If it isn't present, the provided one is
    /// inserted instead. Either way, an element is returned.
    pub fn get_or_insert(&self, key: K, value: V) -> ExistingOrNew<(K, V)> {
        self.get_or_insert_with(key, || value)
    }

    /// Looks up or inserts a newly created element.
    ///
    /// It looks up an element. If it isn't present, the provided
    /// closure is used to create a new one insert it. Either way, an
    /// element is returned.
    ///
    /// # Quirks
    ///
    /// Due to races in case of concurrent accesses, the closure may be
    /// called even if the value is not subsequently inserted and an
    /// existing element is returned. This should be relatively rare
    /// (another thread must insert the new element between this method
    /// observes an empty slot and manages to insert the new element).
    pub fn get_or_insert_with<F>(&self, key: K, create: F) -> ExistingOrNew<(K, V)>
    where
        F: FnOnce() -> V,
    {
        let pin = crossbeam_epoch::pin();

        self.raw
            .get_or_insert_with(
                key,
                |key| {
                    let value: V = create();
                    CloneMapPayload((key, value))
                },
                &pin,
            )
            .map(|payload| (payload.0).clone())
    }

    /// Looks up or inserts a default value of an element.
    ///
    /// This is like [get_or_insert_with][CloneConMap::get_or_insert_with],
    /// but a default value is used instead of manually providing a
    /// closure.
    pub fn get_or_insert_default(&self, key: K) -> ExistingOrNew<(K, V)>
    where
        V: Default,
    {
        self.get_or_insert_with(key, V::default)
    }
}

impl<K, V, S> CloneConMap<K, V, S>
where
    K: Clone + Hash + Eq,
    V: Clone,
    S: BuildHasher,
{
    /// Creates a new empty map, but with the provided hasher implementation.
    pub fn with_hasher(hasher: S) -> Self {
        Self {
            raw: Raw::with_hasher(hasher),
        }
    }

    /// Looks up an element.
    pub fn get<Q>(&self, key: &Q) -> Option<(K, V)>
    where
        Q: ?Sized + Eq + Hash,
        K: Borrow<Q>,
    {
        let pin = crossbeam_epoch::pin();
        self.raw.get(key, &pin).map(|r| (r.0).clone())
    }

    /// Removes an element identified by the given key, returning it.
    pub fn remove<Q>(&self, key: &Q) -> Option<(K, V)>
    where
        Q: ?Sized + Eq + Hash,
        K: Borrow<Q>,
    {
        let pin = crossbeam_epoch::pin();
        self.raw.remove(key, &pin).map(|r| (r.0).clone())
    }
}

impl<K, V, S> CloneConMap<K, V, S>
where
    K: Clone + Hash + Eq,
    V: Clone,
{
    /// Checks if the map is currently empty.
    ///
    /// Note that due to the nature of concurrent map, this is
    /// inherently racy ‒ another thread may add or remove elements
    /// between you call this method and act based on the result.
    pub fn is_empty(&self) -> bool {
        self.raw.is_empty()
    }

    /// Returns an iterator through the elements of the map.
    pub fn iter(&self) -> Iter<K, V, S> {
        Iter {
            inner: raw::iterator::Iter::new(&self.raw),
        }
    }
}

impl<K, V> Default for CloneConMap<K, V>
where
    K: Clone + Hash + Eq,
    V: Clone,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<K, V, S> Debug for CloneConMap<K, V, S>
where
    K: Debug + Clone + Hash + Eq,
    V: Debug + Clone,
{
    fn fmt(&self, fmt: &mut Formatter) -> FmtResult {
        fmt.debug_map().entries(self.iter()).finish()
    }
}

impl<K, V, S> Clone for CloneConMap<K, V, S>
where
    K: Clone + Hash + Eq,
    V: Clone,
    S: Clone + BuildHasher,
{
    fn clone(&self) -> Self {
        let builder = self.raw.hash_builder().clone();
        let mut new = Self::with_hasher(builder);
        new.extend(self);
        new
    }
}

impl<'a, K, V, S> IntoIterator for &'a CloneConMap<K, V, S>
where
    K: Clone + Hash + Eq,
    V: Clone,
{
    type Item = (K, V);
    type IntoIter = Iter<'a, K, V, S>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, K, V, S> Extend<(K, V)> for &'a CloneConMap<K, V, S>
where
    K: Clone + Hash + Eq,
    V: Clone,
    S: BuildHasher,
{
    fn extend<T>(&mut self, iter: T)
    where
        T: IntoIterator<Item = (K, V)>,
    {
        for (k, v) in iter {
            self.insert(k, v);
        }
    }
}

impl<K, V, S> Extend<(K, V)> for CloneConMap<K, V, S>
where
    K: Clone + Hash + Eq,
    V: Clone,
    S: BuildHasher,
{
    fn extend<T>(&mut self, iter: T)
    where
        T: IntoIterator<Item = (K, V)>,
    {
        let mut me: &CloneConMap<_, _, _> = self;
        me.extend(iter);
    }
}

impl<K, V> FromIterator<(K, V)> for CloneConMap<K, V>
where
    K: Clone + Hash + Eq,
    V: Clone,
{
    fn from_iter<T>(iter: T) -> Self
    where
        T: IntoIterator<Item = (K, V)>,
    {
        let mut me = CloneConMap::new();
        me.extend(iter);
        me
    }
}

#[cfg(feature = "rayon")]
impl<K, V, S> ParallelExtend<(K, V)> for CloneConMap<K, V, S>
where
    K: Clone + Hash + Eq + Send + Sync,
    S: BuildHasher + Sync,
    V: Clone + Send + Sync,
{
    fn par_extend<T>(&mut self, par_iter: T)
    where
        T: IntoParallelIterator<Item = (K, V)>,
    {
        par_iter.into_par_iter().for_each(|(k, v)| {
            self.insert(k.clone(), v.clone());
        });
    }
}

#[cfg(feature = "rayon")]
impl<'a, K, V, S> ParallelExtend<(K, V)> for &'a CloneConMap<K, V, S>
where
    K: Clone + Hash + Eq + Send + Sync,
    S: BuildHasher + Sync,
    V: Clone + Send + Sync,
{
    fn par_extend<T>(&mut self, par_iter: T)
    where
        T: IntoParallelIterator<Item = (K, V)>,
    {
        par_iter.into_par_iter().for_each(|(k, v)| {
            self.insert(k.clone(), v.clone());
        });
    }
}

#[cfg(feature = "rayon")]
impl<K, V> FromParallelIterator<(K, V)> for CloneConMap<K, V>
where
    K: Clone + Hash + Eq + Send + Sync,
    V: Clone + Send + Sync,
{
    fn from_par_iter<T>(par_iter: T) -> Self
    where
        T: IntoParallelIterator<Item = (K, V)>,
    {
        let mut me = CloneConMap::new();
        me.par_extend(par_iter);
        me
    }
}

#[cfg(test)]
mod tests {
    use crossbeam_utils::thread;
    use std::rc::Rc;

    #[cfg(feature = "rayon")]
    use rayon::prelude::*;

    use super::*;
    use crate::raw::tests::NoHasher;
    use crate::raw::LEVEL_CELLS;

    const TEST_THREADS: usize = 4;
    const TEST_BATCH: usize = 10000;
    const TEST_BATCH_SMALL: usize = 100;
    const TEST_REP: usize = 20;

    #[test]
    fn create_destroy() {
        let map: CloneConMap<String, usize> = CloneConMap::new();
        drop(map);
    }

    #[test]
    fn debug_formatting() {
        let map: CloneConMap<&str, &str> = CloneConMap::new();
        map.insert("hello", "world");
        assert_eq!("{\"hello\": \"world\"}".to_owned(), format!("{:?}", map));
    }

    #[test]
    fn lookup_empty() {
        let map: CloneConMap<String, usize> = CloneConMap::new();
        assert!(map.get("hello").is_none());
    }

    #[test]
    fn insert_lookup() {
        let map = CloneConMap::new();
        assert!(map.insert("hello", "world").is_none());
        assert!(map.get("world").is_none());
        let found = map.get("hello").unwrap();
        assert_eq!(("hello", "world"), found);
    }

    #[test]
    fn insert_overwrite_lookup() {
        let map = CloneConMap::new();
        assert!(map.insert("hello", "world").is_none());
        let old = map.insert("hello", "universe").unwrap();
        assert_eq!(("hello", "world"), old);
        let found = map.get("hello").unwrap();
        assert_eq!(("hello", "universe"), found);
    }

    // Insert a lot of things, to make sure we have multiple levels.
    #[test]
    fn insert_many() {
        let map = CloneConMap::new();
        for i in 0..TEST_BATCH * LEVEL_CELLS {
            assert!(map.insert(i, i).is_none());
        }

        for i in 0..TEST_BATCH * LEVEL_CELLS {
            assert_eq!(i, map.get(&i).unwrap().1);
        }
    }

    #[test]
    fn par_insert_many() {
        for _ in 0..TEST_REP {
            let map: CloneConMap<usize, usize> = CloneConMap::new();
            thread::scope(|s| {
                for t in 0..TEST_THREADS {
                    let map = &map;
                    s.spawn(move |_| {
                        for i in 0..TEST_BATCH {
                            let num = t * TEST_BATCH + i;
                            assert!(map.insert(num, num).is_none());
                        }
                    });
                }
            })
            .unwrap();

            for i in 0..TEST_BATCH * TEST_THREADS {
                assert_eq!(map.get(&i).unwrap().1, i);
            }
        }
    }

    #[test]
    fn par_get_many() {
        for _ in 0..TEST_REP {
            let map = CloneConMap::new();
            for i in 0..TEST_BATCH * TEST_THREADS {
                assert!(map.insert(i, i).is_none());
            }
            thread::scope(|s| {
                for t in 0..TEST_THREADS {
                    let map = &map;
                    s.spawn(move |_| {
                        for i in 0..TEST_BATCH {
                            let num = t * TEST_BATCH + i;
                            assert_eq!(map.get(&num).unwrap().1, num);
                        }
                    });
                }
            })
            .unwrap();
        }
    }

    #[test]
    fn collisions() {
        let map = CloneConMap::with_hasher(NoHasher);
        // While their hash is the same under the hasher, they don't kick each other out.
        for i in 0..TEST_BATCH_SMALL {
            assert!(map.insert(i, i).is_none());
        }
        // And all are present.
        for i in 0..TEST_BATCH_SMALL {
            assert_eq!(i, map.get(&i).unwrap().1);
        }
        // But reusing the key kicks the other one out.
        for i in 0..TEST_BATCH_SMALL {
            assert_eq!(i, map.insert(i, i + 1).unwrap().1);
            assert_eq!(i + 1, map.get(&i).unwrap().1);
        }
    }

    #[test]
    fn get_or_insert_empty() {
        let map = CloneConMap::new();
        let val = map.get_or_insert("hello", 42);
        assert_eq!(42, val.1);
        assert_eq!("hello", val.0);
        assert!(val.is_new());
    }

    #[test]
    fn get_or_insert_existing() {
        let map = CloneConMap::new();
        assert!(map.insert("hello", 42).is_none());
        let val = map.get_or_insert("hello", 0);
        // We still have the original
        assert_eq!(42, val.1);
        assert_eq!("hello", val.0);
        assert!(!val.is_new());
    }

    #[test]
    fn get_or_insert_existing_with_counter() {
        let map = CloneConMap::new();
        assert!(map.insert("hello", Rc::new(42)).is_none());
        let val = map.get_or_insert("hello", Rc::new(0));
        // We still have the original
        assert_eq!(&42, val.1.borrow());
        assert_eq!("hello", val.0);
        assert_eq!(2, Rc::strong_count(&val.1));
        let val = map.get_or_insert("hello", Rc::new(0));
        assert_eq!(3, Rc::strong_count(&val.1));
        assert!(!val.is_new());
    }

    fn get_or_insert_many_inner<H: BuildHasher>(map: CloneConMap<usize, usize, H>, len: usize) {
        for i in 0..len {
            let val = map.get_or_insert(i, i);
            assert_eq!(i, val.0);
            assert_eq!(i, val.1);
            assert!(val.is_new());
        }

        for i in 0..len {
            let val = map.get_or_insert(i, 0);
            assert_eq!(i, val.0);
            assert_eq!(i, val.1);
            assert!(!val.is_new());
        }
    }

    #[test]
    fn get_or_insert_many() {
        get_or_insert_many_inner(CloneConMap::new(), TEST_BATCH);
    }

    #[test]
    fn get_or_insert_collision() {
        get_or_insert_many_inner(CloneConMap::with_hasher(NoHasher), TEST_BATCH_SMALL);
    }

    #[test]
    fn simple_remove() {
        let map = CloneConMap::new();
        assert!(map.remove(&42).is_none());
        assert!(map.insert(42, "hello").is_none());
        assert_eq!("hello", map.get(&42).unwrap().1);
        assert_eq!("hello", map.remove(&42).unwrap().1);
        assert!(map.get(&42).is_none());
        assert!(map.is_empty());
        assert!(map.remove(&42).is_none());
        assert!(map.is_empty());
    }

    fn remove_many_inner<H: BuildHasher>(mut map: CloneConMap<usize, usize, H>, len: usize) {
        for i in 0..len {
            assert!(map.insert(i, i).is_none());
        }
        for i in 0..len {
            assert_eq!(i, map.get(&i).unwrap().1);
            assert_eq!(i, map.remove(&i).unwrap().1);
            assert!(map.get(&i).is_none());
            map.raw.assert_pruned();
        }

        assert!(map.is_empty());
    }

    #[test]
    fn remove_many() {
        remove_many_inner(CloneConMap::new(), TEST_BATCH);
    }

    #[test]
    fn remove_many_collision() {
        remove_many_inner(CloneConMap::with_hasher(NoHasher), TEST_BATCH_SMALL);
    }

    #[test]
    fn collision_remove_one_left() {
        let mut map = CloneConMap::with_hasher(NoHasher);
        map.insert(1, 1);
        map.insert(2, 2);

        map.raw.assert_pruned();

        assert!(map.remove(&2).is_some());
        map.raw.assert_pruned();

        assert!(map.remove(&1).is_some());

        map.raw.assert_pruned();
        assert!(map.is_empty());
    }

    #[test]
    fn remove_par() {
        let mut map = CloneConMap::new();
        for i in 0..TEST_THREADS * TEST_BATCH {
            map.insert(i, i);
        }

        thread::scope(|s| {
            for t in 0..TEST_THREADS {
                let map = &map;
                s.spawn(move |_| {
                    for i in 0..TEST_BATCH {
                        let num = t * TEST_BATCH + i;
                        let val = map.remove(&num).unwrap();
                        assert_eq!(num, val.1);
                        assert_eq!(num, val.0);
                    }
                });
            }
        })
        .unwrap();

        map.raw.assert_pruned();
        assert!(map.is_empty());
    }

    fn iter_test_inner<S: BuildHasher>(map: CloneConMap<usize, usize, S>) {
        for i in 0..TEST_BATCH_SMALL {
            assert!(map.insert(i, i).is_none());
        }

        let mut extracted = map.iter().map(|v| v.1).collect::<Vec<_>>();
        extracted.sort();
        let expected = (0..TEST_BATCH_SMALL).collect::<Vec<_>>();
        assert_eq!(expected, extracted);
    }

    #[test]
    fn iter() {
        let map = CloneConMap::new();
        iter_test_inner(map);
    }

    #[test]
    fn iter_collision() {
        let map = CloneConMap::with_hasher(NoHasher);
        iter_test_inner(map);
    }

    #[test]
    fn collect() {
        let map = (0..TEST_BATCH_SMALL)
            .map(|i| (i, i))
            .collect::<CloneConMap<_, _>>();

        let mut extracted = map
            .iter()
            .map(|n| {
                assert_eq!(n.0, n.1);
                n.1
            })
            .collect::<Vec<_>>();

        extracted.sort();
        let expected = (0..TEST_BATCH_SMALL).collect::<Vec<_>>();
        assert_eq!(expected, extracted);
    }

    #[test]
    fn par_extend() {
        let map = CloneConMap::new();
        thread::scope(|s| {
            for t in 0..TEST_THREADS {
                let mut map = &map;
                s.spawn(move |_| {
                    let start = t * TEST_BATCH_SMALL;
                    let iter = (start..start + TEST_BATCH_SMALL).map(|i| (i, i));
                    map.extend(iter);
                });
            }
        })
        .unwrap();

        let mut extracted = map
            .iter()
            .map(|n| {
                assert_eq!(n.0, n.1);
                n.1
            })
            .collect::<Vec<_>>();

        extracted.sort();
        let expected = (0..TEST_THREADS * TEST_BATCH_SMALL).collect::<Vec<_>>();
        assert_eq!(expected, extracted);
    }

    #[cfg(feature = "rayon")]
    #[test]
    fn rayon_extend() {
        let mut map = CloneConMap::new();
        map.par_extend((0..TEST_BATCH_SMALL).into_par_iter().map(|i| (i, i)));

        let mut extracted = map
            .iter()
            .map(|n| {
                assert_eq!(n.0, n.1);
                n.1
            })
            .collect::<Vec<_>>();
        extracted.sort();

        let expected = (0..TEST_BATCH_SMALL).collect::<Vec<_>>();
        assert_eq!(expected, extracted);
    }

    #[cfg(feature = "rayon")]
    #[test]
    fn rayon_from_par_iter() {
        let map = CloneConMap::from_par_iter((0..TEST_BATCH_SMALL).into_par_iter().map(|i| (i, i)));
        let mut extracted = map
            .iter()
            .map(|n| {
                assert_eq!(n.0, n.1);
                n.1
            })
            .collect::<Vec<_>>();
        extracted.sort();

        let expected = (0..TEST_BATCH_SMALL).collect::<Vec<_>>();
        assert_eq!(expected, extracted);
    }
}
