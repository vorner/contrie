//! The [`ConMap`][crate::ConMap] type and its helpers.

use std::borrow::Borrow;
use std::collections::hash_map::RandomState;
use std::fmt::{Debug, Formatter, Result as FmtResult};
use std::hash::{BuildHasher, Hash};
use std::iter::FromIterator;
use std::marker::PhantomData;
use std::sync::Arc;

#[cfg(feature = "parallel")]
use rayon::iter::{FromParallelIterator, IntoParallelIterator, ParallelExtend, ParallelIterator};

use crate::existing_or_new::ExistingOrNew;
use crate::raw::config::Config;
use crate::raw::{self, Raw};

// :-( It would be nice if we could provide deref to (K, V). But that is incompatible with unsized
// values.
/// An element stored inside the [`ConMap`].
///
/// Or, more precisely, the [`Arc`] handles to these are stored in there.
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct Element<K, V: ?Sized> {
    key: K,
    value: V,
}

impl<K, V> Element<K, V> {
    /// Creates a new element with given key and value.
    pub fn new(key: K, value: V) -> Self {
        Self { key, value }
    }
}

impl<K, V: ?Sized> Element<K, V> {
    /// Provides access to the key.
    pub fn key(&self) -> &K {
        &self.key
    }

    /// Provides access to the value.
    pub fn value(&self) -> &V {
        &self.value
    }
}

struct MapPayload<K, V: ?Sized>(Arc<Element<K, V>>);

impl<K, V: ?Sized> Clone for MapPayload<K, V> {
    fn clone(&self) -> Self {
        MapPayload(Arc::clone(&self.0))
    }
}

impl<K, V: ?Sized> Borrow<K> for MapPayload<K, V> {
    fn borrow(&self) -> &K {
        self.0.key()
    }
}

struct MapConfig<K, V: ?Sized>(PhantomData<(K, V)>);

impl<K, V> Config for MapConfig<K, V>
where
    V: ?Sized + 'static,
    K: Hash + Eq + 'static,
{
    type Payload = MapPayload<K, V>;
    type Key = K;
}

/// The iterator of the [`ConMap`].
///
/// See the [`iter`][ConMap::iter] method for details.
pub struct Iter<'a, K, V, S>
where
    // TODO: It would be great if the bounds wouldn't have to be on the struct, only on the impls
    K: Hash + Eq + 'static,
    V: ?Sized + 'static,
{
    inner: raw::iterator::Iter<'a, MapConfig<K, V>, S>,
}

impl<'a, K, V, S> Iterator for Iter<'a, K, V, S>
where
    K: Hash + Eq + 'static,
    V: ?Sized + 'static,
{
    type Item = Arc<Element<K, V>>;
    fn next(&mut self) -> Option<Arc<Element<K, V>>> {
        self.inner.next().map(|p| Arc::clone(&p.0))
    }
}

// TODO: Bunch of derives? Which ones? And which one do we need to implement?
/// A concurrent map.
///
/// This flavour stores the data as [`Arc<Element<K, V>>`][Element]. This allows returning handles
/// to the held values cheaply even if the data is larger or impossible to clone. This has several
/// consequences:
///
/// * It is sometimes less convenient to use.
/// * It allows the values to be `?Sized` ‒ you can store trait objects or slices as the values
///   (not the keys).
/// * Entries can be shared between multiple maps.
/// * Cloning of the map doesn't clone the data, it will point to the same objects.
/// * There's another indirection in play.
///
/// Iteration returns (cloned) handles to the elements. The [`FromIterator`] and [`Extend`] traits
/// accept both tuples and element handles. Furthermore, the [`Extend`] is also implemented for
/// shared references (to allow extending the same map concurrently from multiple threads).
///
/// TODO: Support for rayon iterators/extend.
///
/// If this is not suitable, the `CloneConMap` can be used instead (TODO: Implement it).
///
/// # Examples
///
/// ```rust
/// use contrie::ConMap;
/// use crossbeam_utils::thread;
///
/// let map = ConMap::new();
///
/// thread::scope(|s| {
///     s.spawn(|_| {
///         map.insert("hello", 1);
///     });
///     s.spawn(|_| {
///         map.insert("world", 2);
///     });
/// }).unwrap();
/// assert_eq!(1, *map.get("hello").unwrap().value());
/// assert_eq!(2, *map.get("world").unwrap().value());
/// ```
///
/// ```rust
/// use std::sync::Arc;
/// use contrie::map::{ConMap, Element};
/// let map_1: ConMap<usize, [usize]> = ConMap::new();
///
/// map_1.insert_element(Arc::new(Element::new(42, [1, 2, 3])));
/// map_1.insert_element(Arc::new(Element::new(43, [1, 2, 3, 4])));
///
/// assert_eq!(3, map_1.get(&42).unwrap().value().len());
///
/// let map_2 = ConMap::new();
/// map_2.insert_element(map_1.get(&43).unwrap());
/// ```
pub struct ConMap<K, V, S = RandomState>
where
    // TODO: It would be great if the bounds wouldn't have to be on the struct, only on the impls
    K: Hash + Eq + 'static,
    V: ?Sized + 'static,
{
    raw: Raw<MapConfig<K, V>, S>,
}

impl<K, V> ConMap<K, V>
where
    K: Hash + Eq + 'static,
    V: ?Sized + 'static,
{
    /// Creates a new empty map.
    pub fn new() -> Self {
        Self::with_hasher(RandomState::default())
    }
}

// TODO: Once we have the unsized locals, this should be possible to move into the V: ?Sized block
impl<K, V, S> ConMap<K, V, S>
where
    K: Hash + Eq + 'static,
    V: 'static,
    S: BuildHasher,
{
    /// Inserts a new element.
    ///
    /// Any previous element with the same key is replaced and returned.
    pub fn insert(&self, key: K, value: V) -> Option<Arc<Element<K, V>>> {
        self.insert_element(Arc::new(Element::new(key, value)))
    }

    /// Looks up or inserts an element.
    ///
    /// It looks up an element. If it isn't present, the provided one is inserted instead. Either
    /// way, an element is returned.
    pub fn get_or_insert(&self, key: K, value: V) -> ExistingOrNew<Arc<Element<K, V>>> {
        self.get_or_insert_with(key, || value)
    }

    /// Looks up or inserts a newly created element.
    ///
    /// It looks up an element. If it isn't present, the provided closure is used to create a new
    /// one insert it. Either way, an element is returned.
    ///
    /// # Quirks
    ///
    /// Due to races in case of concurrent accesses, the closure may be called even if the value is
    /// not subsequently inserted and an existing element is returned. This should be relatively
    /// rare (another thread must insert the new element between this method observes an empty slot
    /// and manages to insert the new element).
    pub fn get_or_insert_with<F>(&self, key: K, create: F) -> ExistingOrNew<Arc<Element<K, V>>>
    where
        F: FnOnce() -> V,
    {
        self.get_or_insert_with_element(key, |key| {
            let value = create();
            Arc::new(Element::new(key, value))
        })
    }

    /// Looks up or inserts a default value of an element.
    ///
    /// This is like [get_or_insert_with][ConMap::get_or_insert_with], but a default value is used
    /// instead of manually providing a closure.
    pub fn get_or_insert_default(&self, key: K) -> ExistingOrNew<Arc<Element<K, V>>>
    where
        V: Default,
    {
        self.get_or_insert_with(key, V::default)
    }
}

impl<K, V, S> ConMap<K, V, S>
where
    K: Hash + Eq,
    V: ?Sized,
    S: BuildHasher,
{
    /// Creates a new empty map, but with the provided hasher implementation.
    pub fn with_hasher(hasher: S) -> Self {
        Self {
            raw: Raw::with_hasher(hasher),
        }
    }

    /// Inserts a new element.
    ///
    /// This acts the same as [insert][ConMap::insert], but takes the already created element. It
    /// can be used when:
    ///
    /// * `V: ?Sized`.
    /// * You want to insert the same element into multiple maps.
    pub fn insert_element(&self, element: Arc<Element<K, V>>) -> Option<Arc<Element<K, V>>> {
        let pin = crossbeam_epoch::pin();
        self.raw
            .insert(MapPayload(element), &pin)
            .map(|p| Arc::clone(&p.0))
    }

    /// Looks up or inserts a new element.
    ///
    /// This is the same as [get_or_insert_with][ConMap::get_or_insert_with], but the closure
    /// returns a pre-created element. This can be used when:
    ///
    /// * `V: ?Sized`.
    /// * You want to insert the same element into multiple maps.
    pub fn get_or_insert_with_element<F>(
        &self,
        key: K,
        create: F,
    ) -> ExistingOrNew<Arc<Element<K, V>>>
    where
        F: FnOnce(K) -> Arc<Element<K, V>>,
    {
        let pin = crossbeam_epoch::pin();
        self.raw
            .get_or_insert_with(key, |key| MapPayload(create(key)), &pin)
            .map(|payload| Arc::clone(&payload.0))
    }

    /// Looks up an element.
    pub fn get<Q>(&self, key: &Q) -> Option<Arc<Element<K, V>>>
    where
        Q: ?Sized + Eq + Hash,
        K: Borrow<Q>,
    {
        let pin = crossbeam_epoch::pin();
        self.raw.get(key, &pin).map(|r| Arc::clone(&r.0))
    }

    /// Removes an element identified by the given key, returning it.
    pub fn remove<Q>(&self, key: &Q) -> Option<Arc<Element<K, V>>>
    where
        Q: ?Sized + Eq + Hash,
        K: Borrow<Q>,
    {
        let pin = crossbeam_epoch::pin();
        self.raw.remove(key, &pin).map(|r| Arc::clone(&r.0))
    }
}

impl<K, V, S> ConMap<K, V, S>
where
    K: Hash + Eq,
    V: ?Sized,
{
    /// Checks if the map is currently empty.
    ///
    /// Note that due to the nature of concurrent map, this is inherently racy ‒ another thread may
    /// add or remove elements between you call this method and act based on the result.
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

impl<K, V> Default for ConMap<K, V>
where
    K: Hash + Eq,
    V: ?Sized,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<K, V, S> Debug for ConMap<K, V, S>
where
    K: Debug + Hash + Eq,
    V: Debug + ?Sized,
{
    fn fmt(&self, fmt: &mut Formatter) -> FmtResult {
        let mut d = fmt.debug_map();
        // TODO: As we return Arcs, it seem we can't use the iterator approach with .map :-(
        // This might hint to need for better iteration API?
        for n in self {
            // Hack: As of 1.37.0 the parameters to entry need to be &Sized. By using a double-ref,
            // we satisfy the compiler's pickiness in that regard.
            let val: &&V = &n.value();
            d.entry(n.key() as &dyn Debug, val);
        }
        d.finish()
    }
}

impl<K, V, S> Clone for ConMap<K, V, S>
where
    K: Hash + Eq,
    V: ?Sized,
    S: Clone + BuildHasher,
{
    fn clone(&self) -> Self {
        let builder = self.raw.hash_builder().clone();
        let mut new = Self::with_hasher(builder);
        new.extend(self);
        new
    }
}

impl<'a, K, V, S> IntoIterator for &'a ConMap<K, V, S>
where
    K: Hash + Eq,
    V: ?Sized,
{
    type Item = Arc<Element<K, V>>;
    type IntoIter = Iter<'a, K, V, S>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, K, V, S> Extend<Arc<Element<K, V>>> for &'a ConMap<K, V, S>
where
    K: Hash + Eq,
    V: ?Sized,
    S: BuildHasher,
{
    fn extend<T>(&mut self, iter: T)
    where
        T: IntoIterator<Item = Arc<Element<K, V>>>,
    {
        for n in iter {
            self.insert_element(n);
        }
    }
}

impl<'a, K, V, S> Extend<(K, V)> for &'a ConMap<K, V, S>
where
    K: Hash + Eq,
    S: BuildHasher,
{
    fn extend<T>(&mut self, iter: T)
    where
        T: IntoIterator<Item = (K, V)>,
    {
        self.extend(iter.into_iter().map(|(k, v)| Arc::new(Element::new(k, v))));
    }
}

impl<K, V, S> Extend<Arc<Element<K, V>>> for ConMap<K, V, S>
where
    K: Hash + Eq,
    V: ?Sized,
    S: BuildHasher,
{
    fn extend<T>(&mut self, iter: T)
    where
        T: IntoIterator<Item = Arc<Element<K, V>>>,
    {
        let mut me: &ConMap<_, _, _> = self;
        me.extend(iter);
    }
}

impl<K, V, S> Extend<(K, V)> for ConMap<K, V, S>
where
    K: Hash + Eq,
    S: BuildHasher,
{
    fn extend<T>(&mut self, iter: T)
    where
        T: IntoIterator<Item = (K, V)>,
    {
        let mut me: &ConMap<_, _, _> = self;
        me.extend(iter);
    }
}

impl<K, V> FromIterator<Arc<Element<K, V>>> for ConMap<K, V>
where
    K: Hash + Eq,
    V: ?Sized,
{
    fn from_iter<T>(iter: T) -> Self
    where
        T: IntoIterator<Item = Arc<Element<K, V>>>,
    {
        let mut me = ConMap::new();
        me.extend(iter);
        me
    }
}

impl<K, V> FromIterator<(K, V)> for ConMap<K, V>
where
    K: Hash + Eq,
{
    fn from_iter<T>(iter: T) -> Self
    where
        T: IntoIterator<Item = (K, V)>,
    {
        let mut me = ConMap::new();
        me.extend(iter);
        me
    }
}

#[cfg(feature = "parallel")]
impl<'a, K, V, S> ParallelExtend<Arc<Element<K, V>>> for &'a ConMap<K, V, S>
where
    K: Hash + Eq + Send + Sync,
    V: ?Sized + Send + Sync,
    S: BuildHasher + Sync,
{
    fn par_extend<T>(&mut self, par_iter: T)
    where
        T: IntoParallelIterator<Item = Arc<Element<K, V>>>,
    {
        par_iter.into_par_iter().for_each(|n| {
            self.insert_element(n);
        });
    }
}

#[cfg(feature = "parallel")]
impl<K, V, S> ParallelExtend<(K, V)> for ConMap<K, V, S>
where
    K: Hash + Eq + Send + Sync,
    S: BuildHasher + Sync,
    V: Send + Sync,
{
    fn par_extend<T>(&mut self, par_iter: T)
    where
        T: IntoParallelIterator<Item = (K, V)>,
    {
        self.par_extend(
            par_iter
                .into_par_iter()
                .map(|(k, v)| Arc::new(Element::new(k, v))),
        );
    }
}

#[cfg(feature = "parallel")]
impl<K, V, S> ParallelExtend<Arc<Element<K, V>>> for ConMap<K, V, S>
where
    K: Hash + Eq + Send + Sync,
    V: ?Sized + Send + Sync,
    S: BuildHasher + Sync,
{
    fn par_extend<T>(&mut self, par_iter: T)
    where
        T: IntoParallelIterator<Item = Arc<Element<K, V>>>,
    {
        let mut me: &ConMap<_, _, _> = self;
        me.par_extend(par_iter);
    }
}

#[cfg(feature = "parallel")]
impl<'a, K, V, S> ParallelExtend<(K, V)> for &'a ConMap<K, V, S>
where
    K: Hash + Eq + Send + Sync,
    S: BuildHasher + Sync,
    V: Send + Sync,
{
    fn par_extend<T>(&mut self, par_iter: T)
    where
        T: IntoParallelIterator<Item = (K, V)>,
    {
        self.par_extend(
            par_iter
                .into_par_iter()
                .map(|(k, v)| Arc::new(Element::new(k, v))),
        );
    }
}

#[cfg(feature = "parallel")]
impl<K, V> FromParallelIterator<Arc<Element<K, V>>> for ConMap<K, V>
where
    K: Hash + Eq + Send + Sync,
    V: ?Sized + Send + Sync,
{
    fn from_par_iter<T>(par_iter: T) -> Self
    where
        T: IntoParallelIterator<Item = Arc<Element<K, V>>>,
    {
        let mut me = ConMap::new();
        me.par_extend(par_iter);
        me
    }
}

#[cfg(feature = "parallel")]
impl<K, V> FromParallelIterator<(K, V)> for ConMap<K, V>
where
    K: Hash + Eq + Send + Sync,
    V: Send + Sync,
{
    fn from_par_iter<T>(par_iter: T) -> Self
    where
        T: IntoParallelIterator<Item = (K, V)>,
    {
        let mut me = ConMap::new();
        me.par_extend(par_iter);
        me
    }
}

#[cfg(test)]
mod tests {
    use crossbeam_utils::thread;

    #[cfg(feature = "parallel")]
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
        let map: ConMap<String, usize> = ConMap::new();
        drop(map);
    }

    #[test]
    fn lookup_empty() {
        let map: ConMap<String, usize> = ConMap::new();
        assert!(map.get("hello").is_none());
    }

    #[test]
    fn insert_lookup() {
        let map = ConMap::new();
        assert!(map.insert("hello", "world").is_none());
        assert!(map.get("world").is_none());
        let found = map.get("hello").unwrap();
        assert_eq!(Element::new("hello", "world"), *found);
    }

    #[test]
    fn insert_overwrite_lookup() {
        let map = ConMap::new();
        assert!(map.insert("hello", "world").is_none());
        let old = map.insert("hello", "universe").unwrap();
        assert_eq!(Element::new("hello", "world"), *old);
        let found = map.get("hello").unwrap();
        assert_eq!(Element::new("hello", "universe"), *found);
    }

    // Insert a lot of things, to make sure we have multiple levels.
    #[test]
    fn insert_many() {
        let map = ConMap::new();
        for i in 0..TEST_BATCH * LEVEL_CELLS {
            assert!(map.insert(i, i).is_none());
        }

        for i in 0..TEST_BATCH * LEVEL_CELLS {
            assert_eq!(i, *map.get(&i).unwrap().value());
        }
    }

    #[test]
    fn par_insert_many() {
        for _ in 0..TEST_REP {
            let map: ConMap<usize, usize> = ConMap::new();
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
                assert_eq!(*map.get(&i).unwrap().value(), i);
            }
        }
    }

    #[test]
    fn par_get_many() {
        for _ in 0..TEST_REP {
            let map = ConMap::new();
            for i in 0..TEST_BATCH * TEST_THREADS {
                assert!(map.insert(i, i).is_none());
            }
            thread::scope(|s| {
                for t in 0..TEST_THREADS {
                    let map = &map;
                    s.spawn(move |_| {
                        for i in 0..TEST_BATCH {
                            let num = t * TEST_BATCH + i;
                            assert_eq!(*map.get(&num).unwrap().value(), num);
                        }
                    });
                }
            })
            .unwrap();
        }
    }

    #[test]
    fn collisions() {
        let map = ConMap::with_hasher(NoHasher);
        // While their hash is the same under the hasher, they don't kick each other out.
        for i in 0..TEST_BATCH_SMALL {
            assert!(map.insert(i, i).is_none());
        }
        // And all are present.
        for i in 0..TEST_BATCH_SMALL {
            assert_eq!(i, *map.get(&i).unwrap().value());
        }
        // But reusing the key kicks the other one out.
        for i in 0..TEST_BATCH_SMALL {
            assert_eq!(i, *map.insert(i, i + 1).unwrap().value());
            assert_eq!(i + 1, *map.get(&i).unwrap().value());
        }
    }

    #[test]
    fn get_or_insert_empty() {
        let map = ConMap::new();
        let val = map.get_or_insert("hello", 42);
        assert_eq!(42, *val.value());
        assert_eq!("hello", *val.key());
        assert!(val.is_new());
    }

    #[test]
    fn get_or_insert_existing() {
        let map = ConMap::new();
        assert!(map.insert("hello", 42).is_none());
        let val = map.get_or_insert("hello", 0);
        // We still have the original
        assert_eq!(42, *val.value());
        assert_eq!("hello", *val.key());
        assert!(!val.is_new());
    }

    fn get_or_insert_many_inner<H: BuildHasher>(map: ConMap<usize, usize, H>, len: usize) {
        for i in 0..len {
            let val = map.get_or_insert(i, i);
            assert_eq!(i, *val.key());
            assert_eq!(i, *val.value());
            assert!(val.is_new());
        }

        for i in 0..len {
            let val = map.get_or_insert(i, 0);
            assert_eq!(i, *val.key());
            assert_eq!(i, *val.value());
            assert!(!val.is_new());
        }
    }

    #[test]
    fn get_or_insert_many() {
        get_or_insert_many_inner(ConMap::new(), TEST_BATCH);
    }

    #[test]
    fn get_or_insert_collision() {
        get_or_insert_many_inner(ConMap::with_hasher(NoHasher), TEST_BATCH_SMALL);
    }

    #[test]
    fn simple_remove() {
        let map = ConMap::new();
        assert!(map.remove(&42).is_none());
        assert!(map.insert(42, "hello").is_none());
        assert_eq!("hello", *map.get(&42).unwrap().value());
        assert_eq!("hello", *map.remove(&42).unwrap().value());
        assert!(map.get(&42).is_none());
        assert!(map.is_empty());
        assert!(map.remove(&42).is_none());
        assert!(map.is_empty());
    }

    fn remove_many_inner<H: BuildHasher>(mut map: ConMap<usize, usize, H>, len: usize) {
        for i in 0..len {
            assert!(map.insert(i, i).is_none());
        }
        for i in 0..len {
            assert_eq!(i, *map.get(&i).unwrap().value());
            assert_eq!(i, *map.remove(&i).unwrap().value());
            assert!(map.get(&i).is_none());
            map.raw.assert_pruned();
        }

        assert!(map.is_empty());
    }

    #[test]
    fn remove_many() {
        remove_many_inner(ConMap::new(), TEST_BATCH);
    }

    #[test]
    fn remove_many_collision() {
        remove_many_inner(ConMap::with_hasher(NoHasher), TEST_BATCH_SMALL);
    }

    #[test]
    fn collision_remove_one_left() {
        let mut map = ConMap::with_hasher(NoHasher);
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
        let mut map = ConMap::new();
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
                        assert_eq!(num, *val.value());
                        assert_eq!(num, *val.key());
                    }
                });
            }
        })
        .unwrap();

        map.raw.assert_pruned();
        assert!(map.is_empty());
    }

    #[test]
    fn unsized_values() {
        let map: ConMap<usize, [usize]> = ConMap::new();
        assert!(map
            .insert_element(Arc::new(Element::new(42, [1, 2, 3])))
            .is_none());
        let found = map.get(&42).unwrap();
        assert_eq!(&[1, 2, 3], found.value());
        let inserted = map.get_or_insert_with_element(0, |k| {
            assert_eq!(0, k);
            Arc::new(Element::new(k, []))
        });
        assert_eq!(0, *inserted.key());
        assert!(inserted.value().is_empty());
        assert!(inserted.is_new());
        let removed = map.remove(&0).unwrap();
        assert_eq!(inserted.into_inner(), removed);
    }

    fn iter_test_inner<S: BuildHasher>(map: ConMap<usize, usize, S>) {
        for i in 0..TEST_BATCH_SMALL {
            assert!(map.insert(i, i).is_none());
        }

        let mut extracted = map.iter().map(|v| *v.value()).collect::<Vec<_>>();
        extracted.sort();
        let expected = (0..TEST_BATCH_SMALL).collect::<Vec<_>>();
        assert_eq!(expected, extracted);
    }

    #[test]
    fn iter() {
        let map = ConMap::new();
        iter_test_inner(map);
    }

    #[test]
    fn iter_collision() {
        let map = ConMap::with_hasher(NoHasher);
        iter_test_inner(map);
    }

    #[test]
    fn collect() {
        let map = (0..TEST_BATCH_SMALL)
            .map(|i| (i, i))
            .collect::<ConMap<_, _>>();

        let mut extracted = map
            .iter()
            .map(|n| {
                assert_eq!(n.key(), n.value());
                *n.value()
            })
            .collect::<Vec<_>>();

        extracted.sort();
        let expected = (0..TEST_BATCH_SMALL).collect::<Vec<_>>();
        assert_eq!(expected, extracted);
    }

    #[test]
    fn par_extend() {
        let map = ConMap::new();
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
                assert_eq!(n.key(), n.value());
                *n.value()
            })
            .collect::<Vec<_>>();

        extracted.sort();
        let expected = (0..TEST_THREADS * TEST_BATCH_SMALL).collect::<Vec<_>>();
        assert_eq!(expected, extracted);
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn rayon_extend() {
        let mut map = ConMap::new();
        map.par_extend((0..TEST_BATCH_SMALL).into_par_iter().map(|i| (i, i)));

        let mut extracted = map
            .iter()
            .map(|n| {
                assert_eq!(n.key(), n.value());
                *n.value()
            })
            .collect::<Vec<_>>();
        extracted.sort();

        let expected = (0..TEST_BATCH_SMALL).collect::<Vec<_>>();
        assert_eq!(expected, extracted);
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn rayon_from_par_iter() {
        let map = ConMap::from_par_iter((0..TEST_BATCH_SMALL).into_par_iter().map(|i| (i, i)));
        let mut extracted = map
            .iter()
            .map(|n| {
                assert_eq!(n.key(), n.value());
                *n.value()
            })
            .collect::<Vec<_>>();
        extracted.sort();

        let expected = (0..TEST_BATCH_SMALL).collect::<Vec<_>>();
        assert_eq!(expected, extracted);
    }
}
