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
