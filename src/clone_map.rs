use std::borrow::Borrow;
use std::collections::hash_map::RandomState;
use std::hash::{BuildHasher, Hash};
use std::iter::FromIterator;
use std::marker::PhantomData;

use crate::existing_or_new::ExistingOrNew;
use crate::raw::config::Config;
use crate::raw::{self, Raw};

// :-( It would be nice if we could provide deref to (K, V). But that is incompatible with unsized
// values.
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct Node<K, V> {
    key: K,
    value: V,
}

impl<K, V> Node<K, V> {
    pub fn new(key: K, value: V) -> Self {
        Self { key, value }
    }
}

impl<K, V> Node<K, V> {
    pub fn key(&self) -> &K {
        &self.key
    }

    pub fn value(&self) -> &V {
        &self.value
    }
}

impl<K, V> Borrow<K> for Node<K, V> {
    fn borrow(&self) -> &K {
        self.key()
    }
}

struct MapConfig<K, V>(PhantomData<(K, V)>);

impl<K, V> Config for MapConfig<K, V>
where
    K: Clone + Hash + Eq,
    V: Clone,
{
    type Payload = Node<K, V>;
    type Key = K;
}

pub struct Iter<'a, K, V, S>
where
    // TODO: It would be great if the bounds wouldn't have to be on the struct, only on the impls
    K: Clone + Hash + Eq,
    V: Clone,
{
    inner: raw::iterator::Iter<'a, MapConfig<K, V>, S>,
}

impl<'a, K, V, S> Iterator for Iter<'a, K, V, S>
where
    // TODO: It would be great if the bounds wouldn't have to be on the struct, only on the impls
    K: Clone + Hash + Eq,
    V: Clone,
{
    type Item = Node<K, V>;
    fn next(&mut self) -> Option<Node<K, V>> {
        self.inner.next().cloned()
    }
}

// TODO: Bunch of derives? Which ones? And which one do we need to implement?
pub struct CloneConMap<K, V, S = RandomState>
where
    // TODO: It would be great if the bounds wouldn't have to be on the struct, only on the impls
    K: Clone + Hash + Eq,
    V: Clone,
{
    raw: Raw<MapConfig<K, V>, S>,
}

impl<K, V> CloneConMap<K, V>
where
    K: Clone + Hash + Eq,
    V: Clone,
{
    pub fn new() -> Self {
        Self::with_hasher(RandomState::default())
    }
}

// TODO: Once we have the unsized locals, this should be possible to move into the V: ?Sized block
impl<K, V, S> CloneConMap<K, V, S>
where
    K: Clone + Hash + Eq,
    V: Clone,
    S: BuildHasher,
{
    pub fn with_hasher(hasher: S) -> Self {
        Self {
            raw: Raw::with_hasher(hasher),
        }
    }

    pub fn insert(&self, key: K, value: V) -> Option<Node<K, V>> {
        self.insert_node(Node::new(key, value))
    }

    pub fn get_or_insert_with<F>(&self, key: K, create: F) -> ExistingOrNew<Node<K, V>>
    where
        F: FnOnce() -> V,
    {
        self.get_or_insert_with_node(key, |key| {
            let value = create();
            Node::new(key, value)
        })
    }

    pub fn get_or_insert(&self, key: K, value: V) -> ExistingOrNew<Node<K, V>> {
        self.get_or_insert_with(key, || value)
    }

    pub fn get_or_insert_default(&self, key: K) -> ExistingOrNew<Node<K, V>>
    where
        V: Default,
    {
        self.get_or_insert_with(key, V::default)
    }

    pub fn insert_node(&self, node: Node<K, V>) -> Option<Node<K, V>> {
        let pin = crossbeam_epoch::pin();
        self.raw
            .insert(node, &pin)
            .cloned()
    }

    pub fn get_or_insert_with_node<F>(&self, key: K, create: F) -> ExistingOrNew<Node<K, V>>
    where
        F: FnOnce(K) -> Node<K, V>,
    {
        let pin = crossbeam_epoch::pin();
        self.raw
            .get_or_insert_with(key, |key| create(key), &pin)
            .map(|n| n.clone())
    }

    pub fn get<Q>(&self, key: &Q) -> Option<Node<K, V>>
    where
        Q: ?Sized + Eq + Hash,
        K: Borrow<Q>,
    {
        let pin = crossbeam_epoch::pin();
        self.raw.get(key, &pin).cloned()
    }

    pub fn remove<Q>(&self, key: &Q) -> Option<Node<K, V>>
    where
        Q: ?Sized + Eq + Hash,
        K: Borrow<Q>,
    {
        let pin = crossbeam_epoch::pin();
        self.raw.remove(key, &pin).cloned()
    }

    pub fn is_empty(&self) -> bool {
        self.raw.is_empty()
    }

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

impl<'a, K, V, S> IntoIterator for &'a CloneConMap<K, V, S>
where
    K: Clone + Hash + Eq,
    V: Clone,
    S: BuildHasher,
{
    type Item = Node<K, V>;
    type IntoIter = Iter<'a, K, V, S>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, K, V, S> Extend<Node<K, V>> for &'a CloneConMap<K, V, S>
where
    K: Clone + Hash + Eq,
    V: Clone,
    S: BuildHasher,
{
    fn extend<T>(&mut self, iter: T)
    where
        T: IntoIterator<Item = Node<K, V>>,
    {
        for n in iter {
            self.insert_node(n);
        }
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
        self.extend(iter.into_iter().map(|(k, v)| Node::new(k, v)));
    }
}

impl<K, V, S> Extend<Node<K, V>> for CloneConMap<K, V, S>
where
    K: Clone + Hash + Eq,
    V: Clone,
    S: BuildHasher,
{
    fn extend<T>(&mut self, iter: T)
    where
        T: IntoIterator<Item = Node<K, V>>,
    {
        let mut me: &CloneConMap<_, _, _> = self;
        me.extend(iter);
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

impl<K, V> FromIterator<Node<K, V>> for CloneConMap<K, V>
where
    K: Clone + Hash + Eq,
    V: Clone,
{
    fn from_iter<T>(iter: T) -> Self
    where
        T: IntoIterator<Item = Node<K, V>>,
    {
        let mut me = CloneConMap::new();
        me.extend(iter);
        me
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

#[cfg(testXXX)]
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
        assert_eq!(Node::new("hello", "world"), *found);
    }

    #[test]
    fn insert_overwrite_lookup() {
        let map = ConMap::new();
        assert!(map.insert("hello", "world").is_none());
        let old = map.insert("hello", "universe").unwrap();
        assert_eq!(Node::new("hello", "world"), *old);
        let found = map.get("hello").unwrap();
        assert_eq!(Node::new("hello", "universe"), *found);
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
            .insert_node(Arc::new(Node::new(42, [1, 2, 3])))
            .is_none());
        let found = map.get(&42).unwrap();
        assert_eq!(&[1, 2, 3], found.value());
        let inserted = map.get_or_insert_with_node(0, |k| {
            assert_eq!(0, k);
            Arc::new(Node::new(k, []))
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
        let expected = (0..TEST_BATCH_SMALL).into_iter().collect::<Vec<_>>();
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
            .into_iter()
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
        let expected = (0..TEST_BATCH_SMALL).into_iter().collect::<Vec<_>>();
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
                    let iter = (start..start + TEST_BATCH_SMALL)
                        .into_iter()
                        .map(|i| (i, i));
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
        let expected = (0..TEST_THREADS * TEST_BATCH_SMALL)
            .into_iter()
            .collect::<Vec<_>>();
        assert_eq!(expected, extracted);
    }
}
