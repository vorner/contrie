use std::borrow::Borrow;
use std::collections::hash_map::RandomState;
use std::hash::{BuildHasher, Hash};
use std::marker::PhantomData;
use std::ops::Deref;
use std::sync::Arc;

use crate::existing_or_new::ExistingOrNew;
use crate::raw::config::Config;
use crate::raw::Raw;

#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct Node<K, V: ?Sized> {
    data: (K, V),
}

impl<K, V> Node<K, V> {
    pub fn new(key: K, value: V) -> Self {
        Self { data: (key, value) }
    }
}

impl<K, V: ?Sized> Node<K, V> {
    pub fn key(&self) -> &K {
        &self.data.0
    }

    pub fn value(&self) -> &V {
        &self.data.1
    }
}

impl<K, V: ?Sized> Deref for Node<K, V> {
    type Target = (K, V);
    fn deref(&self) -> &(K, V) {
        &self.data
    }
}

struct MapPayload<K, V: ?Sized>(Arc<Node<K, V>>);

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
    V: ?Sized,
    K: Hash + Eq,
{
    type Payload = MapPayload<K, V>;
    type Key = K;
}

// TODO: Bunch of derives? Which ones? And which one do we need to implement?
pub struct ConMap<K, V, S = RandomState>
where
    // TODO: It would be great if the bounds wouldn't have to be on the struct, only on the impls
    K: Hash + Eq,
    V: ?Sized,
{
    raw: Raw<MapConfig<K, V>, S>,
}

impl<K, V> ConMap<K, V>
where
    K: Hash + Eq,
    V: ?Sized,
{
    pub fn new() -> Self {
        Self::with_hasher(RandomState::default())
    }
}

// TODO: Once we have the unsized locals, this should be possible to move into the V: ?Sized block
impl<K, V, S> ConMap<K, V, S>
where
    K: Hash + Eq,
    S: BuildHasher,
{
    pub fn insert(&self, key: K, value: V) -> Option<Arc<Node<K, V>>> {
        self.insert_node(Arc::new(Node::new(key, value)))
    }

    pub fn insert_node(&self, node: Arc<Node<K, V>>) -> Option<Arc<Node<K, V>>> {
        let pin = crossbeam_epoch::pin();
        self.raw
            .insert(MapPayload(node), &pin)
            .map(|p| Arc::clone(&p.0))
    }

    pub fn get_or_insert_with<F>(&self, key: K, create: F) -> ExistingOrNew<Arc<Node<K, V>>>
    where
        F: FnOnce() -> V,
    {
        let pin = crossbeam_epoch::pin();
        self.raw
            .get_or_insert_with(
                key,
                |key| {
                    let value = create();
                    MapPayload(Arc::new(Node::new(key, value)))
                },
                &pin,
            )
            .map(|payload| Arc::clone(&payload.0))
    }

    pub fn get_or_insert(&self, key: K, value: V) -> ExistingOrNew<Arc<Node<K, V>>> {
        self.get_or_insert_with(key, || value)
    }

    pub fn get_or_insert_default(&self, key: K) -> ExistingOrNew<Arc<Node<K, V>>>
    where
        V: Default,
    {
        self.get_or_insert_with(key, V::default)
    }

    pub fn get<Q>(&self, key: &Q) -> Option<Arc<Node<K, V>>>
    where
        Q: ?Sized + Eq + Hash,
        K: Borrow<Q>,
    {
        let pin = crossbeam_epoch::pin();
        self.raw.get(key, &pin).map(|r| Arc::clone(&r.0))
    }

    pub fn remove<Q>(&self, key: &Q) -> Option<Arc<Node<K, V>>>
    where
        Q: ?Sized + Eq + Hash,
        K: Borrow<Q>,
    {
        let pin = crossbeam_epoch::pin();
        self.raw.remove(key, &pin).map(|r| Arc::clone(&r.0))
    }

    pub fn is_empty(&self) -> bool {
        self.raw.is_empty()
    }
}

impl<K, V, S> ConMap<K, V, S>
where
    K: Hash + Eq,
    V: ?Sized,
    S: BuildHasher,
{
    pub fn with_hasher(hasher: S) -> Self {
        Self {
            raw: Raw::with_hasher(hasher),
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
}
