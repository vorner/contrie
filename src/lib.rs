use std::borrow::Borrow;
use std::collections::hash_map::RandomState;
use std::hash::{BuildHasher, Hash, Hasher};
use std::ops::Deref;
use std::sync::Arc;
use std::sync::atomic::Ordering;

use crossbeam_epoch::{Atomic, Owned};

// All directly written, some things are not const fn yet :-(. But tested below.
const LEVEL_BITS: usize = 4;
const LEVEL_MASK: u64 = 0b1111;
const LEVEL_CELLS: usize = 16;

type Cells<K, V> = [Atomic<Node<K, V>>; LEVEL_CELLS];

#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct Leaf<K, V> {
    data: (K, V)
}

impl<K, V> Leaf<K, V> {
    pub fn new(key: K, value: V) -> Self {
        Self {
            data: (key, value),
        }
    }

    pub fn key(&self) -> &K {
        &self.data.0
    }

    pub fn value(&self) -> &V {
        &self.data.1
    }
}

impl<K, V> Deref for Leaf<K, V> {
    type Target = (K, V);
    fn deref(&self) -> &(K, V) {
        &self.data
    }
}

enum Node<K, V> {
    Inner(Cells<K, V>),
    Leaf(Arc<Leaf<K, V>>),
    // TODO: Collision
}

impl<K, V> Node<K, V> {
    fn key(&self) -> Option<&K> {
        if let Node::Leaf(l) = self {
            Some(&l.data.0)
        } else {
            None
        }
    }
}

pub struct ConMap<K, V, S = RandomState> {
    hash_builder: S,
    root: Atomic<Node<K, V>>,
}

impl<K: Eq + Hash, V> ConMap<K, V, RandomState> {
    pub fn new() -> Self {
        Self::default()
    }
}

impl<K, V, S> ConMap<K, V, S>
where
    K: Eq + Hash,
    S: BuildHasher,
{
    pub fn with_hasher(hash_builder: S) -> Self {
        Self {
            hash_builder,
            root: Atomic::null(),
        }
    }

    fn hash<Q>(&self, key: &Q) -> u64
    where
        Q: ?Sized + Hash,
    {
        let mut hasher = self.hash_builder.build_hasher();
        key.hash(&mut hasher);
        hasher.finish()
    }

    pub fn insert(&self, key: K, value: V) -> Option<Arc<Leaf<K, V>>> {
        self.insert_leaf(Arc::new(Leaf::new(key, value)))
    }

    pub fn insert_leaf(&self, leaf: Arc<Leaf<K, V>>) -> Option<Arc<Leaf<K, V>>> {
        let hash = self.hash(&leaf.0);
        let mut shift = 0;
        let mut leaf = Owned::new(Node::Leaf(leaf));
        let mut current = &self.root;
        let pin = crossbeam_epoch::pin();
        loop {
            let node = current.load(Ordering::Acquire, &pin);
            match unsafe { node.as_ref() } {
                None => match current.compare_and_set_weak(node, leaf, Ordering::Release, &pin) {
                    // Didn't replace anything, so we are good.
                    Ok(_) => return None,
                    // Retry
                    Err(fail) => leaf = fail.new,
                }
                Some(Node::Leaf(old)) if &old.0 == leaf.key().unwrap() => {
                    match current.compare_and_set_weak(node, leaf, Ordering::AcqRel, &pin) {
                        // Replaced an old one. Return it, but destroy the internal node.
                        Ok(_) => {
                            unsafe { pin.defer_destroy(node) };
                            return Some(Arc::clone(old))
                        }
                        Err(fail) => leaf = fail.new,
                    }
                }
                Some(Node::Leaf(other)) => {
                    // We need to add another level. Note: there *still* might be a collision.
                    // Therefore, we just add the level and try again.
                    // FIXME: We may run out of hash bits here and go into a collision.
                    // FIXME: Once we have deletion, this should be adding & removing forever and
                    // we need to do it in one step.
                    let other_hash = self.hash(&other.0);
                    let other_bits = (other_hash >> shift) & LEVEL_MASK;
                    let mut inner = Cells::default();
                    inner[other_bits as usize] = Atomic::from(node);
                    let split = Owned::new(Node::Inner(inner));
                    match current.compare_and_set_weak(node, split, Ordering::Release, &pin) {
                        // Just try going there once more
                        Ok(_) => (),
                        // Let's get rid of the one we didn't manage to put in. As it never left
                        // our thread, we can just delete it right away.
                        Err(fail) => drop(fail.new),
                    }
                }
                Some(Node::Inner(inner)) => {
                    let bits = (hash >> shift) & LEVEL_MASK;
                    shift += LEVEL_BITS;
                    current = &inner[bits as usize];
                }
            }
        }
    }

    pub fn get<Q>(&self, key: &Q) -> Option<Arc<Leaf<K, V>>>
    where
        Q: ?Sized + Eq + Hash,
        K: Borrow<Q>,
    {
        let mut current = &self.root;
        let mut hash = self.hash(key);
        let pin = crossbeam_epoch::pin();
        loop {
            let node = current.load(Ordering::Acquire, &pin);
            match unsafe { node.as_ref() }? {
                Node::Leaf(leaf) if leaf.0.borrow() == key => return Some(Arc::clone(leaf)),
                Node::Leaf(_) => return None,
                Node::Inner(inner) => {
                    let bits = hash & LEVEL_MASK;
                    hash >>= LEVEL_BITS;
                    current = &inner[bits as usize];
                }
            }
        }
    }

    pub fn get_or_insert_with<F>(&self, key: K, create: F) -> Arc<Leaf<K, V>>
    where
        F: FnOnce() -> V,
    {
        unimplemented!()
    }

    pub fn get_or_insert(&self, key: K, value: V) -> Arc<Leaf<K, V>> {
        self.get_or_insert_with(key, || value)
    }

    pub fn get_or_insert_default(&self, key: K) -> Arc<Leaf<K, V>>
    where
        V: Default,
    {
        self.get_or_insert_with(key, V::default)
    }

    pub fn remove<Q>(&self, key: &Q) -> Option<Arc<Leaf<K, V>>>
    where
        Q: ?Sized + Eq + Hash,
        K: Borrow<Q>,
    {
        unimplemented!()
    }

    // TODO: Iteration & friends
}

// Implementing manually, derive would ask for K, V: Default
impl<K, V, S> Default for ConMap<K, V, S>
where
    K: Eq + Hash,
    S: BuildHasher + Default,
{
    fn default() -> Self {
        Self::with_hasher(S::default())
    }
}

impl<K, V, S> Drop for ConMap<K, V, S> {
    fn drop(&mut self) {
        /*
         * Notes about unsafety here:
         * * We are in a destructor and that one is &mut self. There are no concurrent accesses to
         *   this data structure any more, therefore we can safely assume we are the only ones
         *   looking at the pointers inside.
         * * Therefore, using unprotected is also fine.
         * * Similarly, the Relaxed ordering here is fine too, as the whole data structure must
         *   have been synchronized into our thread already by this time.
         * * The pointer inside this data structure is never dangling.
         */
        unsafe fn drop_recursive<K, V>(node: &Atomic<Node<K, V>>) {
            let pin = crossbeam_epoch::unprotected();
            let extract = node.load(Ordering::Relaxed, &pin);
            if !extract.is_null() {
                let extract = extract.into_owned();
                match extract.deref() {
                    Node::Leaf(_) => (),
                    Node::Inner(inner) => for sub in inner {
                        drop_recursive(sub);
                    }
                }
                drop(extract);
            }
        }
        unsafe { drop_recursive(&self.root) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crossbeam_utils::thread;

    const TEST_THREADS: usize = 4;
    const TEST_BATCH: usize = 10000;
    const TEST_REP: usize = 20;

    #[test]
    fn consts_consistent() {
        assert_eq!(LEVEL_BITS, LEVEL_MASK.count_ones() as usize);
        assert_eq!(LEVEL_BITS, (!LEVEL_MASK).trailing_zeros() as usize);
        assert_eq!(LEVEL_CELLS, 2usize.pow(LEVEL_BITS as u32));
    }

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
        assert_eq!(Leaf::new("hello", "world"), *found);
    }

    #[test]
    fn insert_overwrite_lookup() {
        let map = ConMap::new();
        assert!(map.insert("hello", "world").is_none());
        let old = map.insert("hello", "universe").unwrap();
        assert_eq!(Leaf::new("hello", "world"), *old);
        let found = map.get("hello").unwrap();
        assert_eq!(Leaf::new("hello", "universe"), *found);
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
            }).unwrap();

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
            }).unwrap();
        }
    }
}
