use std::borrow::Borrow;
use std::collections::hash_map::RandomState;
use std::hash::{BuildHasher, Hash, Hasher};
use std::mem;
use std::ops::Deref;
use std::sync::atomic::Ordering;
use std::sync::Arc;

use crossbeam_epoch::{Atomic, Owned};

// All directly written, some things are not const fn yet :-(. But tested below.
const LEVEL_BITS: usize = 4;
const LEVEL_MASK: u64 = 0b1111;
const LEVEL_CELLS: usize = 16;

type Cells<K, V> = [Atomic<Node<K, V>>; LEVEL_CELLS];

#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct Leaf<K, V> {
    data: (K, V),
}

impl<K, V> Leaf<K, V> {
    pub fn new(key: K, value: V) -> Self {
        Self { data: (key, value) }
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
    Collision(Box<[Arc<Leaf<K, V>>]>),
}

enum TraverseState<K, V, F> {
    Empty, // Invalid temporary state.
    Created(Arc<Leaf<K, V>>),
    Future { key: K, constructor: F },
}

impl<K, V, F: FnOnce() -> V> TraverseState<K, V, F> {
    fn key(&self) -> &K {
        match self {
            TraverseState::Empty => unreachable!("Not supposed to live in the empty state"),
            TraverseState::Created(leaf) => leaf.key(),
            TraverseState::Future { key, .. } => key,
        }
    }
    fn leaf(&mut self) -> Arc<Leaf<K, V>> {
        let (new_val, result) = match mem::replace(self, TraverseState::Empty) {
            TraverseState::Empty => unreachable!("Not supposed to live in the empty state"),
            TraverseState::Created(leaf) => (TraverseState::Created(Arc::clone(&leaf)), leaf),
            TraverseState::Future { key, constructor } => {
                let value = constructor();
                let leaf = Arc::new(Leaf::new(key, value));
                let created = TraverseState::Created(Arc::clone(&leaf));
                (created, leaf)
            }
        };
        *self = new_val;
        result
    }
    fn leaf_owned(&mut self) -> Owned<Node<K, V>> {
        Owned::new(Node::Leaf(self.leaf()))
    }
    fn into_leaf(self) -> Arc<Leaf<K, V>> {
        match self {
            TraverseState::Created(leaf) => leaf,
            TraverseState::Future { key, constructor } => Arc::new(Leaf::new(key, constructor())),
            TraverseState::Empty => unreachable!("Not supposed to live in the empty state"),
        }
    }
    fn into_return(self, mode: TraverseMode) -> Option<Arc<Leaf<K, V>>> {
        match mode {
            TraverseMode::Overwrite => None,
            TraverseMode::IfMissing => Some(self.into_leaf()),
        }
    }
}

#[derive(Copy, Clone, Eq, PartialEq)]
enum TraverseMode {
    Overwrite,
    IfMissing,
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
        self.traverse(
            TraverseState::<K, V, fn() -> V>::Created(leaf),
            TraverseMode::Overwrite,
        )
    }

    fn traverse<F>(
        &self,
        mut state: TraverseState<K, V, F>,
        mode: TraverseMode,
    ) -> Option<Arc<Leaf<K, V>>>
    where
        F: FnOnce() -> V,
    {
        let hash = self.hash(state.key());
        let mut shift = 0;
        let mut current = &self.root;
        let pin = crossbeam_epoch::pin();
        loop {
            let node = current.load(Ordering::Acquire, &pin);
            let replace = |with, delete_previous| {
                // If we fail to set it, the `with` is dropped together with the Err case, freeing
                // whatever was inside it.
                let result = current
                    .compare_and_set_weak(node, with, Ordering::Release, &pin)
                    .is_ok();
                if result && !node.is_null() && delete_previous {
                    unsafe { pin.defer_destroy(node) };
                }
                result
            };
            match unsafe { node.as_ref() } {
                Some(Node::Inner(inner)) => {
                    let bits = (hash >> shift) & LEVEL_MASK;
                    shift += LEVEL_BITS;
                    current = &inner[bits as usize];
                }
                // Not found, create it.
                None => {
                    if replace(state.leaf_owned(), true) {
                        return state.into_return(mode);
                        // else -> retry
                    }
                }
                Some(Node::Leaf(old)) if &old.0 == state.key() => {
                    // If we don't overwrite, we just return the found old value. If we do, try
                    // doing so (and possibly fail & repeat the attempt).
                    if mode == TraverseMode::Overwrite && !replace(state.leaf_owned(), true) {
                        // Retry.
                        continue;
                    }
                    // Note: we abuse the epochs/pins. If we asked for replacement, by now we've
                    // already scheduled the node containing old to be destroyed. But because we
                    // haven't released the pin yet, it still exists so we can +1 the Arc.
                    return Some(Arc::clone(old));
                }
                // Collision on the whole length of the hash :-(
                Some(Node::Leaf(other)) if shift >= mem::size_of_val(&hash) * 8 => {
                    let collision = Box::new([Arc::clone(other), state.leaf()]);
                    let collision = Owned::new(Node::Collision(collision));
                    if replace(collision, true) {
                        return state.into_return(mode);
                    }
                }
                // Not found, going to create it.
                Some(Node::Leaf(other)) => {
                    // We need to add another level. Note: there *still* might be a collision.
                    // Therefore, we just add the level and try again.
                    // FIXME: Once we have deletion, this could be adding & removing forever and
                    // we need to do it in one step.
                    let other_hash = self.hash(&other.0);
                    let other_bits = (other_hash >> shift) & LEVEL_MASK;
                    let mut inner = Cells::default();
                    inner[other_bits as usize] = Atomic::from(node);
                    let split = Owned::new(Node::Inner(inner));
                    // No matter if it succeeds or fails, we try again. We'll either find the newly
                    // inserted value here and continue with another level down, or it gets
                    // destroyed and we try splitting again.
                    replace(split, false);
                }
                Some(Node::Collision(leaves)) => {
                    let old = leaves
                        .iter()
                        .find(|l| l.key().borrow() == state.key())
                        .map(Arc::clone);

                    if old.is_none() || mode == TraverseMode::Overwrite {
                        let mut new = Vec::with_capacity(leaves.len() + 1);
                        new.extend(leaves.iter().filter(|l| l.key() != state.key()).cloned());
                        new.push(state.leaf());
                        let new = Owned::new(Node::Collision(new.into_boxed_slice()));
                        if !replace(new, true) {
                            continue;
                        }
                    }

                    return old.or_else(|| state.into_return(mode));
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
                Node::Collision(leaves) => {
                    return leaves
                        .iter()
                        .find(|l| l.key().borrow() == key)
                        .map(Arc::clone)
                }
            }
        }
    }

    pub fn get_or_insert_with<F>(&self, key: K, create: F) -> Arc<Leaf<K, V>>
    where
        F: FnOnce() -> V,
    {
        let state = TraverseState::Future {
            key,
            constructor: create,
        };
        self.traverse(state, TraverseMode::IfMissing)
            .expect("Should have created one for me")
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
                    Node::Inner(inner) => {
                        for sub in inner {
                            drop_recursive(sub);
                        }
                    }
                    Node::Collision(_) => (),
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

    // A hasher to create collisions on purpose. Let's make the hash trie into a glorified array.
    struct NoHasher;

    impl Hasher for NoHasher {
        fn finish(&self) -> u64 {
            0
        }

        fn write(&mut self, _: &[u8]) {}
    }

    impl BuildHasher for NoHasher {
        type Hasher = NoHasher;

        fn build_hasher(&self) -> NoHasher {
            NoHasher
        }
    }

    #[test]
    fn collisions() {
        let map = ConMap::with_hasher(NoHasher);
        // While their hash is the same under the hasher, they don't kick each other out.
        for i in 1..42 {
            assert!(map.insert(i, i).is_none());
        }
        // And all are present.
        for i in 1..42 {
            assert_eq!(i, *map.get(&i).unwrap().value());
        }
        // But reusing the key kick the other one out.
        for i in 1..42 {
            assert_eq!(i, *map.insert(i, i + 1).unwrap().value());
            assert_eq!(i + 1, *map.get(&i).unwrap().value());
        }
    }

    // TODO: Tests for get_or_insert
}
