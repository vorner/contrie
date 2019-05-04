use std::borrow::Borrow;
use std::collections::hash_map::RandomState;
use std::hash::{BuildHasher, Hash};
use std::sync::Arc;

use crossbeam_epoch::Atomic;

const LEVEL_BITS: usize = 3;
const LEVEL_CELLS: usize = 2^LEVEL_BITS;

type Cells<K, V> = [Atomic<Node<K, V>>; LEVEL_CELLS];

#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct Leaf<K, V>(K, V);

enum Node<K, V> {
    Inner(Cells<K, V>),
    Leaf(Arc<Leaf<K, V>>),
    // TODO: Collision
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

    pub fn insert(&self, key: K, value: V) -> Option<Arc<Leaf<K, V>>> {
        self.insert_leaf(Arc::new(Leaf(key, value)))
    }

    pub fn insert_leaf(&self, leaf: Arc<Leaf<K, V>>) -> Option<Arc<Leaf<K, V>>> {
        unimplemented!()
    }

    pub fn get<Q>(&self, key: &Q) -> Option<Arc<Leaf<K, V>>>
    where
        Q: ?Sized + Eq + Hash,
        K: Borrow<Q>,
    {
        unimplemented!()
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
