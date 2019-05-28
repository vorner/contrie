use std::borrow::Borrow;
use std::hash::Hash;
use std::marker::PhantomData;
use std::ops::Deref;
use std::sync::Arc;

// TODO: Allow our own hash, returning something else than just u64. Then the constants go here
// too.
// TODO: Should Hasher go here too?
pub trait Config {
    type Payload: Clone + Borrow<Self::Key>;
    type Key: Hash + Eq;
}

// TODO: These things probably should go into some map.rs and be mostly hidden there
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct Leaf<K, V: ?Sized> {
    data: (K, V),
}

impl<K, V> Leaf<K, V> {
    pub fn new(key: K, value: V) -> Self {
        Self { data: (key, value) }
    }
}

impl<K, V: ?Sized> Leaf<K, V> {
    pub fn key(&self) -> &K {
        &self.data.0
    }

    pub fn value(&self) -> &V {
        &self.data.1
    }
}

impl<K, V: ?Sized> Deref for Leaf<K, V> {
    type Target = (K, V);
    fn deref(&self) -> &(K, V) {
        &self.data
    }
}

pub(crate) struct MapPayload<K, V: ?Sized>(pub(crate) Arc<Leaf<K, V>>);

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

pub(crate) struct MapConfig<K, V: ?Sized>(pub(crate) PhantomData<(K, V)>);

impl<K, V> Config for MapConfig<K, V>
where
    V: ?Sized,
    K: Hash + Eq,
{
    type Payload = MapPayload<K, V>;
    type Key = K;
}
