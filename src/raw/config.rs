use std::borrow::Borrow;
use std::hash::Hash;
use std::marker::PhantomData;

// TODO: Allow our own hash, returning something else than just u64. Then the constants go here
// too.
// TODO: Should Hasher go here too?
// TODO: Can we get rid of that Clone here? It is currently needed in the collision handling.
/// Customization of the [`Raw`][crate::raw::Raw].
///
/// This specifies how the trie should act. Maybe some more customization will be possible in the
/// future, but for now this allows tweaking what in how is stored.
pub trait Config {
    /// The payload (eg. values) stored inside the trie.
    type Payload: Clone + Borrow<Self::Key>;

    /// Each payload must contain a key as its part. This is the type for the key, which is used
    /// for hashing and identification of values in the tree.
    type Key: Hash + Eq;
}

/// A trivial config, where the payload and the key are the same thing.
pub struct Trivial<T>(PhantomData<T>);

impl<T> Config for Trivial<T>
where
    T: Clone + Hash + Eq,
{
    type Payload = T;
    type Key = T;
}
