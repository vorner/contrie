use std::borrow::Borrow;
use std::hash::Hash;

// TODO: Allow our own hash, returning something else than just u64. Then the constants go here
// too.
// TODO: Should Hasher go here too?
// TODO: Can we get rid of that Clone here? It is currently needed in the collision handling.
pub trait Config {
    type Payload: Clone + Borrow<Self::Key>;
    type Key: Hash + Eq;
}
