use std::borrow::Borrow;
use std::hash::Hash;

// TODO: Allow our own hash, returning something else than just u64. Then the constants go here
// too.
// TODO: Should Hasher go here too?
pub trait Config {
    type Payload: Clone + Borrow<Self::Key>;
    type Key: Hash + Eq;
}
