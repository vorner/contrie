//! The [`ExistingOrNew`][crate::ExistingOrNew] enum.

use std::ops::{Deref, DerefMut};

/// A simple enum to make a distinction if the returned value is an already existing one or a newly
/// created instance.
///
/// As it dereferences to the held value, it acts almost as the value `T` in many circumstances.
/// Otherwise it can also be extracted.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum ExistingOrNew<T> {
    /// The entry already existed.
    Existing(T),
    /// A new entry had to be inserted to satisfy the request.
    New(T),
}

impl<T> ExistingOrNew<T> {
    /// Extracts the inner value.
    pub fn into_inner(self) -> T {
        match self {
            ExistingOrNew::Existing(get) => get,
            ExistingOrNew::New(insert) => insert,
        }
    }

    /// Applies a transformation to the value.
    ///
    /// This applies the function to the inner value, but preserves the information if it was a new
    /// or existing one.
    pub fn map<U, F: FnOnce(T) -> U>(self, f: F) -> ExistingOrNew<U> {
        match self {
            ExistingOrNew::Existing(get) => ExistingOrNew::Existing(f(get)),
            ExistingOrNew::New(insert) => ExistingOrNew::New(f(insert)),
        }
    }

    /// Checks if the value was newly created one.
    pub fn is_new(&self) -> bool {
        match self {
            ExistingOrNew::New(_) => true,
            ExistingOrNew::Existing(_) => false,
        }
    }
}

impl<T> Deref for ExistingOrNew<T> {
    type Target = T;
    fn deref(&self) -> &T {
        match self {
            ExistingOrNew::Existing(get) => get,
            ExistingOrNew::New(insert) => insert,
        }
    }
}

impl<T> DerefMut for ExistingOrNew<T> {
    fn deref_mut(&mut self) -> &mut T {
        match self {
            ExistingOrNew::Existing(get) => get,
            ExistingOrNew::New(insert) => insert,
        }
    }
}
