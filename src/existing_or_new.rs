use std::ops::{Deref, DerefMut};

#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum ExistingOrNew<T> {
    Existing(T),
    New(T),
}

impl<T> ExistingOrNew<T> {
    pub fn into_inner(self) -> T {
        match self {
            ExistingOrNew::Existing(get) => get,
            ExistingOrNew::New(insert) => insert,
        }
    }
    pub fn map<U, F: FnOnce(T) -> U>(self, f: F) -> ExistingOrNew<U> {
        match self {
            ExistingOrNew::Existing(get) => ExistingOrNew::Existing(f(get)),
            ExistingOrNew::New(insert) => ExistingOrNew::New(f(insert)),
        }
    }
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
