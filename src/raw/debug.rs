//! A module containing few debug utilities.
//!
//! In general, they are meant for debugging the *trie itself*, but it is exposed as potentially
//! useful.

use std::fmt::{Debug, Display, Formatter, Result as FmtResult};
use std::sync::atomic::Ordering;

use crossbeam_epoch::{self, Atomic, Guard};

use super::config::Config;
use super::{load_data, nf, Inner, NodeFlags, Raw};

impl<C, S> Raw<C, S>
where
    C: Config,
{
    // Hack: &mut to make sure it is not shared between threads and nobody is modifying the thing
    // right now.
    /// Panics if the trie is not in consistent state and pruned well.
    ///
    /// Note that if the caller can get the mutable reference, it should be in pruned state, even
    /// though during modifications there might be temporary states which are not pruned. Due to
    /// unique access to it, other threads might not be modifying it at the moment.
    #[cfg(test)]
    pub(crate) fn assert_pruned(&mut self) {
        fn handle_ptr<C: Config>(ptr: &Atomic<Inner>, data_cnt: &mut usize, seen_inner: &mut bool) {
            // Unprotected is fine, we are &mut so nobody else is allowed to do stuff to us at the
            // moment.
            let pin = unsafe { crossbeam_epoch::unprotected() };
            // Relaxed is fine for the same reason ‒ we are &mut
            let sub = ptr.load(Ordering::Relaxed, &pin);
            let flags = nf(sub);

            assert!(!flags.contains(NodeFlags::CONDEMNED));

            if sub.is_null() {
                // Do nothing here
            } else if flags.contains(NodeFlags::DATA) {
                let data = unsafe { load_data::<C>(sub) };
                assert!(!data.is_empty(), "Empty data nodes should not exist");
                *data_cnt += data.len();
            } else {
                let sub = unsafe { sub.deref() };
                *seen_inner = true;
                check_node::<C>(sub);
            }
        }
        fn check_node<C: Config>(node: &Inner) {
            let mut data_cnt = 0;
            let mut seen_inner = false;
            for ptr in &node.0 {
                handle_ptr::<C>(ptr, &mut data_cnt, &mut seen_inner);
            }

            assert!(
                data_cnt > 1 || seen_inner,
                "This node should have been pruned"
            );
        }

        handle_ptr::<C>(&self.root, &mut 0, &mut false);
    }

    fn print_shape_ptr(ptr: &Atomic<Inner>, fmt: &mut Formatter, pin: &Guard) -> FmtResult
    where
        C::Payload: Debug,
    {
        let ptr = ptr.load(Ordering::Acquire, pin);
        let flags = nf(ptr);
        write!(fmt, "{:?}/{:?}", ptr.as_raw(), flags)?;

        if ptr.is_null() {
            // Nothing
        } else if flags.contains(NodeFlags::DATA) {
            let data = unsafe { load_data::<C>(ptr) };
            write!(fmt, "{:?}", data)?;
        } else {
            let inner = unsafe { ptr.deref() };
            write!(fmt, "(")?;
            for (idx, sub) in inner.0.iter().enumerate() {
                write!(fmt, " {:X}:", idx)?;
                Self::print_shape_ptr(sub, fmt, pin)?;
            }
            write!(fmt, " )")?;
        }
        Ok(())
    }

    fn print_shape(&self, fmt: &mut Formatter) -> FmtResult
    where
        C::Payload: Debug,
    {
        let pin = crossbeam_epoch::pin();
        Self::print_shape_ptr(&self.root, fmt, &pin)
    }
}

/// A pretty-printing wrapper around the raw trie.
///
/// The structure, including the pointers and flags, is printed if this is used to wrap the raw
/// trie.
pub struct PrintShape<'a, C, S>(pub &'a Raw<C, S>)
where
    C: Config;

impl<C, S> Display for PrintShape<'_, C, S>
where
    C: Config,
    C::Payload: Debug,
{
    fn fmt(&self, fmt: &mut Formatter) -> FmtResult {
        self.0.print_shape(fmt)
    }
}
