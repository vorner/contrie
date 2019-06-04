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
                return;
            } else if flags.contains(NodeFlags::DATA) {
                let data = unsafe { load_data::<C>(sub) };
                assert!(data.len() > 0, "Empty data nodes should not exist");
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
