use std::marker::PhantomData;
use std::mem;
use std::sync::atomic::Ordering;

use arrayvec::ArrayVec;
use crossbeam_epoch::{Guard, Shared};

use super::config::Config;
use super::{load_data, nf, Inner, NodeFlags, Raw, LEVEL_CELLS, MAX_LEVELS};

unsafe fn extend_lifetime<'a, 'b, T: 'a + 'b>(s: Shared<'a, T>) -> Shared<'b, T> {
    mem::transmute(s)
}

struct Level<'a> {
    ptr: Shared<'a, Inner>,
    idx: usize,
}

// Notes about the lifetimes:
// The 'a here is actually a lie. We need two things from lifetimes:
// * We must not outlive the map we are iterating through (because the drop just outright destroys
//   the data).
// * The pointers must not outlive the pin we hold.
// * We do not mind us (or the pin) moving around in memory, we are only interested in when its
//   destructor is called. The references don't actually point inside the pin itself.
//
// The lifetime of the pin is the same as of the pointers we store inside of us. We check the
// lifetime relation of the map and us on the constructor, so we won't outlive the map. But
// technically, the lifetime should be something like `'self`, but it's not possible to describe.
//
// Therefore we have to make very sure to never return a reference with the 'a lifetime.
//
// For the same technical reasons, we do the extend_lifetime thing. It would be great if someone
// knew a better trick ‒ while this is probably correct, something the compiler could check would
// be much better.
pub struct Iter<'a, C, S>
where
    C: Config,
{
    pin: Guard,
    levels: ArrayVec<[Level<'a>; MAX_LEVELS + 1]>,
    _map: PhantomData<&'a Raw<C, S>>,
}

impl<'a, C, S> Iter<'a, C, S>
where
    C: Config,
{
    pub fn new<'m: 'a>(map: &'m Raw<C, S>) -> Self {
        let mut levels = ArrayVec::new();
        let pin = crossbeam_epoch::pin();
        let ptr = map.root.load(Ordering::Acquire, &pin);
        let ptr = unsafe { extend_lifetime(ptr) };
        levels.push(Level { ptr, idx: 0 });
        Iter {
            pin,
            levels,
            _map: PhantomData,
        }
    }

    // Not an iterator because this borrows out of the iterator itself (and effectively its pin).
    #[allow(clippy::should_implement_trait)]
    pub fn next(&mut self) -> Option<&C::Payload> {
        loop {
            let top = self.levels.last_mut()?;

            let flags = nf(top.ptr);
            if top.ptr.is_null() {
                self.levels.pop();
            } else if flags.contains(NodeFlags::DATA) {
                let data = unsafe { load_data::<C>(top.ptr) };
                if top.idx < data.len() {
                    let result = &data[top.idx];
                    top.idx += 1;
                    return Some(result);
                } else {
                    self.levels.pop();
                }
            } else if top.idx < LEVEL_CELLS {
                let node = unsafe { top.ptr.deref() };
                let ptr = node.0[top.idx].load(Ordering::Acquire, &self.pin);
                let ptr = unsafe { extend_lifetime(ptr) };
                top.idx += 1;
                self.levels.push(Level { ptr, idx: 0 });
            } else {
                self.levels.pop();
            }
        }
    }
}

// TODO: Tests. And for colliding things too.
