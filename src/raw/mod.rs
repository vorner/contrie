#![allow(unsafe_code)]
//! The core implementation of the concurrent trie data structure.
//!
//! This module contains the [`Raw`][crate::raw::Raw] type, which is the engine of all the data
//! structures in this crate. This is exposed to allow wrapping it into further APIs, but is
//! probably not the best thing for general use.

// # The data structure
//
// The data structure is inspired by the [article], however severely simplified. Compared to the
// article, what we don't do (if you don't want to read the article, that's fine, explanation is
// below):
//
// * We don't have variable-sized inner nodes. This wastes some more space, but also allows us to
//   keep the same node around instead of creating a new one every time we want to add or remove a
//   pointer.
// * We don't do snapshots for iterations.
// * We got rid of the I-nodes. This gets rid of half of the pointer loads on the way to the
//   element, so in theory it should make the data structure about twice faster.
//
// By this simplification, we lose the ability of having consistent iterations and we use somewhat
// more memory, but get faster data structure. More importantly, the data structure is simpler to
// implement and simpler to prove correct which makes it more likely to actually trust it with some
// data.
//
// ## How it works
//
// The heart of the data structure is a trie where keys are prefixes of the 64bit hash of the key.
// Each inner node has 16 pointer slots, indexed by the next 4 bits of the hash. When we reach a
// level where the prefix is unique, we stop (we don't have all 16 levels of inner nodes if we
// don't have to) and place a data node.
//
// A data nodes contain one or more elements (more in case we get a hash collision on the whole
// hash ‒ in that case, we store all the colliding elements in an array and distinguish by equality
// of the keys in a linear search through the array).
//
// On lookup, we either find the correct element or stop at the first null pointer encountered.
//
// On insertion, if we find a null pointer, we atomically replace that pointer to a new data node
// containing (only) the new element, using the CaS operation. In case we reach a collision or
// replace an existing element, we create a new data node and replace the pointer, again using the
// CaS operation. If we find a non-matching data node in our way, we need to insert another level ‒
// we create a brand new inner node, link the old node there and again, replace the pointer (then
// retry with our insertion on the next level).
//
// Deletion looks up the element and either replaces the pointer to the data node with null (if it
// was the last one), or creates a new data node without the element.
//
// ## Pruning
//
// If implemented as above, the data structure would work. However, deletions would leave unneeded
// dead branches or branches that don't branch (eg. linear ones) behind. That would make the data
// structure perform worse than necessary, because the lookups would have to traverse the dead or
// non-branching branches and it would need more memory. Therefore, after we remove an element, we
// want to walk the path back towards the root and remove the nodes that are no longer needed
// (either remove the branch completely or contract the end that doesn't branch).
//
// If we, however, started to simply delete the nodes and replace the pointers with nulls, we would
// get race conditions:
//
// 1. We check that the current node is empty (or has only one pointer to data in it) and therefore
//    is unneeded.
// 2. After we do the check but before we manage to update the pointer that points to the current
//    node, some other thread adds a new pointer into the node. This would make it ineligible for
//    deletion, but we've already done our check.
// 3. We don't know about the addition from the other thread and go ahead with the deletion. This
//    loses and leaks the added element, or any update the other thread has done.
//
// The original article used the I-nodes and another kind of nodes to solve this problem. We are
// going to get inspired by what they did, but will do it inline in the array of pointers.
//
// On any sane system, data structures like our inner or data nodes are aligned to multiples of
// some number (assuming we have at least 32bit system, it's at least multiple of 4 bytes). This
// leaves two bits that are always 0 in the pointers. We can abuse these bits to store additional
// flags (we use the utilities of crossbeam_utils to manage the bits). One of these bits, when set
// to 1, will mean that the pointer is no longer allowed to be updated.
//
// So, when removing, we first check once if the inner node may be removed. If not, we're done. If
// it looks like it may be, we walk the pointers again, but this time we atomically read and mark
// them with the flag. This'll make sure nobody is allowed to sneak an update to it past our second
// check (the first one is just an optimisation). So, if we pass the second check, we can safely
// proceed and remove the node. Hurray. We can move one level up towards the root and repeat.
//
// In case the second check failed, we have already marked all the pointers that they are never
// ever allowed to be updated again. We can't leave a node like that in the data structure forever.
// Therefore, we create a new copy of the node, with clean pointers and replace the node with that
// (in that case we can stop the processing at this level).
//
// So, what the other thread that wants to update the pointer but can't because it is marked does?
// It can't wait for the thread that did the marking to finish its job, because then the algorithm
// would no longer be lock-free. But it can decide to do its work for it and also do the pruning
// (only of this particular one inner node; it won't walk further towards the root). It proceeds to
// the second check stage ‒ marks all the pointers (even if they are already marked) and deciding
// if it can remove it completely or if it needs to create a brand new copy. One of the threads
// competing for the prune operation will succeed, the other will fail during the CaS update of the
// parent pointer, but both can proceed because the pruning already happened.
//
// As this collision on node being prune is likely to be rare in practice and it is already
// relatively complex (and hard to test for) situation, the other thread simply completely restarts
// the operation instead of trying to get all the corner cases right. The removal thread, however,
// proceeds towards the root even in the collision situation ‒ it is responsible for pruning as far
// as possible and when going up, the corner cases don't actually happen (well, with the exception
// of its whole branch being already removed or contracted away by another removal, but in that
// case it'll just waste a little bit of effort in trying to remove stuff in places that are going
// to be thrown away anyway ‒ but thanks to crossbeam_epoch, it is still valid memory).
//
// ## Iteration
//
// Similar to lookups, iteration doesn't have to care about the flags about non-updateable
// pointers ‒ even the old, marked, pointers form a valid representation of the map at a certain
// point of time, though not necessarily optimally small.
//
// Therefore, the iterator simply keeps a stack of nodes it is in, with indices into either the
// pointer array or the array of elements in a data node and does a DFS through the data structure.
//
// # Safety
//
// The current module contains a lot of unsafe code. In general, there are two kinds of things that
// could go wrong. Well, in addition to coding bugs, of course.
//
// ## Lifetimes & invalid pointers
//
// First, we simply never insert pointers that would be invalid at that time into the data
// structure ‒ whatever gets inserted is just brand new allocated thing. This boils down to just
// being careful and, as this is relatively short code, this is possible to accomplish.
//
// So, we must make sure nothing gets destroyed too soon. To accomplish this, we use the mechanism
// of crossbeam_epoch. When we remove something from the data structure, the destruction is
// postponed to when all the threads leave the current epoch. We never hold onto the pointers or
// references past the current method and we bind the lifetimes of the references to the lifetime
// of passed pin and us.
//
// There are two exceptions to this:
//
// * The destructor deletes things right away. But as it holds a mutable reference, we can't be
//   destroying anything for any other thread ‒ the other thread no longer holds reference to us.
// * The iterator binds the lifetimes to both itself and the map, but it holds a pin alive, so the
//   same would still apply.
//
// ## Inter-thread synchronization of data.
//
// In general, we use release ordering when putting data into the map and consume ordering when
// reading it out of it. The claim is, this is enough. But as the elements are independent on each
// other and only the inner nodes we traverse during the operation play any role to us (and we
// access the further nodes through the loaded pointers) and there's exactly one path from the root
// to each element (therefore anyone getting the element must have touched the same pointers), this
// is basically the definition of how release/consume synchronization edges work.
//
// There are some other orderings through the code, though:
//
// * Relaxed in case we *fail* a CaS update. However, in such case nothing is modified and we just
//   throw the data we've created ourselves out, so there's nothing to synchronize.
// * Relaxed in the first check for pruning. This one does not look at the actual data behind the
//   pointers, it simply counts how many pointers are non-null. The second pass does the actual
//   proper synchronization.
// * Relaxed in the is_empty. As we don't care what data it points to (only that it's not null),
//   this again doesn't have to synchronize anything.
// * Relaxed in the destructor. As argued above (and at the destructor itself), we have gained a
//   unique access to the whole map. Therefore, the whole map, containing the data in it must have
//   been properly synchronized into the thread already and we are in a single-threaded scenario.
// * AcqRel in the pruning. This is because we need to acquire the data pointed to in the case
//   we'll be making a copy and we'll have to re-release it later on. We also modify the value of
//   the pointer (with the flag), therefore anyone reading it after us only synchronizes against
//   us, so we also need to re-release it right now onto that pointer.
//
// [article]: https://www.researchgate.net/publication/221643801_Concurrent_Tries_with_Efficient_Non-Blocking_Snapshots

use std::borrow::Borrow;
use std::hash::{BuildHasher, Hash, Hasher};
use std::marker::PhantomData;
use std::mem;
use std::sync::atomic::Ordering;

use arrayvec::ArrayVec;
use bitflags::bitflags;
use crossbeam_epoch::{Atomic, Guard, Owned, Shared};
use smallvec::SmallVec;

pub mod config;
pub mod debug;
pub mod iterator;

use self::config::Config;
use crate::existing_or_new::ExistingOrNew;

// All directly written, some things are not const fn yet :-(. But tested below.
pub(crate) const LEVEL_BITS: usize = 4;
pub(crate) const LEVEL_MASK: u64 = 0b1111;
pub(crate) const LEVEL_CELLS: usize = 16;
pub(crate) const MAX_LEVELS: usize = mem::size_of::<u64>() * 8 / LEVEL_BITS;

bitflags! {
    /// Flags that can be put onto a pointer pointing to a node, specifying some interesting
    /// things.
    ///
    /// Note that this lives inside the unused bits of a pointer. All nodes align at least to a
    /// machine word and we assume it's at least 32bits, so we have at least 2 bits.
    struct NodeFlags: usize {
        /// The Inner containing this pointer is condemned to replacement/pruning.
        ///
        /// Changing this pointer is pointer is forbidden, and the containing Inner needs to be
        /// replaced first with a clean one.
        const CONDEMNED = 0b01;
        /// The pointer points not to an inner node, but to data node.
        ///
        /// # Rationale
        ///
        /// The [`Inner`] nodes are quite large. On the other hand, the values are usually just
        /// [`Arc`][std::sync::Arc] and there's usually just one at each leaf. That leaves a lot of
        /// wasted space.
        ///
        /// Therefore, instead of having an enum, we have nodes of two distinct types. We recognize
        /// them by this flag in the pointer pointing to them. If it is a leaf with data, this flag
        /// is set and anyone accessing it knows it needs to type cast the pointer before using.
        const DATA = 0b10;
    }
}

/// Extracts [`NodeFlags`] from a pointer.
fn nf(node: Shared<Inner>) -> NodeFlags {
    NodeFlags::from_bits(node.tag()).expect("Invalid node flags")
}

/// Type-casts the pointer to a [`Data`] node.
unsafe fn load_data<'a, C: Config>(node: Shared<'a, Inner>) -> &'a Data<C> {
    assert!(
        nf(node).contains(NodeFlags::DATA),
        "Tried to load data from inner node pointer"
    );
    (node.as_raw() as usize as *const Data<C>)
        .as_ref()
        .expect("A null pointer with data flag found")
}

/// Moves a data node behind an [`Owned`] pointer, casts it and provides the correct flags.
fn owned_data<C: Config>(data: Data<C>) -> Owned<Inner> {
    unsafe {
        Owned::<Inner>::from_raw(Box::into_raw(Box::new(data)) as usize as *mut _)
            .with_tag(NodeFlags::DATA.bits())
    }
}

/// Type-casts and drops the node as data.
unsafe fn drop_data<C: Config>(ptr: Shared<Inner>) {
    assert!(
        nf(ptr).contains(NodeFlags::DATA),
        "Tried to drop data from inner node pointer"
    );
    drop(Owned::from_raw(ptr.as_raw() as usize as *mut Data<C>));
}

/// An inner branching node of the trie.
///
/// This is just a bunch of pointers to lower levels.
#[derive(Default)]
struct Inner([Atomic<Inner>; LEVEL_CELLS]);

// Instead of distinguishing the very common case of single leaf and collision list in our code, we
// just handle everything as a list, possibly with 1 element.
//
// However, as the case with 1 element is much more probable, we don't want the Vec indirection
// there, so we let SmallVec to handle it by not spilling in that case. As the spilled Vec needs 2
// words in addition to the length (pointer and capacity), we have room for 2 Arcs in the not
// spilled case too, so we as well might take advantage of it.
// TODO: We want the union feature.
//
// Alternatively, we probably could use the raw allocator API and structure with len + [Arc<..>; 0].
// TODO: Compute the stack length based on the Payload size.
type Data<C> = SmallVec<[<C as Config>::Payload; 2]>;

enum TraverseState<C: Config, F> {
    Empty, // Invalid temporary state.
    Created(C::Payload),
    Future { key: C::Key, constructor: F },
}

impl<C: Config, F: FnOnce(C::Key) -> C::Payload> TraverseState<C, F> {
    fn key(&self) -> &C::Key {
        match self {
            TraverseState::Empty => unreachable!("Not supposed to live in the empty state"),
            TraverseState::Created(payload) => payload.borrow(),
            TraverseState::Future { key, .. } => key,
        }
    }
    fn payload(&mut self) -> C::Payload {
        let (new_val, result) = match mem::replace(self, TraverseState::Empty) {
            TraverseState::Empty => unreachable!("Not supposed to live in the empty state"),
            TraverseState::Created(payload) => (TraverseState::Created(payload.clone()), payload),
            TraverseState::Future { key, constructor } => {
                let payload = constructor(key);
                let created = TraverseState::Created(payload.clone());
                (created, payload)
            }
        };
        *self = new_val;
        result
    }
    fn data_owned(&mut self) -> Owned<Inner> {
        let mut data = Data::<C>::new();
        data.push(self.payload());
        owned_data::<C>(data)
    }
}

#[derive(Copy, Clone, Eq, PartialEq)]
enum TraverseMode {
    Overwrite,
    IfMissing,
}

/// How well pruning went.
#[derive(Copy, Clone, Eq, PartialEq)]
enum PruneResult {
    /// Removed the node completely, inserted NULL into the parent.
    Null,
    /// Contracted an edge, inserted a lone child.
    Singleton,
    /// Made a copy, as there were multiple pointers leading from the child.
    Copy,
    /// Failed to update the parent, some other thread updated it in the meantime.
    CasFail,
}

/// The raw hash trie data structure.
///
/// This provides the low level data structure. It does provide the lock-free operations on some
/// values. On the other hand, it does not provide user friendly interface. It is designed to
/// separate the single implementation of the core algorithm and provide a way to wrap it into
/// different interfaces for different use cases.
///
/// It, however, can be used to fulfill some less common uses.
///
/// The types stored inside and general behaviour is described by the [`Config`] type parameter and
/// can be customized using that.
///
/// As a general rule, this data structure takes the [`crossbeam_epoch`] [`Guard`] and returns
/// borrowed data whenever appropriate. This allows cheaper manipulation if necessary or grouping
/// multiple operations together. Note than even methods that would return owned values in
/// single-threaded case (eg. [`insert`][Raw::insert] and [`remove`][Raw::remove] return borrowed
/// values. This is because in concurrent situation some other thread might still be accessing
/// them. They are scheduled for destruction once the epoch ends.
///
/// For details of the internal implementation and correctness arguments, see the comments in
/// source code (they probably don't belong into API documentation).
pub struct Raw<C: Config, S> {
    hash_builder: S,
    root: Atomic<Inner>,
    _data: PhantomData<C::Payload>,
}

impl<C, S> Raw<C, S>
where
    C: Config,
    S: BuildHasher,
{
    /// Constructs an empty instance from the given hasher.
    pub fn with_hasher(hash_builder: S) -> Self {
        // Note: on any sane system, these assertions should actually never ever trigger no matter
        // what the user of the crate does. This is *internal* sanity check. If you ever find a
        // case where it *does* fail, open a bug report.
        assert!(
            mem::align_of::<Data<C>>().trailing_zeros() >= NodeFlags::all().bits().count_ones(),
            "BUG: Alignment of Data<Payload> is not large enough to store the internal flags",
        );
        assert!(
            mem::align_of::<Inner>().trailing_zeros() >= NodeFlags::all().bits().count_ones(),
            "BUG: Alignment of Inner not large enough to store internal flags",
        );
        Self {
            hash_builder,
            root: Atomic::null(),
            _data: PhantomData,
        }
    }

    /// Computes a hash (using the stored hasher) of a key.
    fn hash<Q>(&self, key: &Q) -> u64
    where
        Q: ?Sized + Hash,
    {
        let mut hasher = self.hash_builder.build_hasher();
        key.hash(&mut hasher);
        hasher.finish()
    }

    /// Inserts a new value, replacing and returning any previously held value.
    pub fn insert<'s, 'p, 'r>(
        &'s self,
        payload: C::Payload,
        pin: &'p Guard,
    ) -> Option<&'r C::Payload>
    where
        's: 'r,
        'p: 'r,
    {
        self.traverse(
            // Any way to do it without the type parameters here? Older rustc doesn't like them.
            TraverseState::<C, fn(C::Key) -> C::Payload>::Created(payload),
            TraverseMode::Overwrite,
            pin,
        )
        // TODO: Should we sanity-check this is Existing because it returns the previous value?
        .map(ExistingOrNew::into_inner)
    }

    /// Prunes the given node.
    ///
    /// * The parent points to the child node.
    /// * The child must be valid pointer, of course.
    ///
    /// The parent is made to point to either:
    /// * NULL if child is empty.
    /// * child's only child.
    /// * A copy of child.
    ///
    /// Returns how the pruning went.
    unsafe fn prune(pin: &Guard, parent: &Atomic<Inner>, child: Shared<Inner>) -> PruneResult {
        assert!(
            !nf(child).contains(NodeFlags::DATA),
            "Child passed to prune must not be data"
        );
        let inner = child.as_ref().expect("Null child node passed to prune");
        let mut allow_contract = true;
        let mut child_cnt = 0;
        let mut last_leaf = None;
        let mut new_child = Inner::default();

        // 1. Mark all the cells in this one as condemned.
        // 2. Look how many non-null branches are leading from there.
        // 3. Construct a copy of the child *without* the tags on the way.
        for (new, grandchild) in new_child.0.iter_mut().zip(&inner.0) {
            // Acquire ‒ Besides potentially looking at the child, we'll need to republish the
            // child in our swap of the pointer (this one and also the one below, in the CAS). To
            // do that we'll have to have acquired it first.
            //
            // Note that we don't need SeqCst here nor in the CaS below. We don't care about the
            // order ‒ the tagging is just making sure this particular slot never ever changes the
            // pointer. The CaS changes the trie in content-equivalent way, so observing either the
            // old or the new way is fine.
            let gc = grandchild.fetch_or(NodeFlags::CONDEMNED.bits(), Ordering::AcqRel, pin);
            // The flags we insert into the new one should not contain condemned flag even if it
            // was already present here.
            let flags = nf(gc) & !NodeFlags::CONDEMNED;
            let gc = gc.with_tag(flags.bits());
            if gc.is_null() {
                // Do nothing, just skip
            } else if flags.contains(NodeFlags::DATA) {
                last_leaf.replace(gc);
                let gc = load_data::<C>(gc);
                child_cnt += gc.len();
            } else {
                // If we have an inner node here, multiple leaves hang somewhere below there. More
                // importantly, we can't contrack the edge.
                allow_contract = false;
                child_cnt += 1;
            }

            *new = Atomic::from(gc);
        }

        // Now, decide what we want to put into the parent.
        let mut cleanup = None;
        let (insert, prune_result) = match (allow_contract, child_cnt, last_leaf) {
            // If there's exactly one leaf, we just contract the edge to lead there directly. Note
            // that we can't do that if this is not the leaf, because we would mess up the hash
            // matching on the way. But that's fine, we checked that above.
            (true, 1, Some(child)) => (child, PruneResult::Singleton),
            // If there's nothing, simply kill the node outright.
            (_, 0, None) => (Shared::null(), PruneResult::Null),
            // Many nodes (maybe somewhere below) ‒ someone must have inserted in between. But
            // we've already condemned this node, so create a new one and do the replacement.
            _ => {
                let new = Owned::new(new_child).into_shared(pin);
                // Note: we don't store Owned, because we may link it in. If we panicked before
                // disarming it, it would delete something linked in, which is bad. Instead, we
                // prefer deleting manually after the fact.
                cleanup = Some(new);
                (new, PruneResult::Copy)
            }
        };

        assert_eq!(
            0,
            child.tag(),
            "Attempt to replace condemned pointer or prune data node"
        );
        // Orderings: We need to publish the new node. We don't need to acquire the previous value
        // to destroy, because we already have it in case of success and we don't care about it on
        // failure.
        let result = parent
            .compare_and_set(child, insert, (Ordering::Release, Ordering::Relaxed), pin)
            .is_ok();
        if result {
            // We successfully unlinked the old child, so it's time to destroy it (as soon as
            // nobody is looking at it).
            pin.defer_destroy(child);
            prune_result
        } else {
            // We have failed to insert, so we need to clean up after ourselves.
            drop(cleanup.map(|c| Shared::into_owned(c)));
            PruneResult::CasFail
        }
    }

    /// Inner implementation of traversing the tree, creating missing branches and doing
    /// *something* at the leaf.
    fn traverse<'s, 'p, 'r, F>(
        &'s self,
        mut state: TraverseState<C, F>,
        mode: TraverseMode,
        pin: &'p Guard,
    ) -> Option<ExistingOrNew<&'r C::Payload>>
    where
        's: 'r,
        'p: 'r,
        F: FnOnce(C::Key) -> C::Payload,
    {
        let hash = self.hash(state.key());
        let mut shift = 0;
        let mut current = &self.root;
        let mut parent = None;
        loop {
            let node = current.load_consume(&pin);
            let flags = nf(node);

            let replace = |with: Owned<Inner>, delete_previous| {
                // If we fail to set it, the `with` is dropped together with the Err case, freeing
                // whatever was inside it.
                let result = current.compare_and_set_weak(
                    node,
                    with,
                    (Ordering::Release, Ordering::Relaxed),
                    pin,
                );
                match result {
                    Ok(new) if !node.is_null() && delete_previous => {
                        assert!(flags.contains(NodeFlags::DATA));
                        let node = Shared::from(node.as_raw() as usize as *const Data<C>);
                        unsafe { pin.defer_destroy(node) };
                        Some(new)
                    }
                    Ok(new) => Some(new),
                    Err(e) => {
                        if NodeFlags::from_bits(e.new.tag())
                            .expect("Invalid flags")
                            .contains(NodeFlags::DATA)
                        {
                            unsafe { drop_data::<C>(e.new.into_shared(&pin)) };
                        }
                        // Else → just let e drop and destroy the owned in there
                        None
                    }
                }
            };

            if flags.contains(NodeFlags::CONDEMNED) {
                // This one is going away. We are not allowed to modify the cell, we just have to
                // replace the inner node first. So, let's do some cleanup.
                //
                // TODO: In some cases we would not really *have* to do this (in particular, if we
                // just want to walk through and not modify it here at all, it's OK).
                unsafe {
                    let (parent, child) = parent.expect("Condemned the root!");
                    Self::prune(&pin, parent, child);
                }
                // Either us or someone else modified the tree on our path. In many cases we
                // could just continue here, but some cases are complex. For now, we just restart
                // the whole traversal and try from the start, for simplicity. This should be rare
                // anyway, so complicating the code further probably is not worth it.
                shift = 0;
                current = &self.root;
                parent = None;
            } else if node.is_null() {
                // Not found, create it.
                if let Some(new) = replace(state.data_owned(), true) {
                    if mode == TraverseMode::Overwrite {
                        return None;
                    } else {
                        let new = unsafe { load_data::<C>(new) };
                        return Some(ExistingOrNew::New(&new[0]));
                    }
                }
            // else -> retry
            } else if flags.contains(NodeFlags::DATA) {
                let data = unsafe { load_data::<C>(node) };
                assert!(!data.is_empty(), "Empty data nodes must not be kept around");
                if data[0].borrow() != state.key() && shift < mem::size_of_val(&hash) * 8 {
                    assert!(data.len() == 1, "Collision node not deep enough");
                    // There's one data node at this pointer, but we want to place a different one
                    // here too. So we create a new level, push the old one down. Note that we
                    // check both that we are adding something else & that we still have some more
                    // bits to distinguish by.

                    // We need to add another level. Note: there *still* might be a collision.
                    // Therefore, we just add the level and try again.
                    let other_hash = self.hash(data[0].borrow());
                    let other_bits = (other_hash >> shift) & LEVEL_MASK;
                    let mut inner = Inner::default();
                    inner.0[other_bits as usize] = Atomic::from(node);
                    let split = Owned::new(inner);
                    // No matter if it succeeds or fails, we try again. We'll either find the newly
                    // inserted value here and continue with another level down, or it gets
                    // destroyed and we try splitting again.
                    replace(split, false);
                } else {
                    // All the other cases:
                    // * It has the same key
                    // * There's already a collision on this level (because we've already run out of
                    //   bits previously).
                    // * We've run out of the hash bits so there's nothing to split by any more.
                    let mut result = data
                        .iter()
                        .find(|l| (*l).borrow().borrow() == state.key())
                        .map(ExistingOrNew::Existing);

                    if result.is_none() || mode == TraverseMode::Overwrite {
                        let mut new = Data::<C>::with_capacity(data.len() + 1);
                        new.extend(
                            data.iter()
                                .filter(|l| (*l).borrow() != state.key())
                                .cloned(),
                        );
                        new.push(state.payload());
                        new.shrink_to_fit();
                        let new = owned_data::<C>(new);
                        if let Some(new) = replace(new, true) {
                            if result.is_none() && mode == TraverseMode::IfMissing {
                                let new = unsafe { load_data::<C>(new) };
                                result = Some(ExistingOrNew::New(new.last().unwrap()));
                            }
                        } else {
                            continue;
                        }
                    }

                    return result;
                }
            } else {
                // An inner node, go one level deeper.
                let inner = unsafe { node.as_ref().expect("We just checked this is not NULL") };
                let bits = (hash >> shift) & LEVEL_MASK;
                shift += LEVEL_BITS;
                parent = Some((current, node));
                current = &inner.0[bits as usize];
            }
        }
    }

    /// Looks up a value.
    pub fn get<'r, 's, 'p, Q>(&'s self, key: &Q, pin: &'p Guard) -> Option<&'r C::Payload>
    where
        's: 'r,
        'p: 's,
        Q: ?Sized + Eq + Hash,
        C::Key: Borrow<Q>,
    {
        let mut current = &self.root;
        let mut hash = self.hash(key);
        loop {
            let node = current.load_consume(pin);
            let flags = nf(node);
            if node.is_null() {
                return None;
            } else if flags.contains(NodeFlags::DATA) {
                return unsafe { load_data::<C>(node) }
                    .iter()
                    .find(|l| (*l).borrow().borrow() == key);
            } else {
                let inner = unsafe { node.as_ref().expect("We just checked this is not NULL") };
                let bits = hash & LEVEL_MASK;
                hash >>= LEVEL_BITS;
                current = &inner.0[bits as usize];
            }
        }
    }

    /// Looks up a value or create (and insert) a new one.
    ///
    /// Either way, returns the value.
    pub fn get_or_insert_with<'s, 'p, 'r, F>(
        &'s self,
        key: C::Key,
        create: F,
        pin: &'p Guard,
    ) -> ExistingOrNew<&'r C::Payload>
    where
        's: 'r,
        'p: 'r,
        F: FnOnce(C::Key) -> C::Payload,
    {
        let state = TraverseState::Future {
            key,
            constructor: create,
        };
        self.traverse(state, TraverseMode::IfMissing, pin)
            .expect("Should have created one for me")
    }

    /// Removes a value identified by the key from the trie, returning it if it was found.
    pub fn remove<'r, 's, 'p, Q>(&'s self, key: &Q, pin: &'p Guard) -> Option<&'r C::Payload>
    where
        's: 'r,
        'p: 'r,
        Q: ?Sized + Eq + Hash,
        C::Key: Borrow<Q>,
    {
        let mut current = &self.root;
        let hash = self.hash(key);
        let mut shift = 0;
        let mut levels: ArrayVec<[_; MAX_LEVELS]> = ArrayVec::new();
        let deleted = loop {
            let node = current.load_consume(&pin);
            let flags = nf(node);
            let replace = |with: Shared<_>| {
                let result = current.compare_and_set_weak(
                    node,
                    with,
                    (Ordering::Release, Ordering::Relaxed),
                    &pin,
                );
                match result {
                    Ok(_) => {
                        assert!(flags.contains(NodeFlags::DATA));
                        unsafe {
                            let node = Shared::from(node.as_raw() as usize as *const Data<C>);
                            pin.defer_destroy(node);
                        }
                        true
                    }
                    Err(ref e) if !e.new.is_null() => {
                        assert!(nf(e.new).contains(NodeFlags::DATA));
                        unsafe { drop_data::<C>(e.new) };
                        false
                    }
                    Err(_) => false,
                }
            };

            if node.is_null() {
                // Nothing to delete, so just give up (without pruning).
                return None;
            } else if flags.contains(NodeFlags::CONDEMNED) {
                unsafe {
                    let (current, node) = levels.pop().expect("Condemned the root");
                    Self::prune(&pin, current, node);
                }
                // Retry by starting over from the top, for similar reasons to the one in
                // insert.
                levels.clear();
                shift = 0;
                current = &self.root;
            } else if flags.contains(NodeFlags::DATA) {
                let data = unsafe { load_data::<C>(node) };
                // Try deleting the thing.
                let mut deleted = None;
                let new = data
                    .iter()
                    .filter(|l| {
                        if (*l).borrow().borrow() == key {
                            deleted = Some(*l);
                            false
                        } else {
                            true
                        }
                    })
                    .cloned()
                    .collect::<Data<C>>();

                if deleted.is_some() {
                    let new = if new.is_empty() {
                        Shared::null()
                    } else {
                        owned_data::<C>(new).into_shared(&pin)
                    };
                    if !replace(new) {
                        continue;
                    }
                }

                break deleted;
            } else {
                let inner = unsafe { node.as_ref().expect("We just checked for NULL") };
                levels.push((current, node));
                let bits = (hash >> shift) & LEVEL_MASK;
                shift += LEVEL_BITS;
                current = &inner.0[bits as usize];
            }
        };

        // Go from the top and try to clean up.
        if deleted.is_some() {
            for (parent, child) in levels.into_iter().rev() {
                let inner = unsafe { child.as_ref().expect("We just checked for NULL") };

                // This is an optimisation ‒ replacing the thing is expensive, so we want to check
                // first (which is cheaper).
                let non_null = inner
                    .0
                    .iter()
                    .filter(|ptr| !ptr.load(Ordering::Relaxed, &pin).is_null())
                    .count();
                if non_null > 1 {
                    // No reason to go into the upper levels.
                    break;
                }

                // OK, we think we could remove this node. Try doing so.
                if let PruneResult::Copy = unsafe { Self::prune(&pin, parent, child) } {
                    // Even though we tried to count how many pointers there are, someone must have
                    // added some since. So there's no way we can prone anything higher up and we
                    // give up.
                    break;
                }
                // Else:
                // Just continue with higher levels. Even if someone made the contraction for
                // us, it should be safe to do so.
            }
        }

        deleted
    }
}

impl<C: Config, S> Raw<C, S> {
    /// Checks for emptiness.
    pub fn is_empty(&self) -> bool {
        // This relies on proper branch pruning.
        // We can use the unprotected here, because we are not actually interested in where the
        // pointer points to. Therefore we can also use the Relaxed ordering.
        unsafe {
            self.root
                .load(Ordering::Relaxed, &crossbeam_epoch::unprotected())
                .is_null()
        }
    }

    /// Access to the hash builder.
    pub fn hash_builder(&self) -> &S {
        &self.hash_builder
    }
}

impl<C: Config, S> Drop for Raw<C, S> {
    fn drop(&mut self) {
        /*
         * Notes about unsafety here:
         * * We are in a destructor and that one is &mut self. There are no concurrent accesses to
         *   this data structure any more, therefore we can safely assume we are the only ones
         *   looking at the pointers inside.
         * * Therefore, using unprotected is also fine.
         * * Similarly, the Relaxed ordering here is fine too, as the whole data structure must
         *   have been synchronized into our thread already by this time.
         * * The pointer inside this data structure is never dangling.
         */
        unsafe fn drop_recursive<C: Config>(node: &Atomic<Inner>) {
            let pin = crossbeam_epoch::unprotected();
            let extract = node.load(Ordering::Relaxed, &pin);
            let flags = nf(extract);
            if extract.is_null() {
                // Skip
            } else if flags.contains(NodeFlags::DATA) {
                drop_data::<C>(extract);
            } else {
                let owned = extract.into_owned();
                for sub in &owned.0 {
                    drop_recursive::<C>(sub);
                }
                drop(owned);
            }
        }
        unsafe { drop_recursive::<C>(&self.root) };
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use std::ptr;

    use super::config::Trivial as TrivialConfig;
    use super::*;

    // A hasher to create collisions on purpose. Let's make the hash trie into a glorified array.
    // We allow tests in higher-level modules to reuse it for their tests.
    pub(crate) struct NoHasher;

    impl Hasher for NoHasher {
        fn finish(&self) -> u64 {
            0
        }

        fn write(&mut self, _: &[u8]) {}
    }

    impl BuildHasher for NoHasher {
        type Hasher = NoHasher;

        fn build_hasher(&self) -> NoHasher {
            NoHasher
        }
    }

    #[derive(Clone, Copy, Debug, Default)]
    pub(crate) struct SplatHasher(u64);

    impl Hasher for SplatHasher {
        fn finish(&self) -> u64 {
            self.0
        }
        fn write(&mut self, value: &[u8]) {
            for val in value {
                for idx in 0..mem::size_of::<u64>() {
                    self.0 ^= (*val as u64) << 8 * idx;
                }
            }
        }
    }

    pub(crate) struct MakeSplatHasher;

    impl BuildHasher for MakeSplatHasher {
        type Hasher = SplatHasher;

        fn build_hasher(&self) -> SplatHasher {
            SplatHasher::default()
        }
    }

    /// Tests the test hasher.
    ///
    /// Because it was giving us some trouble ☹
    #[test]
    fn splat_hasher() {
        let mut hasher = MakeSplatHasher.build_hasher();
        hasher.write_u8(0);
        assert_eq!(0, hasher.finish());
        hasher.write_u8(8);
        assert_eq!(0x0808080808080808, hasher.finish());
    }

    #[test]
    fn consts_consistent() {
        assert!(LEVEL_CELLS.is_power_of_two());
        assert_eq!(LEVEL_BITS, LEVEL_MASK.count_ones() as usize);
        assert_eq!(LEVEL_BITS, (!LEVEL_MASK).trailing_zeros() as usize);
        assert_eq!(LEVEL_CELLS, 2usize.pow(LEVEL_BITS as u32));
    }

    /// Pretend something left a condemned marker on one of the nodes when we insert. This will get
    /// cleaned up.
    ///
    /// And yes, the test abuses the fact that it knows how the specific hasher works and
    /// distributes the given values.
    #[test]
    fn prune_on_insert() {
        let mut map = Raw::<TrivialConfig<u8>, _>::with_hasher(MakeSplatHasher);
        let pin = crossbeam_epoch::pin();
        for i in 0..LEVEL_CELLS as u8 {
            assert!(map.insert(i, &pin).is_none());
        }

        eprintln!("{}", debug::PrintShape(&map));

        // By now, we should have exactly one data node under each pointer under root. Sanity
        // check that (Relaxed is fine, we are in a single threaded test).
        let root = map.root.load(Ordering::Relaxed, &pin);
        let flags = nf(root);
        assert_eq!(
            NodeFlags::empty(),
            flags,
            "Root should be non-condemned inner node"
        );
        assert!(!root.is_null());
        let old_root = root.as_raw();
        let root = unsafe { root.deref() };

        for ptr in &root.0 {
            let ptr = ptr.load(Ordering::Relaxed, &pin);
            assert!(!ptr.is_null());
            let flags = nf(ptr);
            assert_eq!(
                NodeFlags::DATA,
                flags,
                "Expected a data node, found {:?}",
                ptr
            );
        }

        // Now, *start* condemning the node. Mark the first slot, the one we'll eventually use.
        root.0[0].fetch_or(NodeFlags::CONDEMNED.bits(), Ordering::Relaxed, &pin);

        // This touches the condemned slot, so it should trigger fixing stuff.
        let old = map.insert(0, &pin);
        assert_eq!(0, *old.unwrap());

        // The condemned flag must have disappeared by now.
        map.assert_pruned();

        // And the root should have changed for a brand new one.
        let new_root = map.root.load(Ordering::Relaxed, &pin).as_raw();
        assert!(!ptr::eq(old_root, new_root), "Condemned node not replaced");

        // But all the content is preserved
        for i in 0..LEVEL_CELLS as u8 {
            assert_eq!(i, *map.get(&i, &pin).unwrap());
        }
    }

    /// Creates an effectively empty map with a leftover (unpruned) but condemned node.
    ///
    /// As the algorithm goes, almost everyone who finds it is responsible for cleaning it up.
    fn with_leftover() -> Raw<TrivialConfig<u8>, MakeSplatHasher> {
        let map = Raw::<TrivialConfig<u8>, _>::with_hasher(MakeSplatHasher);
        let pin = crossbeam_epoch::pin();

        let i = Inner::default();
        i.0[0].fetch_or(NodeFlags::CONDEMNED.bits(), Ordering::Relaxed, &pin);
        map.root.store(Owned::new(i), Ordering::Relaxed);

        // There's nothing in this map effectively, but it doesn't claim to be empty due to the
        // non-null pointer.
        assert!(iterator::Iter::new(&map).next().is_none());
        assert!(!map.is_empty());

        map
    }

    /// Similar as the above, but with empty condemned node.
    ///
    /// Here we put a fake node somewhere into the aether, make it condemned and see how it
    /// disappears on insertion.
    #[test]
    fn prune_on_insert_empty() {
        let mut map = with_leftover();
        let pin = crossbeam_epoch::pin();
        let old_root = map.root.load(Ordering::Relaxed, &pin).as_raw();

        // Now, let's insert something so it meets the condemned mark
        assert!(map.insert(0, &pin).is_none());

        map.assert_pruned();
        let new_root = map.root.load(Ordering::Relaxed, &pin);
        // It got replaced and the root is directly the data node
        let new_flags = nf(new_root);
        assert_eq!(NodeFlags::DATA, new_flags);
        assert!(
            !ptr::eq(old_root, new_root.as_raw()),
            "Condemned node not replaced"
        );
    }

    /// Test that if someone left a un-pruned node and remove finds it, it gets rid of it (even in
    /// cases it does not actually remove anything in particular).
    #[test]
    fn prune_on_remove() {
        let map = Raw::<TrivialConfig<u8>, _>::with_hasher(MakeSplatHasher);
        let pin = crossbeam_epoch::pin();

        let i_inner = Inner::default();
        let i_outer = Inner::default();
        i_outer.0[0].store(
            Owned::new(i_inner).with_tag(NodeFlags::CONDEMNED.bits()),
            Ordering::Relaxed,
        );
        map.root.store(Owned::new(i_outer), Ordering::Relaxed);

        // There's nothing in this map effectively, but it doesn't claim to be empty due to the
        // non-null pointer.
        assert!(iterator::Iter::new(&map).next().is_none());
        assert!(!map.is_empty());

        assert!(map.remove(&0, &pin).is_none());

        eprintln!("{}", debug::PrintShape(&map));

        assert_eq!(0, map.root.load(Ordering::Relaxed, &pin).tag());
        // Note: it is still *not* properly pruned. The inner node should have a thread it'll clean
        // up later on. And we can't contract it as the one below is inner node, not data node.
    }
}
