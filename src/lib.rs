use std::borrow::Borrow;
use std::collections::hash_map::RandomState;
use std::hash::{BuildHasher, Hash, Hasher};
use std::marker::PhantomData;
use std::mem;
use std::ops::Deref;
use std::sync::atomic::Ordering;
use std::sync::Arc;

use arrayvec::ArrayVec;
use bitflags::bitflags;
use crossbeam_epoch::{Atomic, Guard, Owned, Shared};
use smallvec::SmallVec;

// TODO: Rename Leaf. It's a public name and a bit silly/leaks implementation details.
// TODO: Make this whole type private and implement better/all APIs around it? Maybe make it
// customizable even more ‒ synchronization, keys other than hashes (arbitrary byte strings?),
// copy/clone directly instead of storing just the key. But certainly different things for sets (we
// want the whole API to be Arc<K>, not Arc<Node>).
// TODO: Iterators (from, into, extend)
// TODO: Rayon support (from and into parallel iterator, extend) under a feature flag.
// TODO: Distinguish between the leaf and inner node by tag on the pointer.
// TODO: Valgrind into the CI
// FIXME: We seem to be leaking somewhere. Where?!

// All directly written, some things are not const fn yet :-(. But tested below.
const LEVEL_BITS: usize = 4;
const LEVEL_MASK: u64 = 0b1111;
const LEVEL_CELLS: usize = 16;
const MAX_LEVELS: usize = mem::size_of::<u64>() * 8 / LEVEL_BITS;

// TODO: Checks that we really do have the bits in the alignment.
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
        /// TODO: Describe the trick better.
        const DATA = 0b10;
    }
}

fn nf(node: Shared<Inner>) -> NodeFlags {
    NodeFlags::from_bits(node.tag()).expect("Invalid node flags")
}

unsafe fn load_data<'a, K: 'a, V: 'a>(node: Shared<'a, Inner>) -> &'a Data<K, V> {
    assert!(
        nf(node).contains(NodeFlags::DATA),
        "Tried to load data from inner node pointer"
    );
    (node.as_raw() as usize as *const Data<K, V>)
        .as_ref()
        .expect("A null pointer with data flag found")
}

fn owned_data<K, V>(data: Data<K, V>) -> Owned<Inner> {
    unsafe {
        Owned::<Inner>::from_raw(Box::into_raw(Box::new(data)) as usize as *mut _)
            .with_tag(NodeFlags::DATA.bits())
    }
}

#[derive(Default)]
struct Inner([Atomic<Inner>; LEVEL_CELLS]);

#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct Leaf<K, V> {
    data: (K, V),
}

impl<K, V> Leaf<K, V> {
    pub fn new(key: K, value: V) -> Self {
        Self { data: (key, value) }
    }

    pub fn key(&self) -> &K {
        &self.data.0
    }

    pub fn value(&self) -> &V {
        &self.data.1
    }
}

impl<K, V> Deref for Leaf<K, V> {
    type Target = (K, V);
    fn deref(&self) -> &(K, V) {
        &self.data
    }
}

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
type Data<K, V> = SmallVec<[Arc<Leaf<K, V>>; 2]>;

enum TraverseState<K, V, F> {
    Empty, // Invalid temporary state.
    Created(Arc<Leaf<K, V>>),
    Future { key: K, constructor: F },
}

impl<K, V, F: FnOnce() -> V> TraverseState<K, V, F> {
    fn key(&self) -> &K {
        match self {
            TraverseState::Empty => unreachable!("Not supposed to live in the empty state"),
            TraverseState::Created(leaf) => leaf.key(),
            TraverseState::Future { key, .. } => key,
        }
    }
    fn leaf(&mut self) -> Arc<Leaf<K, V>> {
        let (new_val, result) = match mem::replace(self, TraverseState::Empty) {
            TraverseState::Empty => unreachable!("Not supposed to live in the empty state"),
            TraverseState::Created(leaf) => (TraverseState::Created(Arc::clone(&leaf)), leaf),
            TraverseState::Future { key, constructor } => {
                let value = constructor();
                let leaf = Arc::new(Leaf::new(key, value));
                let created = TraverseState::Created(Arc::clone(&leaf));
                (created, leaf)
            }
        };
        *self = new_val;
        result
    }
    fn data_owned(&mut self) -> Owned<Inner> {
        let mut data = Data::new();
        data.push(self.leaf());
        owned_data(data)
    }
    fn into_leaf(self) -> Arc<Leaf<K, V>> {
        match self {
            TraverseState::Created(leaf) => leaf,
            TraverseState::Future { key, constructor } => Arc::new(Leaf::new(key, constructor())),
            TraverseState::Empty => unreachable!("Not supposed to live in the empty state"),
        }
    }
    fn into_return(self, mode: TraverseMode) -> Option<Arc<Leaf<K, V>>> {
        match mode {
            TraverseMode::Overwrite => None,
            TraverseMode::IfMissing => Some(self.into_leaf()),
        }
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

pub struct ConMap<K, V, S = RandomState> {
    hash_builder: S,
    root: Atomic<Inner>,
    _data: PhantomData<Leaf<K, V>>,
}

impl<K: Eq + Hash, V> ConMap<K, V, RandomState> {
    pub fn new() -> Self {
        Self::default()
    }
}

impl<K, V, S> ConMap<K, V, S>
where
    K: Eq + Hash,
    S: BuildHasher,
{
    pub fn with_hasher(hash_builder: S) -> Self {
        Self {
            hash_builder,
            root: Atomic::null(),
            _data: PhantomData,
        }
    }

    fn hash<Q>(&self, key: &Q) -> u64
    where
        Q: ?Sized + Hash,
    {
        let mut hasher = self.hash_builder.build_hasher();
        key.hash(&mut hasher);
        hasher.finish()
    }

    pub fn insert(&self, key: K, value: V) -> Option<Arc<Leaf<K, V>>> {
        self.insert_leaf(Arc::new(Leaf::new(key, value)))
    }

    pub fn insert_leaf(&self, leaf: Arc<Leaf<K, V>>) -> Option<Arc<Leaf<K, V>>> {
        self.traverse(
            // Any way to do it without the type parameters here? Older rustc doesn't like them.
            TraverseState::<K, V, fn() -> V>::Created(leaf),
            TraverseMode::Overwrite,
        )
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
            // Acquire ‒ we don't need the grandchild ourselves, only the pointer. But we'll need
            // to "republish" it through the parent pointer later on and for that we have to get it
            // first.
            //
            // FIXME: May we actually need SeqCst here to order it relative to the CAS below?
            let gc = grandchild.fetch_or(NodeFlags::CONDEMNED.bits(), Ordering::Acquire, pin);
            // The flags we insert into the new one should not contain condemned flag even if it
            // was already present here.
            let flags = nf(gc) & !NodeFlags::CONDEMNED;
            let gc = gc.with_tag(flags.bits());
            if gc.is_null() {
                // Do nothing, just skip
            } else if flags.contains(NodeFlags::DATA) {
                last_leaf.replace(gc);
                child_cnt += 1;
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

        // FIXME: This is probably reachable from remove, right?
        assert_eq!(0, child.tag(), "Attempt to replace condemned pointer");
        // Orderings: We need to publish the new node. We don't need to acquire the previous value
        // to destroy, because we already have it in case of success and we don't care about it on
        // failure.
        let result = parent
            .compare_and_set_weak(child, insert, (Ordering::Release, Ordering::Relaxed), pin)
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

    fn traverse<F>(
        &self,
        mut state: TraverseState<K, V, F>,
        mode: TraverseMode,
    ) -> Option<Arc<Leaf<K, V>>>
    where
        F: FnOnce() -> V,
    {
        let hash = self.hash(state.key());
        let mut shift = 0;
        let mut current = &self.root;
        let mut parent = None;
        let pin = crossbeam_epoch::pin();
        loop {
            let node = current.load(Ordering::Acquire, &pin);
            let flags = nf(node);

            let replace = |with, delete_previous| {
                // If we fail to set it, the `with` is dropped together with the Err case, freeing
                // whatever was inside it.
                let result = current
                    .compare_and_set_weak(node, with, Ordering::Release, &pin)
                    .is_ok();
                if result && !node.is_null() && delete_previous {
                    assert!(flags.contains(NodeFlags::DATA));
                    let node = Shared::from(node.as_raw() as usize as *const Data<K, V>);
                    unsafe { pin.defer_destroy(node) };
                }
                result
            };

            if flags.contains(NodeFlags::CONDEMNED) {
                // This one is going away. We are not allowed to modify the cell, we just have to
                // replace the inner node first. So, let's do some cleanup.
                //
                // TODO: In some cases we would not really *have* to do this (in particular, if we
                // just want to walk through and not modify it here at all, it's OK).
                unsafe {
                    Self::prune(&pin, parent.expect("Condemned the root!"), node);
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
                if replace(state.data_owned(), true) {
                    return state.into_return(mode);
                }
            // else -> retry
            } else if flags.contains(NodeFlags::DATA) {
                let data: &Data<K, V> = unsafe { load_data(node) };
                if data.len() == 1
                    && data[0].key() != state.key()
                    && shift < mem::size_of_val(&hash) * 8
                {
                    // There's one data node at this pointer, but we want to place a different one
                    // here too. So we create a new level, push the old one down. Note that we
                    // check both that we are adding something else & that we still have some more
                    // bits to distinguish by.

                    // We need to add another level. Note: there *still* might be a collision.
                    // Therefore, we just add the level and try again.
                    let other_hash = self.hash(data[0].key());
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
                    let old = data
                        .iter()
                        .find(|l| l.key().borrow() == state.key())
                        .map(Arc::clone);

                    if old.is_none() || mode == TraverseMode::Overwrite {
                        let mut new = Data::with_capacity(data.len() + 1);
                        new.extend(data.iter().filter(|l| l.key() != state.key()).cloned());
                        new.push(state.leaf());
                        new.shrink_to_fit();
                        let new = owned_data(new);
                        if !replace(new, true) {
                            continue;
                        }
                    }

                    return old.or_else(|| state.into_return(mode));
                }
            } else {
                // An inner node, go one level deeper.
                let inner = unsafe { node.as_ref().expect("We just checked this is not NULL") };
                let bits = (hash >> shift) & LEVEL_MASK;
                shift += LEVEL_BITS;
                parent = Some(current);
                current = &inner.0[bits as usize];
            }
        }
    }

    pub fn get<Q>(&self, key: &Q) -> Option<Arc<Leaf<K, V>>>
    where
        Q: ?Sized + Eq + Hash,
        K: Borrow<Q>,
    {
        let mut current = &self.root;
        let mut hash = self.hash(key);
        let pin = crossbeam_epoch::pin();
        loop {
            let node = current.load(Ordering::Acquire, &pin);
            let flags = nf(node);
            if node.is_null() {
                return None;
            } else if flags.contains(NodeFlags::DATA) {
                return unsafe { load_data::<K, V>(node) }
                    .iter()
                    .find(|l| l.key().borrow() == key)
                    .cloned();
            } else {
                let inner = unsafe { node.as_ref().expect("We just checked this is not NULL") };
                let bits = hash & LEVEL_MASK;
                hash >>= LEVEL_BITS;
                current = &inner.0[bits as usize];
            }
        }
    }

    pub fn get_or_insert_with<F>(&self, key: K, create: F) -> Arc<Leaf<K, V>>
    where
        F: FnOnce() -> V,
    {
        let state = TraverseState::Future {
            key,
            constructor: create,
        };
        self.traverse(state, TraverseMode::IfMissing)
            .expect("Should have created one for me")
    }

    // TODO: Return a tuple if it inserted a new one or found existing.
    pub fn get_or_insert(&self, key: K, value: V) -> Arc<Leaf<K, V>> {
        self.get_or_insert_with(key, || value)
    }

    pub fn get_or_insert_default(&self, key: K) -> Arc<Leaf<K, V>>
    where
        V: Default,
    {
        self.get_or_insert_with(key, V::default)
    }

    pub fn remove<Q>(&self, key: &Q) -> Option<Arc<Leaf<K, V>>>
    where
        Q: ?Sized + Eq + Hash,
        K: Borrow<Q>,
    {
        let mut current = &self.root;
        let hash = self.hash(key);
        let pin = crossbeam_epoch::pin();
        let mut shift = 0;
        let mut levels: ArrayVec<[_; MAX_LEVELS]> = ArrayVec::new();
        let deleted = loop {
            let node = current.load(Ordering::Acquire, &pin);
            let flags = nf(node);
            let replace = |with| {
                let result = current
                    .compare_and_set_weak(node, with, Ordering::Release, &pin)
                    .is_ok();
                if result {
                    assert!(flags.contains(NodeFlags::DATA));
                    unsafe {
                        let node = Shared::from(node.as_raw() as usize as *const Data<K, V>);
                        pin.defer_destroy(node);
                    }
                }
                result
            };

            if node.is_null() {
                // Nothing to delete, so just give up (without pruning).
                return None;
            } else if flags.contains(NodeFlags::CONDEMNED) {
                unsafe {
                    Self::prune(&pin, &current, node);
                }
                // Retry by starting over from the top, for similar reasons to the one in
                // insert.
                levels.clear();
                shift = 0;
                current = &self.root;
                continue;
            } else if flags.contains(NodeFlags::DATA) {
                let data: &Data<K, V> = unsafe { load_data(node) };
                // Try deleting the thing.
                let mut deleted = None;
                let new = data
                    .iter()
                    .filter(|l| {
                        if l.key().borrow() == key {
                            deleted = Some(Arc::clone(l));
                            false
                        } else {
                            true
                        }
                    })
                    .cloned()
                    .collect::<Data<_, _>>();

                if deleted.is_some() {
                    let new = if new.is_empty() {
                        Shared::null()
                    } else {
                        owned_data(new).into_shared(&pin)
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

    // TODO: Iteration & friends
}

// Implementing manually, derive would ask for K, V: Default
impl<K, V, S> Default for ConMap<K, V, S>
where
    K: Eq + Hash,
    S: BuildHasher + Default,
{
    fn default() -> Self {
        Self::with_hasher(S::default())
    }
}

impl<K, V, S> Drop for ConMap<K, V, S> {
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
        unsafe fn drop_recursive<K, V>(node: &Atomic<Inner>) {
            let pin = crossbeam_epoch::unprotected();
            let extract = node.load(Ordering::Relaxed, &pin);
            let flags = nf(extract);
            if extract.is_null() {
                // Skip
            } else if flags.contains(NodeFlags::DATA) {
                let ptr = Owned::from_raw(extract.as_raw() as usize as *mut Data<K, V>);
                drop(ptr);
            } else {
                let owned = extract.into_owned();
                for sub in &owned.0 {
                    drop_recursive::<K, V>(sub);
                }
                drop(owned);
            }
        }
        unsafe { drop_recursive::<K, V>(&self.root) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crossbeam_utils::thread;

    const TEST_THREADS: usize = 4;
    const TEST_BATCH: usize = 10000;
    const TEST_BATCH_SMALL: usize = 100;
    const TEST_REP: usize = 20;

    #[test]
    fn consts_consistent() {
        assert_eq!(LEVEL_BITS, LEVEL_MASK.count_ones() as usize);
        assert_eq!(LEVEL_BITS, (!LEVEL_MASK).trailing_zeros() as usize);
        assert_eq!(LEVEL_CELLS, 2usize.pow(LEVEL_BITS as u32));
    }

    #[test]
    fn create_destroy() {
        let map: ConMap<String, usize> = ConMap::new();
        drop(map);
    }

    #[test]
    fn lookup_empty() {
        let map: ConMap<String, usize> = ConMap::new();
        assert!(map.get("hello").is_none());
    }

    #[test]
    fn insert_lookup() {
        let map = ConMap::new();
        assert!(map.insert("hello", "world").is_none());
        assert!(map.get("world").is_none());
        let found = map.get("hello").unwrap();
        assert_eq!(Leaf::new("hello", "world"), *found);
    }

    #[test]
    fn insert_overwrite_lookup() {
        let map = ConMap::new();
        assert!(map.insert("hello", "world").is_none());
        let old = map.insert("hello", "universe").unwrap();
        assert_eq!(Leaf::new("hello", "world"), *old);
        let found = map.get("hello").unwrap();
        assert_eq!(Leaf::new("hello", "universe"), *found);
    }

    // Insert a lot of things, to make sure we have multiple levels.
    #[test]
    fn insert_many() {
        let map = ConMap::new();
        for i in 0..TEST_BATCH * LEVEL_CELLS {
            assert!(map.insert(i, i).is_none());
        }

        for i in 0..TEST_BATCH * LEVEL_CELLS {
            assert_eq!(i, *map.get(&i).unwrap().value());
        }
    }

    #[test]
    fn par_insert_many() {
        for _ in 0..TEST_REP {
            let map: ConMap<usize, usize> = ConMap::new();
            thread::scope(|s| {
                for t in 0..TEST_THREADS {
                    let map = &map;
                    s.spawn(move |_| {
                        for i in 0..TEST_BATCH {
                            let num = t * TEST_BATCH + i;
                            assert!(map.insert(num, num).is_none());
                        }
                    });
                }
            })
            .unwrap();

            for i in 0..TEST_BATCH * TEST_THREADS {
                assert_eq!(*map.get(&i).unwrap().value(), i);
            }
        }
    }

    #[test]
    fn par_get_many() {
        for _ in 0..TEST_REP {
            let map = ConMap::new();
            for i in 0..TEST_BATCH * TEST_THREADS {
                assert!(map.insert(i, i).is_none());
            }
            thread::scope(|s| {
                for t in 0..TEST_THREADS {
                    let map = &map;
                    s.spawn(move |_| {
                        for i in 0..TEST_BATCH {
                            let num = t * TEST_BATCH + i;
                            assert_eq!(*map.get(&num).unwrap().value(), num);
                        }
                    });
                }
            })
            .unwrap();
        }
    }

    // A hasher to create collisions on purpose. Let's make the hash trie into a glorified array.
    struct NoHasher;

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

    #[test]
    fn collisions() {
        let map = ConMap::with_hasher(NoHasher);
        // While their hash is the same under the hasher, they don't kick each other out.
        for i in 0..TEST_BATCH_SMALL {
            assert!(map.insert(i, i).is_none());
        }
        // And all are present.
        for i in 0..TEST_BATCH_SMALL {
            assert_eq!(i, *map.get(&i).unwrap().value());
        }
        // But reusing the key kicks the other one out.
        for i in 0..TEST_BATCH_SMALL {
            assert_eq!(i, *map.insert(i, i + 1).unwrap().value());
            assert_eq!(i + 1, *map.get(&i).unwrap().value());
        }
    }

    #[test]
    fn get_or_insert_empty() {
        let map = ConMap::new();
        let val = map.get_or_insert("hello", 42);
        assert_eq!(42, *val.value());
        assert_eq!("hello", *val.key());
    }

    #[test]
    fn get_or_insert_existing() {
        let map = ConMap::new();
        assert!(map.insert("hello", 42).is_none());
        let val = map.get_or_insert("hello", 0);
        // We still have the original
        assert_eq!(42, *val.value());
        assert_eq!("hello", *val.key());
    }

    fn get_or_insert_many_inner<H: BuildHasher>(map: ConMap<usize, usize, H>, len: usize) {
        for i in 0..len {
            let val = map.get_or_insert(i, i);
            assert_eq!(i, *val.key());
            assert_eq!(i, *val.value());
        }

        for i in 0..len {
            let val = map.get_or_insert(i, 0);
            assert_eq!(i, *val.key());
            assert_eq!(i, *val.value());
        }
    }

    #[test]
    fn get_or_insert_many() {
        get_or_insert_many_inner(ConMap::new(), TEST_BATCH);
    }

    #[test]
    fn get_or_insert_collision() {
        get_or_insert_many_inner(ConMap::with_hasher(NoHasher), TEST_BATCH_SMALL);
    }

    #[test]
    fn simple_remove() {
        let map = ConMap::new();
        assert!(map.remove(&42).is_none());
        assert!(map.insert(42, "hello").is_none());
        assert_eq!("hello", *map.get(&42).unwrap().value());
        assert_eq!("hello", *map.remove(&42).unwrap().value());
        assert!(map.get(&42).is_none());
        assert!(map.is_empty());
        assert!(map.remove(&42).is_none());
        assert!(map.is_empty());
    }

    fn remove_many_inner<H: BuildHasher>(map: ConMap<usize, usize, H>, len: usize) {
        for i in 0..len {
            assert!(map.insert(i, i).is_none());
        }
        for i in 0..len {
            assert_eq!(i, *map.get(&i).unwrap().value());
            assert_eq!(i, *map.remove(&i).unwrap().value());
            assert!(map.get(&i).is_none());
        }

        assert!(map.is_empty());
    }

    #[test]
    fn remove_many() {
        remove_many_inner(ConMap::new(), TEST_BATCH);
    }

    #[test]
    fn remove_many_collision() {
        remove_many_inner(ConMap::with_hasher(NoHasher), TEST_BATCH_SMALL);
    }

    #[test]
    fn collision_remove_one_left() {
        let map = ConMap::with_hasher(NoHasher);
        map.insert(1, 1);
        map.insert(2, 2);

        fn find_data<'g>(
            map: &ConMap<usize, usize, NoHasher>,
            pin: &'g Guard,
        ) -> Option<&'g Data<usize, usize>> {
            let mut cur = &map.root;
            // Relaxed ‒ we are the only thread around
            loop {
                let node = cur.load(Ordering::Relaxed, &pin);
                assert!(!node.is_null());
                let flags = nf(node);
                if flags.contains(NodeFlags::DATA) {
                    return Some(unsafe { load_data(node) });
                } else {
                    let inner = unsafe { node.deref() };
                    cur = inner
                        .0
                        .iter()
                        .find(|c| !c.load(Ordering::Relaxed, pin).is_null())?;
                }
            }
        };

        {
            let pin = crossbeam_epoch::pin();
            assert_eq!(2, find_data(&map, &pin).expect("Leaf missing").len());
        }

        assert!(map.remove(&2).is_some());

        {
            let pin = crossbeam_epoch::pin();
            assert_eq!(1, find_data(&map, &pin).expect("Leaf missing").len());
        }

        assert!(map.remove(&1).is_some());

        assert!(map.is_empty());
    }
}

// TODO: Tests for correct dropping of values. And maybe add some canary values during tests?
