#![allow(dead_code)] // Allow the unused structs

//! Compile fail tests
//!
//! Implemented in a minimal way, as doc tests in a hidden module.

/// ```compile_fail
/// use std::rc::Rc;
///
/// use contrie::ConMap;
/// use crossbeam_utils::thread;
///
/// let map: ConMap<usize, Rc<usize>> = ConMap::new();
///
/// thread::scope(|s| {
///     s.spawn(|_| {
///         drop(map);
///     });
/// }).unwrap();
/// ```
///
/// Similar one, but with Arc should work fine, though.
///
/// ```
/// use std::sync::Arc;
///
/// use contrie::ConMap;
/// use crossbeam_utils::thread;
///
/// let map: ConMap<usize, Arc<usize>> = ConMap::new();
///
/// thread::scope(|s| {
///     s.spawn(|_| {
///         drop(map);
///     });
/// }).unwrap();
/// ```
struct ShouldNotBeSend;

/// ```compile_fail
/// use std::rc::Rc;
///
/// use contrie::ConMap;
/// use crossbeam_utils::thread;
///
/// let map: ConMap<usize, Rc<usize>> = ConMap::new();
///
/// thread::scope(|s| {
///     s.spawn(|_| {
///         map.get(&42);
///     });
/// }).unwrap();
/// ```
///
/// Similar one, but with Arc should work fine, though.
///
/// ```
/// use std::sync::Arc;
///
/// use contrie::ConMap;
/// use crossbeam_utils::thread;
///
/// let map: ConMap<usize, Arc<usize>> = ConMap::new();
///
/// thread::scope(|s| {
///     s.spawn(|_| {
///         map.get(&42);
///     });
/// }).unwrap();
/// ```
struct ShouldNotSync;

/// ```compile_fail
/// use std::collections::hash_map::RandomState;
///
/// use contrie::raw::config::Trivial;
/// use contrie::raw::Raw;
///
/// let map: Raw<Trivial<usize>, RandomState> = Raw::with_hasher(RandomState::default());
/// let pin = crossbeam_epoch::pin();
/// let element = map.get(&42, &pin);
/// drop(map);
/// // Must not outlive the map
/// assert!(element.is_none());
/// ```
///
/// This one should be fine, as we don't drop the map.
///
/// ```
/// use std::collections::hash_map::RandomState;
///
/// use contrie::raw::config::Trivial;
/// use contrie::raw::Raw;
///
/// let map: Raw<Trivial<usize>, RandomState> = Raw::with_hasher(RandomState::default());
/// let pin = crossbeam_epoch::pin();
/// let element = map.get(&42, &pin);
/// assert!(element.is_none());
/// ```
struct CantExtendBeyondDestroy;

/// Dropping map makes it impossible to borrow.
///
/// ```compile_fail
/// use std::collections::hash_map::RandomState;
///
/// use contrie::raw::config::Trivial;
/// use contrie::raw::iterator::Iter;
/// use contrie::raw::Raw;
///
/// let map: Raw<Trivial<usize>, RandomState> = Raw::with_hasher(RandomState::default());
/// let mut iter = Iter::new(&map);
/// let element = iter.next();
/// drop(map);
/// // Must not outlive the map
/// assert!(element.is_none());
/// ```
///
/// We are not allowed to drop the iterator either.
///
/// ```compile_fail
/// use std::collections::hash_map::RandomState;
///
/// use contrie::raw::config::Trivial;
/// use contrie::raw::iterator::Iter;
/// use contrie::raw::Raw;
///
/// let map: Raw<Trivial<usize>, RandomState> = Raw::with_hasher(RandomState::default());
/// let mut iter = Iter::new(&map);
/// let element = iter.next();
/// drop(iter);
/// // Must not outlive the iterator
/// assert!(element.is_none());
/// ```
///
/// But if we don't drop anything, everything is fine.
/// ```
/// use std::collections::hash_map::RandomState;
///
/// use contrie::raw::config::Trivial;
/// use contrie::raw::iterator::Iter;
/// use contrie::raw::Raw;
///
/// let map: Raw<Trivial<usize>, RandomState> = Raw::with_hasher(RandomState::default());
/// let mut iter = Iter::new(&map);
/// let element = iter.next();
/// assert!(element.is_none());
/// ```
struct DoesntOutliveIterator;
