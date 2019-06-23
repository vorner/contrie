use std::collections::hash_map::RandomState;
use std::collections::HashSet;
use std::hash::{BuildHasher, Hash};

use proptest::collection::vec;
use proptest::prelude::*;
use rayon::prelude::*;

use crate::raw::tests::{MakeSplatHasher, NoHasher};
use crate::ConSet;

fn insert_parallel_test<
    T: Clone + Hash + Eq + Send + Sync + 'static,
    H: BuildHasher + Send + Sync,
>(
    values: Vec<T>,
    hasher: H,
) -> Result<(), TestCaseError> {
    let set: HashSet<_> = values.iter().cloned().collect();
    let trie = ConSet::with_hasher(hasher);
    values.into_par_iter().for_each(|v| {
        trie.insert(v);
    });
    for v in set {
        prop_assert!(trie.contains(&v));
    }

    Ok(())
}

proptest! {
    #[test]
    fn insert_all_large(values in vec(any::<usize>(), 1..10_000)) {
        // Make them unique
        let set: HashSet<_> = values.iter().cloned().collect();
        let trie = ConSet::new();
        for v in values {
            trie.insert(v);
        }
        for v in &trie {
            prop_assert!(set.contains(&v));
        }
        for v in set {
            prop_assert!(trie.get(&v).is_some());
        }
    }

    #[test]
    fn insert_all_small_parallel(values in vec(any::<u8>(), 1..10_000)) {
        insert_parallel_test(values, RandomState::default())?;
    }

    #[test]
    fn insert_all_mid_parallel(values in vec(any::<u16>(), 1..10_000)) {
        insert_parallel_test(values, RandomState::default())?;
    }

    #[test]
    fn insert_all_mid_parallel_nohash(values in vec(any::<u16>(), 1..100)) {
        insert_parallel_test(values, NoHasher)?;
    }

    #[test]
    fn insert_all_mid_parallel_bad_hasher(values in vec(any::<u16>(), 1..1_000)) {
        insert_parallel_test(values, MakeSplatHasher)?;
    }

    #[test]
    fn insert_all_large_parallel(values in vec(any::<usize>(), 1..10_000)) {
        insert_parallel_test(values, RandomState::default())?;
    }
}
