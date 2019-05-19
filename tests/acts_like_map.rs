//! In these tests, we make sure the ConTrie works as a HashMap in single threaded context, and
//! sometimes in multithreaded too.
//!
//! To do that we simply generate a series of inserts, lookups and deletions and try them on both
//! maps. They need to return the same things.
//!
//! Furthermore, each test is run in several instances, with keys in differently sized universe.
//! The small ones likely generate only short hashes, but are more likely to reuse the same value.

use std::collections::{HashMap, HashSet};
use std::fmt::Debug;
use std::hash::Hash;

use contrie::ConMap;
use proptest::collection::vec;
use proptest::prelude::*;
use rayon::prelude::*;

#[derive(Debug, Clone)]
enum Instruction<K, V> {
    Lookup(K),
    Remove(K),
    Insert(K, V),
}

impl<K, V> Instruction<K, V>
where
    K: Arbitrary + Clone + Debug + Eq + Hash,
    V: Arbitrary + Clone + Debug + PartialEq,
{
    fn strategy() -> impl Strategy<Value = Self> {
        use Instruction::*;

        prop_oneof![
            any::<K>().prop_map(Lookup),
            any::<K>().prop_map(Remove),
            any::<(K, V)>().prop_map(|(k, v)| Insert(k, v)),
        ]
    }

    fn run(instructions: Vec<Self>) -> Result<(), TestCaseError> {
        use Instruction::*;

        let trie = ConMap::new();
        let mut map = HashMap::new();
        for ins in instructions {
            match ins {
                Lookup(key) => {
                    let expected = map.get(&key);
                    let found = trie.get(&key);
                    prop_assert_eq!(expected, found.as_ref().map(|l| l.value()));
                }
                Remove(key) => {
                    let expected = map.remove(&key);
                    let found = trie.remove(&key);
                    prop_assert_eq!(expected.as_ref(), found.as_ref().map(|l| l.value()));
                }
                Insert(key, value) => {
                    let expected = map.insert(key.clone(), value.clone());
                    let found = trie.insert(key, value);
                    prop_assert_eq!(expected.as_ref(), found.as_ref().map(|l| l.value()));
                }
            }
        }

        Ok(())
    }
}

fn insert_parallel_test<T: Clone + Hash + Eq + Send + Sync>(
    values: Vec<T>,
) -> Result<(), TestCaseError> {
    let set: HashSet<_> = values.iter().cloned().collect();
    let trie = ConMap::new();
    values.into_par_iter().for_each(|v| {
        trie.insert(v, ());
    });
    for v in set {
        prop_assert!(trie.get(&v).is_some());
    }

    Ok(())
}

proptest! {
    #[test]
    fn small_keys(instructions in vec(Instruction::<u8, usize>::strategy(), 1..10_000)) {
        Instruction::run(instructions)?;
    }

    #[test]
    fn mid_keys(instructions in vec(Instruction::<u16, usize>::strategy(), 1..10_000)) {
        Instruction::run(instructions)?;
    }

    #[test]
    fn large_keys(instructions in vec(Instruction::<usize, usize>::strategy(), 1..10_000)) {
        Instruction::run(instructions)?;
    }

    #[test]
    fn string_keys(instructions in vec(Instruction::<String, usize>::strategy(), 1..100)) {
        Instruction::run(instructions)?;
    }

    // TODO: This test and following needs improvements.
    // 1. We need actual ConSet.
    // 2. We need iteration.
    #[test]
    fn insert_all_large(values in vec(any::<usize>(), 1..10_000)) {
        // Make them unique
        let set: HashSet<_> = values.iter().cloned().collect();
        let trie = ConMap::new();
        for v in values {
            trie.insert(v, ());
        }
        for v in set {
            prop_assert!(trie.get(&v).is_some());
        }
    }

    #[test]
    fn insert_all_small_parallel(values in vec(any::<u8>(), 1..10_000)) {
        insert_parallel_test(values)?;
    }

    #[test]
    fn insert_all_mid_parallel(values in vec(any::<u16>(), 1..10_000)) {
        insert_parallel_test(values)?;
    }

    #[test]
    fn insert_all_large_parallel(values in vec(any::<usize>(), 1..10_000)) {
        insert_parallel_test(values)?;
    }
}
