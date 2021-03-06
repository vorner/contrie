//! In these tests, we make sure the ConTrie works as a HashMap in single threaded context, and
//! sometimes in multithreaded too.
//!
//! To do that we simply generate a series of inserts, lookups and deletions and try them on both
//! maps. They need to return the same things.
//!
//! Furthermore, each test is run in several instances, with keys in differently sized universe.
//! The small ones likely generate only short hashes, but are more likely to reuse the same value.

use std::collections::hash_map::RandomState;
use std::collections::HashMap;
use std::fmt::Debug;
use std::hash::{BuildHasher, Hash};

use proptest::collection::vec;
use proptest::prelude::*;

use crate::raw::tests::{MakeSplatHasher, NoHasher};
use crate::ConMap;

#[derive(Debug, Clone)]
enum Instruction<K, V> {
    Lookup(K),
    Remove(K),
    Insert(K, V),
}

impl<K, V> Instruction<K, V>
where
    K: Arbitrary + Clone + Debug + Eq + Hash + 'static,
    V: Arbitrary + Clone + Debug + PartialEq + 'static,
{
    fn strategy() -> impl Strategy<Value = Self> {
        use Instruction::*;

        prop_oneof![
            any::<K>().prop_map(Lookup),
            any::<K>().prop_map(Remove),
            any::<(K, V)>().prop_map(|(k, v)| Insert(k, v)),
        ]
    }

    fn run<H: BuildHasher>(instructions: Vec<Self>, hasher: H) -> Result<(), TestCaseError> {
        use Instruction::*;

        let trie = ConMap::new();
        let mut map = HashMap::with_hasher(hasher);
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
                    prop_assert_eq!(map.is_empty(), trie.is_empty());
                }
                Insert(key, value) => {
                    let expected = map.insert(key.clone(), value.clone());
                    let found = trie.insert(key, value);
                    prop_assert_eq!(expected.as_ref(), found.as_ref().map(|l| l.value()));
                    assert!(!map.is_empty());
                }
            }
        }

        Ok(())
    }
}

// TODO: Do the same set of tests with some lousy hasher? One that hashes a bit, but has a lot of
// collisions?

proptest! {
    #[test]
    fn small_keys(instructions in vec(Instruction::<u8, usize>::strategy(), 1..10_000)) {
        Instruction::run(instructions, RandomState::default())?;
    }

    #[test]
    fn mid_keys_collisions(instructions in vec(Instruction::<u16, usize>::strategy(), 1..100)) {
        Instruction::run(instructions, NoHasher)?;
    }

    #[test]
    fn mid_keys_bad_hasher(instructions in vec(Instruction::<u16, usize>::strategy(), 1..1_000)) {
        Instruction::run(instructions, MakeSplatHasher)?;
    }

    #[test]
    fn mid_keys(instructions in vec(Instruction::<u16, usize>::strategy(), 1..10_000)) {
        Instruction::run(instructions, RandomState::default())?;
    }

    #[test]
    fn large_keys(instructions in vec(Instruction::<usize, usize>::strategy(), 1..10_000)) {
        Instruction::run(instructions, RandomState::default())?;
    }

    #[test]
    fn string_keys(instructions in vec(Instruction::<String, usize>::strategy(), 1..100)) {
        Instruction::run(instructions, RandomState::default())?;
    }
}
