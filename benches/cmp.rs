#![feature(test)]

extern crate test;

use std::iter;

use rand::prelude::*;

fn vals(cnt: usize) -> Vec<usize> {
    iter::repeat_with(random).take(cnt).collect()
}

macro_rules! typed_bench {
    ($name: ident, $type: ty) => {
        mod $name {
            use test::{black_box, Bencher};

            use super::vals;

            fn fill(cnt: usize) -> ($type, Vec<usize>) {
                let vals = vals(cnt);

                let map = vals.iter().cloned().map(|i| (i, i)).collect();
                (map, vals)
            }

            fn lookup_n(cnt: usize, bencher: &mut Bencher) {
                let (map, mut to_lookup) = fill(cnt);
                to_lookup.drain(50..);
                to_lookup.extend(vals(50).into_iter());

                bencher.iter(|| {
                    for val in &to_lookup {
                        black_box(map.get(&val));
                    }
                });
            }

            #[bench]
            fn lookup_small(bencher: &mut Bencher) {
                lookup_n(100, bencher);
            }

            #[bench]
            fn lookup_mid(bencher: &mut Bencher) {
                lookup_n(10_000, bencher);
            }

            #[bench]
            fn lookup_huge(bencher: &mut Bencher) {
                lookup_n(10_000_000, bencher);
            }
        }
    };
}

typed_bench!(hash_map, std::collections::HashMap<usize, usize>);
typed_bench!(btree_map, std::collections::BTreeMap<usize, usize>);
typed_bench!(contrie_map, contrie::ConMap<usize, usize>);
