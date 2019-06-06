mod existing_or_new;
pub mod clone_map;
pub mod map;
pub mod raw;
// Some integration-like tests live here, instead of crate/tests. This is because this allows cargo
// to compile them in parallel with the crate and also run them more in parallel. And I like to get
// all the test failures at once.
//
// Interface is tested through doctests anyway.
#[cfg(test)]
mod tests;

pub use self::existing_or_new::ExistingOrNew;
pub use self::clone_map::CloneConMap;
pub use self::map::ConMap;
