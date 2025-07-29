// only enables the `doc_cfg` feature when
// the `docsrs` configuration attribute is defined
#![cfg_attr(docsrs, feature(doc_cfg))]
#![doc = include_str!("../README.md")]

mod error;
pub mod interpreter;
pub mod model;
pub mod signature_runner;
pub mod tensor;

pub(crate) mod bindings {
    #![allow(clippy::all)]
    #![allow(non_upper_case_globals)]
    #![allow(non_camel_case_types)]
    #![allow(non_snake_case)]
    #![allow(unused)]
    #![allow(improper_ctypes)]
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}

pub use self::error::{Error, ErrorKind, Result};

/// Returns TensorFlow Lite version.
pub fn tf_lite_version() -> &'static str {
    use std::ffi::CStr;
    unsafe { CStr::from_ptr(bindings::TfLiteVersion()).to_str().unwrap() }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_version() {
        let version = super::tf_lite_version();
        assert_eq!(version, "2.19.0")
    }
}
