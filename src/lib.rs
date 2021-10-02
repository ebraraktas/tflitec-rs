// only enables the `doc_cfg` feature when
// the `docsrs` configuration attribute is defined
#![cfg_attr(docsrs, feature(doc_cfg))]

#![doc = include_str!("../README.md")]

mod error;
pub mod interpreter;
pub mod model;
pub mod tensor;

pub(crate) mod bindings {
    #![allow(clippy::all)]
    #![allow(non_upper_case_globals)]
    #![allow(non_camel_case_types)]
    #![allow(non_snake_case)]
    #![allow(unused)]
    #![allow(improper_ctypes)]

    // since rustc 1.53, bindgen causes UB warnings -- see
    // https://github.com/rust-lang/rust-bindgen/issues/1651
    // remove this once bindgen has fixed the issue
    // (currently at version 0.59.1)
    #![allow(deref_nullptr)]

    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}

pub use self::error::{Error, ErrorKind, Result};
