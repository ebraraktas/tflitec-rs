#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
use std::ffi::CString;

pub struct Model {
    /// The underlying `TfLiteModel` C pointer.
    pub(crate) model_ptr: *mut TfLiteModel,
}

unsafe impl Send for Model {}
unsafe impl Sync for Model {}

impl Model {
    /// Creates a new instance with the given `filepath`.
    ///
    /// # Arguments
    ///
    /// * `filepath`: The local file path to a TensorFlow Lite model.
    ///
    /// returns: Model
    pub fn new(filepath: &str) -> Model {
        let model_ptr = unsafe {
            let path = CString::new(filepath).unwrap();
            TfLiteModelCreateFromFile(path.as_ptr())
        };
        Model { model_ptr }
    }
}

impl Drop for Model {
    fn drop(&mut self) {
        unsafe { TfLiteModelDelete(self.model_ptr) }
    }
}
