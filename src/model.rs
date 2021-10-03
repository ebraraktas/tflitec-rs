//! TensorFlow Lite [`Model`] loader.
//!
//! # Examples
//!
//! ```
//! use tflitec::model::Model;
//! let model = Model::new("tests/add.bin")?;
//! # Ok::<(), tflitec::Error>(())
//! ```
use crate::bindings::{TfLiteModel, TfLiteModelCreateFromFile, TfLiteModelDelete};
use crate::{Error, ErrorKind, Result};
use std::ffi::CString;
use std::fmt::{Debug, Formatter};

/// A TensorFlow Lite model used by the [`Interpreter`][crate::interpreter::Interpreter] to perform inference.
pub struct Model {
    /// The underlying [`TfLiteModel`] C pointer.
    pub(crate) model_ptr: *mut TfLiteModel,
}

impl Debug for Model {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Model")
            .finish()
    }
}

unsafe impl Send for Model {}
unsafe impl Sync for Model {}

impl Model {
    /// Creates a new instance with the given `filepath`.
    ///
    /// # Arguments
    ///
    /// * `filepath`: The local file path to a TensorFlow Lite model.
    pub fn new(filepath: &str) -> Result<Model> {
        let model_ptr = unsafe {
            let path = CString::new(filepath).unwrap();
            TfLiteModelCreateFromFile(path.as_ptr())
        };
        if model_ptr.is_null() {
            Err(Error::new(ErrorKind::FailedToLoadModel))
        } else {
            Ok(Model { model_ptr })
        }
    }
}

impl Drop for Model {
    fn drop(&mut self) {
        unsafe { TfLiteModelDelete(self.model_ptr) }
    }
}
