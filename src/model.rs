//! TensorFlow Lite [`Model`] loader.
//!
//! # Examples
//!
//! ```
//! use tflitec::model::Model;
//! let model = Model::new("tests/add.bin")?;
//! # Ok::<(), tflitec::Error>(())
//! ```
use crate::bindings::{
    TfLiteModel, TfLiteModelCreate, TfLiteModelCreateFromFile, TfLiteModelDelete,
};
use crate::{Error, ErrorKind, Result};
use std::ffi::{c_void, CString};
use std::fmt::{Debug, Formatter};

/// A TensorFlow Lite model used by the [`Interpreter`][crate::interpreter::Interpreter] to perform inference.
pub struct Model<'a> {
    /// The underlying [`TfLiteModel`] C pointer.
    pub(crate) model_ptr: *mut TfLiteModel,

    #[allow(dead_code)]
    /// The model data if initialized with bytes.
    ///
    /// This reference is taken to guarantee that bytes
    /// must be immutable and outlive the model
    pub(crate) bytes: Option<&'a [u8]>,
}

impl Debug for Model<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Model").finish()
    }
}

unsafe impl Send for Model<'_> {}
unsafe impl Sync for Model<'_> {}

impl Model<'_> {
    /// Creates a new instance with the given `filepath`.
    ///
    /// # Arguments
    ///
    /// * `filepath`: The local file path to a TensorFlow Lite model.
    ///
    /// # Errors
    ///
    /// Returns error if TensorFlow Lite C fails to read model from file.
    pub fn new<'a>(filepath: &str) -> Result<Model<'a>> {
        let model_ptr = unsafe {
            let path = CString::new(filepath).unwrap();
            TfLiteModelCreateFromFile(path.as_ptr())
        };
        if model_ptr.is_null() {
            Err(Error::new(ErrorKind::FailedToLoadModel))
        } else {
            Ok(Model {
                model_ptr,
                bytes: None,
            })
        }
    }

    /// Creates a new instance from the given `bytes`.
    ///
    /// # Arguments
    ///
    /// * `bytes`: TensorFlow Lite model data.
    ///
    /// # Errors
    ///
    /// Returns error if TensorFlow Lite C fails to load model from the buffer.
    pub fn from_bytes(bytes: &[u8]) -> std::result::Result<Model, Error> {
        let model_ptr = unsafe {
            TfLiteModelCreate(
                bytes.as_ptr() as *const c_void,
                bytes.len() as crate::bindings::size_t,
            )
        };
        if model_ptr.is_null() {
            Err(Error::new(ErrorKind::FailedToLoadModel))
        } else {
            Ok(Model {
                model_ptr,
                bytes: Some(bytes),
            })
        }
    }
}

impl Drop for Model<'_> {
    fn drop(&mut self) {
        unsafe { TfLiteModelDelete(self.model_ptr) }
    }
}

#[cfg(test)]
mod tests {
    use crate::model::Model;

    const MODEL_PATH: &str = "tests/add.bin";

    #[test]
    fn test_model_from_bytes() {
        let mut bytes = std::fs::read("tests/add.bin").unwrap();
        // If we assign model to a named variable, this test won't compile, because
        // model borrows bytes and will be dropped at the end of the test (after mutation).
        let _ = Model::from_bytes(&bytes).expect("Cannot load model from bytes");
        bytes[0] = 1;
    }

    #[test]
    fn test_model_from_path() {
        let mut filepath = String::from(MODEL_PATH);
        let _model = Model::new(&filepath).expect("Cannot load model from file");
        // We can mutate filepath here, because it is not borrowed.
        filepath.push('/');
    }
}
