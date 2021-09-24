//! API of TensorFlow Lite [`Interpreter`] that performs inference.
use std::ffi::c_void;
use std::os::raw::c_int;

use crate::bindings::*;
use crate::model::Model;
use crate::tensor;
use crate::tensor::Tensor;
use crate::{Error, ErrorKind, Result};

/// Options for configuring the [`Interpreter`].
#[derive(Debug, Eq, PartialEq, Copy, Clone, Hash, Ord, PartialOrd)]
pub struct Options {
    /// The maximum number of CPU threads that the interpreter should run on.
    ///
    /// The default is -1 indicating that the [`Interpreter`] will decide
    /// the number of threads to use. `thread_count` should be >= -1.
    /// Setting `thread_count` to 0 has the effect to disable multithreading,
    /// which is equivalent to setting `thread_count` to 1.
    /// If set to the value -1, the number of threads used will be
    /// implementation-defined and platform-dependent.
    pub thread_count: i32,

    /// Indicates whether an optimized set of floating point CPU kernels, provided by XNNPACK, is
    /// enabled.
    ///
    /// # Details (from TensorFlowLiteSwift documentation)
    /// ## Experiment:
    /// Enabling this flag will enable use of a new, highly optimized set of CPU kernels provided
    /// via the XNNPACK delegate. ~~Currently, this is restricted to a subset of floating point
    /// operations. Eventually, we plan to enable this by default, as it can provide significant
    /// performance benefits for many classes of floating point models.~~
    /// See [official README](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/delegates/xnnpack/README.md) for more details.
    ///
    /// ## Important:
    /// Things to keep in mind when enabling this flag:
    ///
    /// * Startup time and resize time may increase.
    /// * Baseline memory consumption may increase.
    /// * Compatibility with other delegates (e.g., GPU) has not been fully validated.
    /// * Quantized models will not see any benefit.
    ///
    /// **Warning:** This is an experimental interface that is subject to change.
    #[cfg(feature = "xnnpack")]
    #[cfg_attr(docsrs, doc(cfg(feature = "xnnpack")))]
    pub is_xnnpack_enabled: bool,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            thread_count: -1,
            #[cfg(feature = "xnnpack")]
            is_xnnpack_enabled: false
        }
    }
}

/// A TensorFlow Lite interpreter that performs inference from a given model.
///
/// - Note: Interpreter instances are *not* thread-safe.
pub struct Interpreter {
    /// The configuration options for the [`Interpreter`].
    options: Option<Options>,

    /// The underlying [`TfLiteInterpreter`] C pointer.
    interpreter_ptr: *mut TfLiteInterpreter,

    /// The underlying [`TfLiteDelegate`] C pointer for XNNPACK delegate.
    #[cfg(feature = "xnnpack")]
    xnnpack_delegate_ptr: Option<*mut TfLiteDelegate>
}

unsafe impl Send for Interpreter {}

impl Interpreter {
    /// Creates new [`Interpreter`]
    ///
    /// # Arguments
    ///
    /// * `model`: TensorFlow Lite [model][`Model`]
    /// * `options`: Interpreter [options][`Options`]
    ///
    /// # Examples
    ///
    /// ```
    /// use tflitec::model::Model;
    /// use tflitec::interpreter::Interpreter;
    /// let model = Model::new("tests/add.bin")?;
    /// let interpreter = Interpreter::new(&model, None)?;
    /// # Ok::<(), tflitec::Error>(())
    /// ```
    pub fn new(model: &Model, options: Option<Options>) -> Result<Interpreter> {
        unsafe {
            let options_ptr = TfLiteInterpreterOptionsCreate();
            if options_ptr.is_null() {
                return Err(Error::new(ErrorKind::FailedToCreateInterpreter));
            }
            if let Some(thread_count) = options.as_ref().map(|s| s.thread_count) {
                TfLiteInterpreterOptionsSetNumThreads(options_ptr, thread_count);
            }

            #[cfg(feature = "xnnpack")]
            let mut xnnpack_delegate_ptr: Option<*mut TfLiteDelegate> = None;
            #[cfg(feature = "xnnpack")]
            {
                if let Some(options) = options.as_ref() {
                    if options.is_xnnpack_enabled {
                        xnnpack_delegate_ptr =
                            Some(Interpreter::configure_xnnpack(options, options_ptr));
                    }
                }
            }

            // TODO(ebraraktas): TfLiteInterpreterOptionsSetErrorReporter
            let model_ptr = model.model_ptr as *const TfLiteModel;
            let interpreter_ptr = TfLiteInterpreterCreate(model_ptr, options_ptr);
            TfLiteInterpreterOptionsDelete(options_ptr);
            if interpreter_ptr.is_null() {
                Err(Error::new(ErrorKind::FailedToCreateInterpreter))
            } else {
                Ok(Interpreter {
                    options,
                    interpreter_ptr,
                    #[cfg(feature = "xnnpack")]
                    xnnpack_delegate_ptr
                })
            }
        }
    }

    /// Creates an [`Interpreter`] with model path.
    ///
    /// # Arguments
    ///
    /// * `model_path`: Path to TensorFlow Lite model
    /// * `options`: Interpreter [`Options`]
    ///
    /// returns: Result<Interpreter, Error>
    ///
    /// # Examples
    ///
    /// ```
    /// use tflitec::interpreter::Interpreter;
    /// let interpreter = Interpreter::with_model_path("tests/add.bin", None)?;
    /// # Ok::<(), tflitec::Error>(())
    /// ```
    pub fn with_model_path(model_path: &str, options: Option<Options>) -> Result<Interpreter> {
        let model = Model::new(model_path)?;
        Interpreter::new(&model, options)
    }

    /// Returns the total number of input [`Tensor`]s associated with the model.
    pub fn input_tensor_count(&self) -> usize {
        unsafe { TfLiteInterpreterGetInputTensorCount(self.interpreter_ptr) as usize }
    }

    /// Returns the total number of output `Tensor`s associated with the model.
    pub fn output_tensor_count(&self) -> usize {
        unsafe { TfLiteInterpreterGetOutputTensorCount(self.interpreter_ptr) as usize }
    }

    /// Invokes the interpreter to perform inference from the loaded graph.
    pub fn invoke(&self) -> Result<()> {
        if TfLiteStatus_kTfLiteOk == unsafe { TfLiteInterpreterInvoke(self.interpreter_ptr) } {
            Ok(())
        } else {
            Err(Error::new(ErrorKind::AllocateTensorsRequired))
        }
    }

    /// Returns the input [`Tensor`] at the given `index`.
    ///
    /// # Arguments
    ///
    /// * `index`: The index for the input [`Tensor`].
    ///
    /// returns: `Result<Tensor, Error>`
    pub fn input(&self, index: usize) -> Result<Tensor> {
        let max_index = self.input_tensor_count() - 1;
        if index > max_index {
            return Err(Error::new(ErrorKind::InvalidTensorIndex(index, max_index)));
        }
        unsafe {
            let tensor_ptr = TfLiteInterpreterGetInputTensor(self.interpreter_ptr, index as i32);
            Tensor::from_raw(tensor_ptr as *mut TfLiteTensor).map_err(|error| {
                if error.kind() == ErrorKind::ReadTensorError {
                    Error::new(ErrorKind::AllocateTensorsRequired)
                } else {
                    error
                }
            })
        }
    }


    /// Returns the output [`Tensor`] at the given `index`.
    ///
    /// # Arguments
    ///
    /// * `index`: The index for the output [`Tensor`].
    ///
    /// returns: `Result<Tensor, Error>`
    pub fn output(&self, index: usize) -> Result<Tensor> {
        let max_index = self.output_tensor_count() - 1;
        if index > max_index {
            return Err(Error::new(ErrorKind::InvalidTensorIndex(index, max_index)));
        }
        unsafe {
            let tensor_ptr = TfLiteInterpreterGetOutputTensor(self.interpreter_ptr, index as i32);
            Tensor::from_raw(tensor_ptr as *mut TfLiteTensor).map_err(|error| {
                if error.kind() == ErrorKind::ReadTensorError {
                    Error::new(ErrorKind::InvokeInterpreterRequired)
                } else {
                    error
                }
            })
        }
    }

    /// Resizes the input [`Tensor`] at the given index to the
    /// specified [`Shape`][tensor::Shape].
    ///
    /// - Note: After resizing an input tensor, the client **must** explicitly call
    /// [`Interpreter::allocate_tensors()`] before attempting to access the resized tensor data
    /// or invoking the interpreter to perform inference.
    ///
    /// # Arguments
    ///
    /// * `index`: The index for the input [`Tensor`].
    /// * `shape`: The shape to resize the input [`Tensor`] to.
    ///
    /// returns: `Result<(), Error>`
    pub fn resize_input(&self, index: usize, shape: tensor::Shape) -> Result<()> {
        let max_index = self.input_tensor_count() - 1;
        if index > max_index {
            return Err(Error::new(ErrorKind::InvalidTensorIndex(index, max_index)));
        }
        let dims = shape
            .dimensions()
            .iter()
            .map(|v| *v as i32)
            .collect::<Vec<i32>>();

        unsafe {
            if TfLiteStatus_kTfLiteOk
                == TfLiteInterpreterResizeInputTensor(
                    self.interpreter_ptr,
                    index as i32,
                    dims.as_ptr() as *const c_int,
                    dims.len() as i32,
                )
            {
                Ok(())
            } else {
                Err(Error::new(ErrorKind::FailedToResizeInputTensor(index)))
            }
        }
    }

    /// Allocates memory for all input [`Tensor`]s based on their [`Shape`][tensor::Shape]s.
    ///
    /// - Note: This is a relatively expensive operation and should only be called
    /// after creating the interpreter and resizing any input tensors.
    ///
    /// returns: An [`ErrorKind::FailedToAllocateTensors`] if memory could not be allocated
    /// for the input tensors.
    pub fn allocate_tensors(&self) -> Result<()> {
        if TfLiteStatus_kTfLiteOk
            == unsafe { TfLiteInterpreterAllocateTensors(self.interpreter_ptr) }
        {
            Ok(())
        } else {
            Err(Error::new(ErrorKind::FailedToAllocateTensors))
        }
    }

    /// Copies the given `data` to the input [`Tensor`] at the given `index`.
    ///
    /// # Arguments
    ///
    /// * `data`: The data to be copied to the input `Tensor`'s data buffer
    /// * `index`: The index for the input `Tensor`
    ///
    /// returns: An [Error] if the `data.len()` does not match the input tensor's
    /// `data.len()` or if the given index is invalid.
    fn copy_bytes(&self, data: &[u8], index: usize) -> Result<()> {
        let max_index = self.input_tensor_count() - 1;
        if index > max_index {
            return Err(Error::new(ErrorKind::InvalidTensorIndex(index, max_index)));
        }
        unsafe {
            let tensor_ptr = TfLiteInterpreterGetInputTensor(self.interpreter_ptr, index as i32);
            let byte_count = TfLiteTensorByteSize(tensor_ptr) as usize;
            if data.len() != byte_count {
                return Err(Error::new(ErrorKind::InvalidTensorDataCount(
                    data.len(),
                    byte_count,
                )));
            }
            let status = TfLiteTensorCopyFromBuffer(
                tensor_ptr,
                data.as_ptr() as *const c_void,
                data.len() as size_t,
            );
            if status != TfLiteStatus_kTfLiteOk {
                Err(Error::new(ErrorKind::FailedToCopyDataToInputTensor))
            } else {
                Ok(())
            }
        }
    }

    /// Copies the given `data` to the input [`Tensor`] at the given `index`.
    ///
    /// # Arguments
    ///
    /// * `data`: The data to be copied to the input `Tensor`'s data buffer.
    /// * `index`: The index for the input [`Tensor`].
    ///
    /// returns: An [Error] if the byte count of `data` does not match the input tensor's
    /// buffer size or if the given `index` is invalid.
    pub fn copy<T>(&self, data: &[T], index: usize) -> Result<()> {
        let element_size = std::mem::size_of::<T>();
        let d = unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * element_size)
        };
        self.copy_bytes(d, index)
    }

    /// Returns optional reference of [`Options`].
    pub fn options(&self) -> Option<&Options> {
        self.options.as_ref()
    }

    #[cfg(feature = "xnnpack")]
    unsafe fn configure_xnnpack(options: &Options, interpreter_options_ptr: *mut TfLiteInterpreterOptions) -> *mut TfLiteDelegate {
        let mut xnnpack_options = TfLiteXNNPackDelegateOptionsDefault();
        if options.thread_count > 0 {
            xnnpack_options.num_threads = options.thread_count
        }

        let xnnpack_delegate_ptr = TfLiteXNNPackDelegateCreate(&xnnpack_options);
        TfLiteInterpreterOptionsAddDelegate(interpreter_options_ptr, xnnpack_delegate_ptr);
        xnnpack_delegate_ptr
    }
}

impl Drop for Interpreter {
    fn drop(&mut self) {
        unsafe {
            TfLiteInterpreterDelete(self.interpreter_ptr);

            #[cfg(feature = "xnnpack")]
            {
                if let Some(delegate_ptr) = self.xnnpack_delegate_ptr {
                    TfLiteXNNPackDelegateDelete(delegate_ptr)
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::interpreter::Interpreter;
    use crate::tensor;
    use crate::ErrorKind;

    const MODEL_PATH: &'static str = "tests/add.bin";

    #[test]
    fn test_interpreter_with_model_path() {
        let _ = Interpreter::with_model_path(MODEL_PATH, None);
    }

    #[test]
    fn test_interpreter_input_output_count() {
        let interpreter = Interpreter::with_model_path(MODEL_PATH, None).unwrap();
        assert_eq!(interpreter.input_tensor_count(), 1);
        assert_eq!(interpreter.output_tensor_count(), 1);
    }

    #[test]
    fn test_interpreter_get_input_tensor() {
        let interpreter = Interpreter::with_model_path(MODEL_PATH, None).unwrap();

        let invalid_tensor = interpreter.input(1);
        assert!(invalid_tensor.is_err());
        let err = invalid_tensor.err().unwrap();
        assert_eq!(ErrorKind::InvalidTensorIndex(1, 0), err.kind());

        let invalid_tensor = interpreter.input(0);
        assert!(invalid_tensor.is_err());
        let err = invalid_tensor.err().unwrap();
        assert_eq!(ErrorKind::AllocateTensorsRequired, err.kind());

        interpreter.allocate_tensors().unwrap();
        let valid_tensor = interpreter.input(0);
        assert!(valid_tensor.is_ok());
        let tensor = valid_tensor.ok().unwrap();
        assert_eq!(tensor.shape().dimensions(), &vec![1, 8, 8, 3])
    }

    #[test]
    fn test_interpreter_allocate_tensors() {
        let interpreter = Interpreter::with_model_path(MODEL_PATH, None).unwrap();
        interpreter
            .resize_input(0, tensor::Shape::new(vec![10, 8, 8, 3]))
            .expect("Resize failed");
        interpreter
            .allocate_tensors()
            .expect("Cannot allocate tensors");
        let tensor = interpreter.input(0).unwrap();
        assert_eq!(tensor.shape().dimensions(), &vec![10, 8, 8, 3])
    }

    #[test]
    fn test_interpreter_copy_input() {
        let interpreter = Interpreter::with_model_path(MODEL_PATH, None).unwrap();
        interpreter
            .resize_input(0, tensor::Shape::new(vec![10, 8, 8, 3]))
            .expect("Resize failed");
        interpreter
            .allocate_tensors()
            .expect("Cannot allocate tensors");
        let tensor = interpreter.input(0).unwrap();
        let data = (0..1920).map(|x| x as f32).collect::<Vec<f32>>();
        assert!(interpreter.copy(&data[..], 0).is_ok());
        assert_eq!(data, tensor.data());
    }

    #[test]
    fn test_interpreter_invoke() {
        let interpreter = Interpreter::with_model_path(MODEL_PATH, None).unwrap();
        interpreter
            .resize_input(0, tensor::Shape::new(vec![10, 8, 8, 3]))
            .expect("Resize failed");
        interpreter
            .allocate_tensors()
            .expect("Cannot allocate tensors");

        let data = (0..1920).map(|x| x as f32).collect::<Vec<f32>>();
        assert!(interpreter.copy(&data[..], 0).is_ok());
        assert!(interpreter.invoke().is_ok());
        let expected: Vec<f32> = data.iter().map(|e| e * 3.0).collect();
        let output_tensor = interpreter.output(0).unwrap();
        assert_eq!(output_tensor.shape().dimensions(), &vec![10, 8, 8, 3]);
        let output_vector = output_tensor.data::<f32>().to_vec();
        assert_eq!(expected, output_vector);
    }
}
