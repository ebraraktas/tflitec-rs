#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

use crate::model::Model;
use crate::tensor;
use crate::tensor::Tensor;
use std::ffi::c_void;
use std::os::raw::c_int;
include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

#[derive(Debug, Eq, PartialEq)]
pub enum InterpreterError {
    InvalidTensorIndex(usize, usize),     // index, max_index
    InvalidTensorDataCount(usize, usize), // provided, required
    FailedToResizeInputTensor(usize),
    AllocateTensorsRequired,
    InvalidTensorDataType,
    FailedToAllocateTensors,
    FailedToCopyDataToInputTensor,
}
pub struct Options {
    pub thread_count: i32,
}

/// A TensorFlow Lite interpreter that performs inference from a given model.
pub struct Interpreter {
    /// The configuration options for the `Interpreter`.
    options: Option<Options>,

    /// The underlying `TfLiteInterpreter` C pointer.
    interpreter_ptr: *mut TfLiteInterpreter,
}

unsafe impl Send for Interpreter{}

impl Interpreter {
    pub fn new(model: &Model, options: Option<Options>) -> Interpreter {
        unsafe {
            let options_ptr = TfLiteInterpreterOptionsCreate();
            if let Some(thread_count) = options.as_ref().and_then(|s| Some(s.thread_count)) {
                TfLiteInterpreterOptionsSetNumThreads(options_ptr, thread_count);
            }
            let model_ptr = model.model_ptr as *const TfLiteModel;
            let interpreter_ptr = TfLiteInterpreterCreate(model_ptr, options_ptr);
            TfLiteInterpreterOptionsDelete(options_ptr);
            Interpreter {
                options,
                interpreter_ptr,
            }
        }
    }

    pub fn with_model_path(model_path: &str, options: Option<Options>) -> Interpreter {
        let model = Model::new(model_path);
        Interpreter::new(&model, options)
    }

    /// The total number of input `Tensor`s associated with the model.
    pub fn input_tensor_count(&self) -> usize {
        unsafe { TfLiteInterpreterGetInputTensorCount(self.interpreter_ptr) as usize }
    }

    /// The total number of output `Tensor`s associated with the model.
    pub fn output_tensor_count(&self) -> usize {
        unsafe { TfLiteInterpreterGetOutputTensorCount(self.interpreter_ptr) as usize }
    }

    /// Invokes the interpreter to perform inference from the loaded graph.
    pub fn invoke(&self) -> Result<(), InterpreterError> {
        if TfLiteStatus_kTfLiteOk == unsafe { TfLiteInterpreterInvoke(self.interpreter_ptr) } {
            Ok(())
        } else {
            Err(InterpreterError::AllocateTensorsRequired)
        }
    }

    pub fn get_input_tensor(&self, index: usize) -> Result<Tensor, InterpreterError> {
        let max_index = self.input_tensor_count() - 1;
        if index > max_index {
            return Err(InterpreterError::InvalidTensorIndex(index, max_index));
        }
        unsafe {
            let tensor_ptr = TfLiteInterpreterGetInputTensor(self.interpreter_ptr, index as i32);
            Tensor::from_raw(tensor_ptr as *mut tensor::TfLiteTensor)
        }
    }

    pub fn get_output_tensor(&self, index: usize) -> Result<Tensor, InterpreterError> {
        let max_index = self.output_tensor_count() - 1;
        if index > max_index {
            return Err(InterpreterError::InvalidTensorIndex(index, max_index));
        }
        unsafe {
            let tensor_ptr = TfLiteInterpreterGetOutputTensor(self.interpreter_ptr, index as i32);
            Tensor::from_raw(tensor_ptr as *mut tensor::TfLiteTensor)
        }
    }

    pub fn resize_input(&self, index: usize, shape: tensor::Shape) -> Result<(), InterpreterError> {
        let max_index = self.input_tensor_count() - 1;
        if index > max_index {
            return Err(InterpreterError::InvalidTensorIndex(index, max_index));
        }
        let dims = shape
            .get_dimensions()
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
                Err(InterpreterError::FailedToResizeInputTensor(index))
            }
        }
    }

    pub fn allocate_tensors(&self) -> Result<(), InterpreterError> {
        if TfLiteStatus_kTfLiteOk
            == unsafe { TfLiteInterpreterAllocateTensors(self.interpreter_ptr) }
        {
            Ok(())
        } else {
            Err(InterpreterError::FailedToAllocateTensors)
        }
    }

    /// Copies the given data to the input `Tensor` at the given index.
    ///
    /// # Arguments
    ///
    /// * `data`: The data to be copied to the input `Tensor`'s data buffer
    /// * `index`: The index for the input `Tensor`
    ///
    /// returns: Result<(), InterpreterError>
    pub fn copy_bytes(&self, data: &[u8], index: usize) -> Result<(), InterpreterError> {
        let max_index = self.input_tensor_count() - 1;
        if index > max_index {
            return Err(InterpreterError::InvalidTensorIndex(index, max_index));
        }
        unsafe {
            let tensor_ptr = TfLiteInterpreterGetInputTensor(self.interpreter_ptr, index as i32);
            let byte_count = TfLiteTensorByteSize(tensor_ptr) as usize;
            if data.len() != byte_count {
                return Err(InterpreterError::InvalidTensorDataCount(
                    data.len(),
                    byte_count,
                ));
            }
            let status = TfLiteTensorCopyFromBuffer(
                tensor_ptr,
                data.as_ptr() as *const c_void,
                data.len() as size_t,
            );
            if status != TfLiteStatus_kTfLiteOk {
                Err(InterpreterError::FailedToCopyDataToInputTensor)
            } else {
                Ok(())
            }
        }
    }

    pub fn copy<T>(&self, data: &[T], index: usize) -> Result<(), InterpreterError> {
        let element_size = std::mem::size_of::<T>();
        let d = unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * element_size)
        };
        self.copy_bytes(d, index)
    }
}

impl Drop for Interpreter {
    fn drop(&mut self) {
        unsafe { TfLiteInterpreterDelete(self.interpreter_ptr) }
    }
}

#[cfg(test)]
mod tests {
    use crate::interpreter::{Interpreter, InterpreterError};
    use crate::tensor;

    const MODEL_PATH: &'static str = "tests/add.bin";

    #[test]
    fn test_interpreter_with_model_path() {
        let _ = Interpreter::with_model_path(MODEL_PATH, None);
    }

    #[test]
    fn test_interpreter_input_output_count() {
        let interpreter = Interpreter::with_model_path(MODEL_PATH, None);
        assert_eq!(interpreter.input_tensor_count(), 1);
        assert_eq!(interpreter.output_tensor_count(), 1);
    }

    #[test]
    fn test_interpreter_get_input_tensor() {
        let interpreter = Interpreter::with_model_path(MODEL_PATH, None);
        let invalid_tensor = interpreter.get_input_tensor(1);
        assert!(invalid_tensor.is_err());
        let err = invalid_tensor.err().unwrap();
        assert_eq!(InterpreterError::InvalidTensorIndex(1, 0), err);
        let valid_tensor = interpreter.get_input_tensor(0);
        assert!(valid_tensor.is_ok());
        let tensor = valid_tensor.ok().unwrap();
        assert_eq!(tensor.get_shape().get_dimensions(), &vec![1, 8, 8, 3])
    }

    #[test]
    fn test_interpreter_allocate_tensors() {
        let interpreter = Interpreter::with_model_path(MODEL_PATH, None);
        interpreter
            .resize_input(0, tensor::Shape::new(vec![10, 8, 8, 3]))
            .expect("Resize failed");
        interpreter
            .allocate_tensors()
            .expect("Cannot allocate tensors");
        let tensor = interpreter.get_input_tensor(0).unwrap();
        assert_eq!(tensor.get_shape().get_dimensions(), &vec![10, 8, 8, 3])
    }

    #[test]
    fn test_interpreter_copy_input() {
        let interpreter = Interpreter::with_model_path(MODEL_PATH, None);
        interpreter
            .resize_input(0, tensor::Shape::new(vec![10, 8, 8, 3]))
            .expect("Resize failed");
        interpreter
            .allocate_tensors()
            .expect("Cannot allocate tensors");
        let tensor = interpreter.get_input_tensor(0).unwrap();
        let data = (0..1920).map(|x| x as f32).collect::<Vec<f32>>();
        assert!(interpreter.copy(&data[..], 0).is_ok());
        assert_eq!(data, tensor.get_data());
    }

    #[test]
    fn test_interpreter_invoke() {
        let interpreter = Interpreter::with_model_path(MODEL_PATH, None);
        interpreter
            .resize_input(0, tensor::Shape::new(vec![10, 8, 8, 3]))
            .expect("Resize failed");
        interpreter
            .allocate_tensors()
            .expect("Cannot allocate tensors");
        let tensor = interpreter.get_input_tensor(0).unwrap();
        let data = (0..1920).map(|x| x as f32).collect::<Vec<f32>>();
        assert!(interpreter.copy(&data[..], 0).is_ok());
        assert!(interpreter.invoke().is_ok());
        let expected: Vec<f32> = data.iter().map(|e| e * 3.0).collect();
        let output_tensor = interpreter.get_output_tensor(0).unwrap();
        assert_eq!(
            output_tensor.get_shape().get_dimensions(),
            &vec![10, 8, 8, 3]
        );
        let output_vector = output_tensor.get_data::<f32>().to_vec();
        assert_eq!(expected, output_vector);
    }
}
