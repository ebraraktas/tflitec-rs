use std::ffi::{c_int, CString};

use crate::error::Result;
use crate::tensor::{Shape, Tensor};
use crate::{bindings::*, Error, ErrorKind};

pub struct SignatureRunner<'a> {
    signature_runner_ptr: *mut TfLiteSignatureRunner,

    /// The underlying `Interpreter` to limit lifetime of the signature runner.
    #[allow(dead_code)]
    interpreter: &'a crate::interpreter::Interpreter<'a>,
}

impl<'a> SignatureRunner<'a> {
    pub(crate) fn new(
        signature_runner_ptr: *mut TfLiteSignatureRunner,
        interpreter: &'a crate::interpreter::Interpreter<'a>,
    ) -> Self {
        Self {
            signature_runner_ptr,
            interpreter,
        }
    }

    /// Returns the number of inputs associated with the signature.
    pub fn input_count(&self) -> usize {
        unsafe { TfLiteSignatureRunnerGetInputCount(self.signature_runner_ptr) }
    }

    /// Returns the name of the input at the specified index.
    pub fn get_input_name(&self, index: usize) -> &str {
        unsafe {
            let name = TfLiteSignatureRunnerGetInputName(self.signature_runner_ptr, index as i32);
            std::ffi::CStr::from_ptr(name).to_str().unwrap()
        }
    }

    /// Resizes the input tensor to the specified shape.
    ///
    /// # Arguments
    ///
    /// * `input_name`: The name of the input tensor.
    /// * `shape`: The shape to resize the input tensor to.
    ///
    /// # Errors
    ///
    /// Returns error if TensorFlow Lite C fails internally.
    pub fn resize_input(&self, input_name: &str, shape: Shape) -> Result<()> {
        let dims = shape
            .dimensions()
            .iter()
            .map(|v| *v as i32)
            .collect::<Vec<i32>>();
        unsafe {
            let r = TfLiteSignatureRunnerResizeInputTensor(
                self.signature_runner_ptr,
                CString::new(input_name).unwrap().as_ptr(),
                dims.as_ptr() as *const c_int,
                dims.len() as i32,
            );
            if r == TfLiteStatus_kTfLiteOk {
                Ok(())
            } else {
                Err(Error::new(ErrorKind::FailedToResizeNamedInputTensor))
            }
        }
    }

    /// Allocates tensors for the signature runner.
    ///
    /// # Errors
    ///
    /// Returns error if TensorFlow Lite C fails internally.
    pub fn allocate_tensors(&self) -> Result<()> {
        unsafe {
            let r = TfLiteSignatureRunnerAllocateTensors(self.signature_runner_ptr);
            if r == TfLiteStatus_kTfLiteOk {
                Ok(())
            } else {
                Err(Error::new(ErrorKind::FailedToAllocateTensors))
            }
        }
    }

    /// Returns the input tensor with the specified name.
    ///
    /// # Arguments
    ///
    /// * `name`: The name of the input tensor.
    ///
    /// # Errors
    ///
    /// Returns error if TensorFlow Lite C fails internally.
    pub fn get_input_tensor(&self, name: &str) -> Result<Tensor> {
        unsafe {
            let tensor = TfLiteSignatureRunnerGetInputTensor(
                self.signature_runner_ptr,
                CString::new(name).unwrap().as_ptr(),
            );
            if tensor.is_null() {
                Err(Error::new(ErrorKind::InvalidTensorName))
            } else {
                Tensor::from_raw(tensor)
            }
        }
    }

    /// Invokes the signature runner.
    ///
    /// # Errors
    ///
    /// Returns error if TensorFlow Lite C fails internally.
    pub fn invoke(&self) -> Result<()> {
        unsafe {
            let r = TfLiteSignatureRunnerInvoke(self.signature_runner_ptr);
            if r == TfLiteStatus_kTfLiteOk {
                Ok(())
            } else {
                Err(Error::new(ErrorKind::AllocateTensorsRequired))
            }
        }
    }

    /// Returns the number of outputs associated with the signature.
    pub fn output_count(&self) -> usize {
        unsafe { TfLiteSignatureRunnerGetOutputCount(self.signature_runner_ptr) }
    }

    /// Returns the name of the output at the specified index.
    pub fn get_output_name(&self, index: usize) -> &str {
        unsafe {
            let name = TfLiteSignatureRunnerGetOutputName(self.signature_runner_ptr, index as i32);
            std::ffi::CStr::from_ptr(name).to_str().unwrap()
        }
    }

    /// Returns the output tensor with the specified name.
    ///
    /// # Arguments
    ///
    /// * `name`: The name of the output tensor.
    ///
    /// # Errors
    ///
    /// Returns error if TensorFlow Lite C fails internally.
    pub fn get_output_tensor(&self, name: &str) -> Result<Tensor> {
        unsafe {
            let tensor = TfLiteSignatureRunnerGetOutputTensor(
                self.signature_runner_ptr,
                CString::new(name).unwrap().as_ptr(),
            );
            if tensor.is_null() {
                Err(Error::new(ErrorKind::InvalidTensorName))
            } else {
                Tensor::from_raw(tensor as *mut TfLiteTensor)
            }
        }
    }
}

impl<'a> Drop for SignatureRunner<'a> {
    fn drop(&mut self) {
        unsafe { TfLiteSignatureRunnerDelete(self.signature_runner_ptr) }
    }
}

#[cfg(test)]
mod tests {
    use crate::interpreter::Interpreter;
    use crate::model::Model;
    use crate::tensor;
    use crate::ErrorKind;

    #[cfg(target_os = "windows")]
    const MODEL_PATH: &str = "tests\\signatures.bin";
    #[cfg(not(target_os = "windows"))]
    const MODEL_PATH: &str = "tests/signatures.bin";

    #[test]
    fn test_signature_model_interpreter() {
        let model = Model::new(MODEL_PATH).unwrap();
        let interpreter = Interpreter::new(&model, None).unwrap();

        assert_eq!(interpreter.signature_count(), 2);
        assert_eq!(interpreter.get_signature_key(0), Ok("add"));
        assert_eq!(interpreter.get_signature_key(1), Ok("multiply"));
    }

    #[test]
    fn test_signature_runner_invoke() {
        let model = Model::new(MODEL_PATH).unwrap();
        let interpreter = Interpreter::new(&model, None).unwrap();

        let add = interpreter.get_signature_runner("add").unwrap();
        assert_eq!(add.input_count(), 2);
        assert_eq!(add.output_count(), 1);
        assert_eq!(add.get_input_name(0), "x");
        assert_eq!(add.get_input_name(1), "y");
        assert_eq!(add.get_output_name(0), "z");
        assert_eq!(add.resize_input("x", tensor::Shape::new(vec![1])), Ok(()));
        assert_eq!(add.resize_input("y", tensor::Shape::new(vec![1])), Ok(()));
        let r = add.resize_input("y1", tensor::Shape::new(vec![1]));
        assert!(matches!(
            r.err().unwrap().kind(),
            ErrorKind::FailedToResizeNamedInputTensor
        ));
        assert!(matches!(add.allocate_tensors(), Ok(_)));
        add.get_input_tensor("x")
            .unwrap()
            .set_data(&[1.0_f32])
            .unwrap();
        add.get_input_tensor("y")
            .unwrap()
            .set_data(&[2.0_f32])
            .unwrap();
        assert!(matches!(add.invoke(), Ok(_)));
        assert_eq!(add.get_output_tensor("z").unwrap().data::<f32>(), [3.0_f32]);
    }
}
