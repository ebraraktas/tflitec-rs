use crate::interpreter::{ErrorKind, Error};
use std::ffi::CStr;
use crate::bindings::*;
use crate::bindings;


/// Parameters that determine the mapping of quantized values to real values. Quantized values can
/// be mapped to float values using the following conversion:
/// `realValue = scale * (quantizedValue - zeroPoint)`.
#[derive(Copy, Clone, PartialEq, Debug)]
pub struct QuantizationParameters {
    /// The difference between real values corresponding to consecutive quantized values differing by
    /// 1. For example, the range of quantized values for `UInt8` data type is [0, 255].
    scale: f32,

    /// The quantized value that corresponds to the real 0 value.
    zero_point: i32,
}

impl QuantizationParameters {
    /// Creates a new instance with the given values.
    ///
    /// # Arguments
    ///
    /// * `scale`: The scale value for asymmetric quantization
    /// * `zero_point`: The zero point for asymmetric quantization
    ///
    /// returns: QuantizationParameters
    pub(crate) fn new(scale: f32, zero_point: i32) -> QuantizationParameters {
        QuantizationParameters { scale, zero_point }
    }
}

#[derive(Copy, Clone, Eq, PartialEq)]
pub enum DataType {
    /// A boolean.
    Bool,
    /// An 8-bit unsigned integer.
    Uint8,
    /// A 16-bit signed integer.
    Int16,
    /// A 32-bit signed integer.
    Int32,
    /// A 64-bit signed integer.
    Int64,
    /// A 16-bit half precision floating point.
    Float16,
    /// A 32-bit single precision floating point.
    Float32,
    /// A 64-bit double precision floating point.
    Float64,
}

impl DataType {
    pub(crate) fn new(tflite_type: TfLiteType) -> Option<DataType> {
        match tflite_type {
            bindings::TfLiteType_kTfLiteBool => Some(DataType::Bool),
            bindings::TfLiteType_kTfLiteUInt8 => Some(DataType::Uint8),
            bindings::TfLiteType_kTfLiteInt16 => Some(DataType::Int16),
            bindings::TfLiteType_kTfLiteInt32 => Some(DataType::Int32),
            bindings::TfLiteType_kTfLiteInt64 => Some(DataType::Int64),
            bindings::TfLiteType_kTfLiteFloat16 => Some(DataType::Float16),
            bindings::TfLiteType_kTfLiteFloat32 => Some(DataType::Float32),
            bindings::TfLiteType_kTfLiteFloat64 => Some(DataType::Float64),
            _ => None,
        }
    }
}

pub struct Shape {
    /// The number of dimensions of the `Tensor`
    rank: usize,

    /// An array of dimensions for the `Tensor`
    dimensions: Vec<usize>,
}

impl Shape {
    pub fn new(dimensions: Vec<usize>) -> Shape {
        Shape {
            rank: dimensions.len(),
            dimensions,
        }
    }

    pub fn dimensions(&self) -> &Vec<usize> {
        &self.dimensions
    }

    pub fn rank(&self) -> usize {
        self.rank
    }
}

pub(crate) struct TensorData {
    data_ptr: *mut u8,
    data_length: usize,
}

pub struct Tensor {
    /// The name of the `Tensor`.
    name: String,

    /// The data type of the `Tensor`.
    data_type: DataType,

    /// The shape of the `Tensor`.
    shape: Shape,

    /// The data in the input or output `Tensor`.
    data: TensorData,

    /// The quantization parameters for the `Tensor` if using a quantized model.
    quantization_parameters: Option<QuantizationParameters>,
}

impl Tensor {
    pub(crate) fn new(
        name: String,
        data_type: DataType,
        shape: Shape,
        data: TensorData,
        quantization_parameters: Option<QuantizationParameters>,
    ) -> Tensor {
        Tensor {
            name,
            data_type,
            shape,
            data,
            quantization_parameters,
        }
    }

    pub(crate) fn from_raw(tensor_ptr: *mut TfLiteTensor) -> Result<Tensor, Error> {
        unsafe {
            let data_type = DataType::new(TfLiteTensorType(tensor_ptr))
                .ok_or_else(|| Error::new(ErrorKind::InvalidTensorDataType))?;

            let name = CStr::from_ptr(TfLiteTensorName(tensor_ptr))
                .to_str()
                .unwrap()
                .to_owned();
            let rank = TfLiteTensorNumDims(tensor_ptr);
            let dimensions = (0..rank)
                .map(|i| TfLiteTensorDim(tensor_ptr, i) as usize)
                .collect();
            let shape = Shape::new(dimensions);
            let data_ptr = TfLiteTensorData(tensor_ptr) as *mut u8;
            let data_length = TfLiteTensorByteSize(tensor_ptr) as usize;
            let data = TensorData {
                data_ptr,
                data_length,
            };
            let quantization_parameters_ptr = TfLiteTensorQuantizationParams(tensor_ptr);
            let scale = quantization_parameters_ptr.scale;
            let quantization_parameters = if scale == 0.0 || data_type != DataType::Uint8 {
                None
            } else {
                Some(QuantizationParameters::new(
                    quantization_parameters_ptr.scale,
                    quantization_parameters_ptr.zero_point,
                ))
            };
            let tensor = Tensor::new(name, data_type, shape, data, quantization_parameters);
            Ok(tensor)
        }
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    pub fn get_data<T>(&self) -> &[T] {
        let element_size = std::mem::size_of::<T>();
        // TODO: check => self.data.data_length % element_size == 0
        unsafe {
            std::slice::from_raw_parts(
                self.data.data_ptr as *const T,
                self.data.data_length / element_size,
            )
        }
    }

    pub fn data_type(&self) -> DataType {
        self.data_type
    }

    pub fn quantization_parameters(&self) -> Option<QuantizationParameters> {
        self.quantization_parameters
    }

    pub fn name(&self) -> &str {
        self.name.as_str()
    }
}
