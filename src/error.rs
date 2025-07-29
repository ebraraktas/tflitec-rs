//! Definitions of `Error` type and `ErrorKind`s of the crate.
use core::fmt::{Display, Formatter};

/// A list specifying general categories of TensorFlow Lite errors.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ErrorKind {
    /// Indicates given tensor index (first value) is larger than maximum index (second value).
    InvalidTensorIndex(/* index: */ usize, /* max_index: */ usize),
    /// Indicates given data length (first value) is not equal to required length (second value).
    InvalidTensorDataCount(/* provided: */ usize, /* required: */ usize),
    /// Indicates failure to resize tensor with index (first value).
    FailedToResizeInputTensor(/* index: */ usize),
    FailedToResizeNamedInputTensor,
    AllocateTensorsRequired,
    InvalidTensorDataType,
    FailedToAllocateTensors,
    FailedToCopyDataToInputTensor,
    FailedToLoadModel,
    FailedToCreateInterpreter,
    ReadTensorError,
    InvokeInterpreterRequired,
    InvalidSignatureRunner,
    InvalidTensorName,
}

impl ErrorKind {
    pub(crate) fn as_string(&self) -> String {
        match *self {
            ErrorKind::InvalidTensorIndex(index, max_index) => {
                format!("invalid tensor index {index}, max index is {max_index}")
            }
            ErrorKind::InvalidTensorDataCount(provided, required) => {
                format!("provided data count {provided} must match the required count {required}")
            }
            ErrorKind::InvalidTensorDataType => {
                "tensor data type is unsupported or could not be determined due to a model error"
                    .to_string()
            }
            ErrorKind::FailedToResizeInputTensor(index) => {
                format!("failed to resize input tensor at index {index}")
            }
            ErrorKind::FailedToResizeNamedInputTensor => {
                "failed to resize tensor for the given name".to_string()
            }
            ErrorKind::AllocateTensorsRequired => "must call allocate_tensors()".to_string(),
            ErrorKind::FailedToAllocateTensors => {
                "failed to allocate memory for input tensors".to_string()
            }
            ErrorKind::FailedToCopyDataToInputTensor => {
                "failed to copy data to input tensor".to_string()
            }
            ErrorKind::FailedToLoadModel => "failed to load the given model".to_string(),
            ErrorKind::FailedToCreateInterpreter => "failed to create the interpreter".to_string(),
            ErrorKind::ReadTensorError => "failed to read tensor".to_string(),
            ErrorKind::InvokeInterpreterRequired => "must call invoke()".to_string(),
            ErrorKind::InvalidSignatureRunner => {
                "failed to get signature runner for the given key".to_string()
            }
            ErrorKind::InvalidTensorName => "failed to get tensor for the given name".to_string(),
        }
    }
}

impl Display for ErrorKind {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.as_string())
    }
}

/// The error type for TensorFlow Lite operations.
#[derive(Debug, Copy, Clone, Hash, Eq, PartialEq)]
pub struct Error {
    kind: ErrorKind,
}

impl Display for Error {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.kind)
    }
}

impl std::error::Error for Error {}

impl Error {
    pub(crate) fn new(kind: ErrorKind) -> Error {
        Error { kind }
    }
    pub fn kind(&self) -> ErrorKind {
        self.kind
    }
}

/// A specialized [`Result`] type for API operations.
pub type Result<T> = std::result::Result<T, Error>;
