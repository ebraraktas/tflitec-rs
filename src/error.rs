use core::fmt::{Display, Formatter};

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ErrorKind {
    InvalidTensorIndex(/* index: */ usize, /* max_index: */ usize),
    InvalidTensorDataCount(/* provided: */ usize, /* required: */ usize),
    FailedToResizeInputTensor(/* index: */ usize),
    AllocateTensorsRequired,
    InvalidTensorDataType,
    FailedToAllocateTensors,
    FailedToCopyDataToInputTensor,
}

impl ErrorKind {
    pub(crate) fn as_string(&self) -> String {
        match *self {
            ErrorKind::InvalidTensorIndex(index, max_index) => {
                format!("invalid tensor index {}, max index is {}", index, max_index)
            }
            ErrorKind::InvalidTensorDataCount(provided, required) => format!(
                "provided data count {} must match the required count {}",
                provided, required
            ),
            ErrorKind::InvalidTensorDataType => {
                "tensor data type is unsupported or could not be determined due to a model error"
                    .to_string()
            }
            ErrorKind::FailedToResizeInputTensor(index) => {
                format!("failed to resize input tensor at index {}", index)
            }
            ErrorKind::AllocateTensorsRequired => "must call allocate_tensors()".to_string(),
            ErrorKind::FailedToAllocateTensors => {
                "failed to allocate memory for input tensors".to_string()
            }
            ErrorKind::FailedToCopyDataToInputTensor => {
                "failed to copy data to input tensor".to_string()
            }
        }
    }
}

impl Display for ErrorKind {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.as_string())
    }
}

#[derive(Debug)]
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
