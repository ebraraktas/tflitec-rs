This crate is a safe Rust wrapper of [TensorFlow Lite C API].
Its API is very similar to that of [TensorFlow Lite Swift API].

# Supported Targets

Targets below are tested. However, others may work, too.
* iOS: `aarch64-apple-ios` and `x86_64-apple-ios`
* MacOS: `x86_64-apple-darwin`
* Linux: `x86_64-unknown-linux-gnu`
* Android: `aarch64-linux-android` and `armv7-linux-androideabi`
* Windows ([see details](#Windows))

See [compilation](#compilation) section to see build instructions for your target. Please
read [Optimized Build](#optimized_build) section carefully.

# Features

* `xnnpack` - Compiles XNNPACK and allows you to use XNNPACK delegate. See details of XNNPACK
on [here][XNNPACK_blog].
* `xnnpack_qs8` - Compiles XNNPACK with additional build flags to accelerate inference of
operators with symmetric quantization. See details in [this blog post][XNNPACK_quant_blog].
Implies `xnnpack`.
* `xnnpack_qu8` - Similar to `xnnpack_qs8`, but accelerates few operators with
asymmetric quantization. Implies `xnnpack`.

*Note:* `xnnpack` is already enabled for iOS, but `xnnpack_qs8` and `xnnpack_qu8`
should be enabled manually.

# Examples

The example below shows running inference on a TensorFlow Lite model.

```rust
use tflitec::interpreter::{Interpreter, Options};
use tflitec::tensor;
use std::path::MAIN_SEPARATOR;

// Create interpreter options
let mut options = Options::default();
options.thread_count = 1;

// Load example model which outputs y = 3 * x
let path = format!("tests{}add.bin", MAIN_SEPARATOR);
let interpreter = Interpreter::with_model_path(&path, Some(options))?;
// Resize input
let input_shape = tensor::Shape::new(vec![10, 8, 8, 3]);
interpreter.resize_input(0, input_shape)?;
// Allocate tensors if you just created Interpreter or resized its inputs
interpreter.allocate_tensors()?;

// Create dummy input
let input_element_count = 10 * 8 * 8 * 3;
let data = (0..input_element_count).map(|x| x as f32).collect::<Vec<f32>>();

let input_tensor = interpreter.input(0)?;
assert_eq!(input_tensor.data_type(), tensor::DataType::Float32);

// Copy input to buffer of first tensor (with index 0)
// You have 2 options:
// Set data using Tensor handle if you have it already
assert!(input_tensor.set_data(&data[..]).is_ok());
// Or set data using Interpreter:
assert!(interpreter.copy(&data[..], 0).is_ok());

// Invoke interpreter
assert!(interpreter.invoke().is_ok());

// Get output tensor
let output_tensor = interpreter.output(0)?;

assert_eq!(output_tensor.shape().dimensions(), &vec![10, 8, 8, 3]);
let output_vector = output_tensor.data::<f32>().to_vec();
let expected: Vec<f32> = data.iter().map(|e| e * 3.0).collect();
assert_eq!(expected, output_vector);
# // The line below is needed for doctest, please ignore it
# Ok::<(), tflitec::Error>(())
```

# Compilation

Current version of the crate builds `r2.6` branch of [tensorflow project].
Compiled dynamic library or Framework will be available under `OUT_DIR`
(see [cargo documentation]) of `tflitec`.
You won't need this most of the time, because the crate output is linked appropriately.
For all environments and targets you will need to have:

* `git` CLI to fetch [TensorFlow]
* [Bazel] to build [TensorFlow], supported versions: `[3.7.2, 3.99.0]`
* Python3 to build [TensorFlow]

## Optimized Build
To build [TensorFlow] for your machine with native optimizations
or pass other `--copts` to [Bazel], set environment variable below:
```sh
BAZEL_COPTS="OPT1 OPT2 ..." # space seperated values will be passed as `--copt=OPTN` to bazel
BAZEL_COPTS="-march=native" # for native optimized build
```
---
Some OSs or targets may require additional steps.

## Android:
* [Android NDK]
* Following environment variables should be set appropriately
to build [TensorFlow] for android:
    * `ANDROID_NDK_HOME`
    * `ANDROID_NDK_API_LEVEL`
    * `ANDROID_SDK_HOME`
    * `ANDROID_API_LEVEL`
    * `ANDROID_BUILD_TOOLS_VERSION`
* [Bindgen] needs extra arguments, so set the environment variable below:
```sh
# Set appropriate host tag and target name.
# see https://developer.android.com/ndk/guides/other_build_systems
HOST_TAG=darwin-x86_64 # as example
TARGET_TRIPLE=arm-linux-androideabi # as example
BINDGEN_EXTRA_CLANG_ARGS="\
-I${ANDROID_NDK_HOME}/sources/cxx-stl/llvm-libc++/include/ \
-I${ANDROID_NDK_HOME}/sysroot/usr/include/ \
-I${ANDROID_NDK_HOME}/toolchains/llvm/prebuilt/${HOST_TAG}/sysroot/usr/include/${TARGET_TRIPLE}/"
```
* (Recommended) [cargo-ndk] simplifies `cargo build` process.

## Windows

Windows support is experimental. It is tested on Windows 10. You should follow instructions in
the `Setup for Windows` section on [TensorFlow Build Instructions for Windows]. In other words,
you should install following before build:
* Python 3.8.x 64 bit (the instructions suggest 3.6.x but this package is tested with 3.8.x)
* Bazel, supported versions: `[3.7.2, 3.99.0]`
* [MSYS2]
* Visual C++ Build Tools 2019

Do not forget to add relevant paths to `%PATH%` environment variable by following the 
[TensorFlow Build Instructions for Windows] **carefully** (the only exception is the Python version). 

[TensorFlow]: https://www.tensorflow.org/
[Bazel]: https://bazel.build/
[Bindgen]: https://github.com/rust-lang/rust-bindgen
[tensorflow project]: https://github.com/tensorflow/tensorflow
[TensorFlow Lite Swift API]: https://www.tensorflow.org/lite/guide/ios
[TensorFlow Lite C API]: https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/c
[XNNPACK_blog]: https://blog.tensorflow.org/2020/07/accelerating-tensorflow-lite-xnnpack-integration.html
[XNNPACK_quant_blog]: https://blog.tensorflow.org/2021/09/faster-quantized-inference-with-xnnpack.html
[Android NDK]: https://developer.android.com/ndk/guides
[cargo documentation]: https://doc.rust-lang.org/cargo/reference/environment-variables.html#environment-variables-cargo-sets-for-crates
[cargo-ndk]: https://github.com/bbqsrc/cargo-ndk
[TensorFlow Build Instructions for Windows]: https://www.tensorflow.org/install/source_windows
[MSYS2]: https://www.msys2.org/