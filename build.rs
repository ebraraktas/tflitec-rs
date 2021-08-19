extern crate bindgen;

use std::env;
use std::path::PathBuf;
use std::time::Instant;

use fs_extra;

fn copy_tensorflow() -> PathBuf {
    let src = "third_party/tensorflow";
    let tgt_result = out_dir().join("tensorflow");
    if !tgt_result.exists() {
        let mut opts = fs_extra::dir::CopyOptions::new();
        opts.overwrite = true;
        opts.buffer_size = 65536;
        println!("Copy started {} -> {}", &src, &tgt_result.display());
        let start = Instant::now();
        fs_extra::dir::copy(src, &tgt_result.parent().unwrap(), &opts).unwrap();
        println!("Copy took {:?}", Instant::now() - start);
    }
    tgt_result
}

fn check_and_set_envs() {
    // TODO: this won't work on Windows
    let python_bin_path = PathBuf::from(
        String::from_utf8(
            std::process::Command::new("which")
                .arg("python3")
                .output()
                .or_else(|_| std::process::Command::new("which").arg("python").output())
                .expect("Cannot get python path")
                .stdout,
        )
        .expect("Cannot decode utf8")
        .trim(),
    );
    let os = env::var("CARGO_CFG_TARGET_OS").expect("Unable to get TARGET_OS");
    let default_envs = [
        ["PYTHON_BIN_PATH", python_bin_path.to_str().unwrap()],
        ["USE_DEFAULT_PYTHON_LIB_PATH", "1"],
        ["TF_NEED_OPENCL", "0"],
        ["TF_CUDA_CLANG", "0"],
        ["TF_NEED_TENSORRT", "0"],
        ["TF_DOWNLOAD_CLANG", "0"],
        ["TF_NEED_MPI", "0"],
        ["TF_NEED_ROCM", "0"],
        ["TF_NEED_CUDA", "0"],
        ["CC_OPT_FLAGS", "-Wno-sign-compare"],
        [
            "TF_SET_ANDROID_WORKSPACE",
            if os == "android" { "1" } else { "0" },
        ],
        ["TF_CONFIGURE_IOS", if os == "ios" { "1" } else { "0" }],
    ];
    for kv in default_envs {
        let name = kv[0];
        let val = kv[1];
        if env::var(name).is_err() {
            env::set_var(name, val);
        }
    }
    let true_vals = ["1", "t", "true", "y", "yes"];
    if true_vals.contains(&env::var("TF_SET_ANDROID_WORKSPACE").unwrap().as_str()) {
        let android_env_vars = [
            "ANDROID_NDK_HOME",
            "ANDROID_NDK_API_LEVEL",
            "ANDROID_SDK_HOME",
            "ANDROID_API_LEVEL",
            "ANDROID_BUILD_TOOLS_VERSION",
        ];
        for name in android_env_vars {
            env::var(name).expect(format!("{} should be set to build for Android", name).as_str());
        }
    }
}

fn build_tensorflow_with_bazel(tf_src_path: &str, config: &str) -> PathBuf {
    let os = env::var("CARGO_CFG_TARGET_OS").expect("Unable to get TARGET_OS");
    let output_path_buf;
    let bazel_output_path_buf;
    let bazel_target;
    if os != "ios" {
        let ext = if os != "macos" { "so" } else { "dylib" };
        bazel_output_path_buf = PathBuf::from(tf_src_path).join(format!(
            "bazel-bin/tensorflow/lite/c/libtensorflowlite_c.{}",
            ext
        ));
        bazel_target = "//tensorflow/lite/c:tensorflowlite_c";
        output_path_buf = out_dir().join(format!("libtensorflowlite_c.{}", ext));
    } else {
        bazel_output_path_buf = PathBuf::from(tf_src_path)
            .join("bazel-bin/tensorflow/lite/ios/TensorFlowLiteC_framework.zip");
        bazel_target = "//tensorflow/lite/ios:TensorFlowLiteC_framework";
        output_path_buf = out_dir().join("TensorFlowLiteC.framework");
    };

    if !output_path_buf.exists() {
        let python_bin_path = env::var("PYTHON_BIN_PATH").expect("Cannot read PYTHON_BIN_PATH");
        if !std::process::Command::new(&python_bin_path)
            .arg("configure.py")
            .current_dir(tf_src_path)
            .status()
            .expect(format!("Cannot execute python at {}", &python_bin_path).as_str())
            .success()
        {
            panic!("Cannot configure tensorflow")
        }
        let mut bazel = std::process::Command::new("bazel");
        bazel
            .arg("build")
            .arg("-c")
            .arg("opt")
            .arg(format!("--config={}", config))
            .arg(bazel_target)
            .current_dir(tf_src_path);
        if os == "ios" {
            bazel.args(&["--apple_bitcode=embedded", "--copt=-fembed-bitcode"]);
        }
        if !bazel.status().expect("Cannot execute bazel").success() {
            panic!("Cannot build TensorFlowLiteC");
        }
        if !bazel_output_path_buf.exists() {
            panic!(
                "Library/Framework not found in {}",
                bazel_output_path_buf.display()
            )
        }
        if os != "ios" {
            std::fs::copy(bazel_output_path_buf, &output_path_buf)
                .expect("Cannot copy bazel output");
        } else {
            let mut unzip = std::process::Command::new("unzip");
            unzip.args(&[
                "-q",
                bazel_output_path_buf.to_str().unwrap(),
                "-d",
                out_dir().to_str().unwrap(),
            ]);
            unzip.status().expect("Cannot execute unzip");
        }
    }
    output_path_buf
}

fn out_dir() -> PathBuf {
    PathBuf::from(env::var("OUT_DIR").unwrap())
}

fn main() {
    let out_path = out_dir();
    let os = env::var("CARGO_CFG_TARGET_OS").expect("Unable to get TARGET_OS");
    let arch = env::var("CARGO_CFG_TARGET_ARCH").expect("Unable to get TARGET_ARCH");
    let arch = match arch.as_str() {
        "aarch64" => String::from("arm64"),
        "armv7" => {
            if os == "android" {
                String::from("arm")
            } else {
                arch
            }
        }
        _ => arch,
    };
    if os != "ios" {
        println!("cargo:rustc-link-search=native={}", out_path.display());
        println!("cargo:rustc-link-lib=dylib=tensorflowlite_c");
    } else {
        println!("cargo:rustc-link-search=framework={}", out_path.display());
        println!("cargo:rustc-link-lib=framework=TensorFlowLiteC");
    }

    let config = if os == "android" || os == "ios" {
        format!("{}_{}", os, arch)
    } else {
        os
    };
    let tf_src_path = copy_tensorflow();
    check_and_set_envs();
    build_tensorflow_with_bazel(tf_src_path.to_str().unwrap(), config.as_str());

    // The bindgen::Builder is the main entry point
    // to bindgen, and lets you build up options for
    // the resulting bindings.
    let bindings = bindgen::Builder::default()
        // The input header we would like to generate
        // bindings for.
        .header("third_party/tensorflow/tensorflow/lite/c/c_api.h")
        .clang_arg("-Ithird_party/tensorflow")
        // Tell cargo to invalidate the built crate whenever any of the
        // included header files changed.
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        // Finish the builder and generate the bindings.
        .generate()
        // Unwrap the Result and panic on failure.
        .expect("Unable to generate bindings");

    // Write the bindings to the $OUT_DIR/bindings.rs file.
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}
