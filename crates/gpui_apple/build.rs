#![allow(clippy::disallowed_methods, reason = "build scripts are exempt")]

fn main() {
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    apple_build::run();
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
mod apple_build {
    use std::{
        env,
        path::{Path, PathBuf},
    };

    use cbindgen::Config;

    pub fn run() {
        let header_path = generate_shader_bindings();

        // Compiled here whenever the Metal toolchain is installed, even with `runtime_shaders`
        // on, which only allows compiling at launch on a machine without it.
        println!("cargo::rustc-check-cfg=cfg(gpui_runtime_shaders)");
        if let Err(error) = compile_metal_shaders(&header_path) {
            if cfg!(feature = "runtime_shaders") {
                println!("cargo::warning=compiling Metal shaders at runtime: {error}");
                emit_stitched_shaders(&header_path);
                println!("cargo::rustc-cfg=gpui_runtime_shaders");
            } else {
                println!("cargo::error={error}");
                std::process::exit(1);
            }
        }
    }

    fn generate_shader_bindings() -> PathBuf {
        let output_path = PathBuf::from(env::var("OUT_DIR").unwrap()).join("scene.h");

        let gpui_dir = find_gpui_crate_dir();

        let mut config = Config {
            include_guard: Some("SCENE_H".into()),
            language: cbindgen::Language::C,
            no_includes: true,
            ..Default::default()
        };
        config.export.include.extend([
            "Bounds".into(),
            "Corners".into(),
            "Edges".into(),
            "Size".into(),
            "Pixels".into(),
            "PointF".into(),
            "Hsla".into(),
            "ContentMask".into(),
            "Uniforms".into(),
            "AtlasTile".into(),
            "PathRasterizationInputIndex".into(),
            "PathVertex_ScaledPixels".into(),
            "PathRasterizationVertex".into(),
            "ShadowInputIndex".into(),
            "Shadow".into(),
            "QuadInputIndex".into(),
            "Underline".into(),
            "UnderlineInputIndex".into(),
            "Quad".into(),
            "BorderStyle".into(),
            "SpriteInputIndex".into(),
            "MonochromeSprite".into(),
            "PolychromeSprite".into(),
            "PathSprite".into(),
            "SurfaceInputIndex".into(),
            "SurfaceBounds".into(),
            "TransformationMatrix".into(),
            "Backdrop".into(),
            "BlurParams".into(),
            "MaskParams".into(),
            "FilterInputIndex".into(),
        ]);
        config.no_includes = true;
        config.enumeration.prefix_with_name = true;

        let mut builder = cbindgen::Builder::new();

        let crate_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());

        // Source files from gpui that define types used in shaders
        let gpui_src_paths = [
            gpui_dir.join("src/scene.rs"),
            gpui_dir.join("src/geometry.rs"),
            gpui_dir.join("src/color.rs"),
            gpui_dir.join("src/window.rs"),
            gpui_dir.join("src/platform.rs"),
        ];

        // Source files from this crate
        let local_src_paths = [crate_dir.join("src/metal_renderer.rs")];

        for src_path in gpui_src_paths.iter().chain(local_src_paths.iter()) {
            println!("cargo:rerun-if-changed={}", src_path.display());
            builder = builder.with_src(src_path);
        }

        builder
            .with_config(config)
            .generate()
            .expect("Unable to generate bindings")
            .write_to_file(&output_path);

        output_path
    }

    /// Locate the gpui crate directory relative to this crate. Resolved at
    /// build-script runtime against this crate's manifest dir, so no checkout
    /// path is baked into a compiled artifact (which corgi rejects).
    fn find_gpui_crate_dir() -> PathBuf {
        PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap()).join("../gpui")
    }

    /// To enable runtime compilation, we need to "stitch" the shaders file with the generated header
    /// so that it is self-contained.
    fn emit_stitched_shaders(header_path: &Path) {
        fn stitch_header(header: &Path, shader_path: &Path) -> std::io::Result<PathBuf> {
            let header_contents = std::fs::read_to_string(header)?;
            let shader_contents = std::fs::read_to_string(shader_path)?;
            let stitched_contents = format!("{header_contents}\n{shader_contents}");
            let out_path =
                PathBuf::from(env::var("OUT_DIR").unwrap()).join("stitched_shaders.metal");
            std::fs::write(&out_path, stitched_contents)?;
            Ok(out_path)
        }
        let shader_source_path = "./src/shaders.metal";
        let shader_path = PathBuf::from(shader_source_path);
        stitch_header(header_path, &shader_path).unwrap();
        println!("cargo:rerun-if-changed={shader_source_path}");
    }

    fn compile_metal_shaders(header_path: &Path) -> Result<(), String> {
        use std::process::Command;

        // Build scripts run on the host, so the target platform must come from
        // Cargo's environment rather than `cfg!`. The target environment, not the
        // target name, identifies simulators: `x86_64-apple-ios` has no `-sim` suffix.
        let target_os = env::var("CARGO_CFG_TARGET_OS")
            .expect("Cargo sets CARGO_CFG_TARGET_OS for build scripts");
        let target_env = env::var("CARGO_CFG_TARGET_ENV")
            .expect("Cargo sets CARGO_CFG_TARGET_ENV for build scripts");
        let (sdk, minimum_version_argument) = match (target_os.as_str(), target_env.as_str()) {
            ("macos", _) => ("macosx", "-mmacosx-version-min=10.15.7"),
            ("ios", "sim") => ("iphonesimulator", "-mios-simulator-version-min=15.0"),
            ("ios", "") => ("iphoneos", "-mios-version-min=15.0"),
            _ => {
                return Err(format!(
                    "unsupported Metal shader target: target_os={target_os}, target_env={target_env}"
                ));
            }
        };

        let shader_path = "./src/shaders.metal";
        let air_output_path = PathBuf::from(env::var("OUT_DIR").unwrap()).join("shaders.air");
        let metallib_output_path =
            PathBuf::from(env::var("OUT_DIR").unwrap()).join("shaders.metallib");
        println!("cargo:rerun-if-changed={}", shader_path);

        // The metal compiler records the resolved absolute path of its input
        // unconditionally. Compile a copy staged in OUT_DIR so the recorded
        // location is the build's canonical output directory, never the
        // checkout (corgi rejects artifacts that embed the build path).
        let staged_shader_path = PathBuf::from(env::var("OUT_DIR").unwrap()).join("shaders.metal");
        std::fs::copy(shader_path, &staged_shader_path).unwrap();

        let output = Command::new("xcrun")
            .args([
                "-sdk",
                sdk,
                "metal",
                "-gline-tables-only",
                minimum_version_argument,
                "-MO",
                "-c",
            ])
            .arg(&staged_shader_path)
            .args(["-include", header_path.to_str().unwrap(), "-o"])
            .arg(&air_output_path)
            .output()
            .map_err(|error| format!("could not run xcrun: {error}"))?;

        if !output.status.success() {
            return Err(format!(
                "metal shader compilation failed:\n{}",
                String::from_utf8_lossy(&output.stderr)
            ));
        }

        let output = Command::new("xcrun")
            .args(["-sdk", sdk, "metallib"])
            .arg(&air_output_path)
            .arg("-o")
            .arg(metallib_output_path)
            .output()
            .map_err(|error| format!("could not run xcrun: {error}"))?;

        if !output.status.success() {
            return Err(format!(
                "metallib compilation failed:\n{}",
                String::from_utf8_lossy(&output.stderr)
            ));
        }

        // The .air intermediate records the compiler's working directory in
        // its debug info; the metallib built from it does not. Nothing reads
        // the .air after this point, so drop it rather than leave a
        // checkout-path-bearing file in OUT_DIR.
        std::fs::remove_file(&air_output_path).unwrap();
        Ok(())
    }
}
