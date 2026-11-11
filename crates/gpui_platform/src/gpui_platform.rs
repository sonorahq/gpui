//! Convenience crate that re-exports GPUI's platform traits and the
//! `current_platform` constructor so consumers don't need `#[cfg]` gating.

pub use gpui::Platform;

#[cfg(target_family = "wasm")]
pub use gpui_web::WebBackendPreference;

use std::rc::Rc;

/// Returns a background executor for the current platform.
pub fn background_executor() -> gpui::BackgroundExecutor {
    current_platform(true).background_executor()
}

pub fn application() -> gpui::Application {
    gpui::Application::with_platform(current_platform(false))
}

pub fn headless() -> gpui::Application {
    gpui::Application::with_platform(current_platform(true))
}

/// Returns a Linux app that may switch among `allowed_modes`.
///
/// It starts windowed in the process's own environment, or headless if that names no allowed
/// display server. Set another initial mode with [`gpui::Application::with_windowing`], and
/// switch later with [`gpui::App::request_windowing`].
#[cfg(all(
    any(target_os = "linux", target_os = "freebsd"),
    not(target_env = "ohos")
))]
pub fn linux(allowed_modes: gpui::WindowingModes) -> gpui::Application {
    gpui::Application::with_platform(gpui_linux::linux_platform(allowed_modes))
}

#[cfg(target_family = "wasm")]
pub fn application_with_web_backend(backend_preference: WebBackendPreference) -> gpui::Application {
    let platform = Rc::new(gpui_web::WebPlatform::new_with_backend(
        true,
        backend_preference,
    ));
    gpui::Application::with_platform(platform)
}

/// Unlike `application`, this function returns a single-threaded web application.
#[cfg(target_family = "wasm")]
pub fn single_threaded_web() -> gpui::Application {
    let platform = Rc::new(gpui_web::WebPlatform::new(false));
    gpui::Application::with_platform(platform)
}

/// Initializes panic hooks and logging for the web platform.
/// Call this before running the application in a wasm_bindgen entrypoint.
#[cfg(target_family = "wasm")]
pub fn web_init() {
    console_error_panic_hook::set_once();
    gpui_web::init_logging();
}

/// Returns the default [`Platform`] for the current OS.
pub fn current_platform(headless: bool) -> Rc<dyn Platform> {
    cfg_select! {
        target_os = "ios" => gpui_ios::current_platform(headless),
        target_os = "macos" => Rc::new(gpui_macos::MacPlatform::new(headless)),
        target_os = "windows" => Rc::new(
            gpui_windows::WindowsPlatform::new(headless)
                .expect("failed to initialize Windows platform"),
        ),
        all(
            any(target_os = "linux", target_os = "freebsd"),
            not(target_env = "ohos")
        ) => gpui_linux::current_platform(headless),
        target_family = "wasm" => {
            let _ = headless;
            Rc::new(gpui_web::WebPlatform::new_with_backend(
                true,
                gpui_web::WebBackendPreference::Auto,
            ))
        },
    }
}

/// Returns the current platform's text system for benchmark contexts, built once per
/// thread and shared by every context on it.
///
/// `#[gpui::bench]` creates a context for every Criterion routine call. Building a whole
/// platform each time only to take its text system leaks the platform's threads on Linux
/// (a 20-worker dispatcher and a timer thread per call), which slows every later
/// benchmark in the process.
#[cfg(feature = "bench-support")]
pub fn bench_text_system() -> std::sync::Arc<dyn gpui::PlatformTextSystem> {
    thread_local! {
        static TEXT_SYSTEM: std::sync::Arc<dyn gpui::PlatformTextSystem> =
            current_platform(true).text_system();
    }
    TEXT_SYSTEM.with(|text_system| text_system.clone())
}

/// Returns a new [`HeadlessRenderer`] for the current platform, if available.
#[cfg(any(feature = "bench-support", feature = "test-support"))]
pub fn current_headless_renderer() -> anyhow::Result<Option<Box<dyn gpui::PlatformHeadlessRenderer>>>
{
    #[cfg(target_os = "macos")]
    {
        Ok(Some(Box::new(
            gpui_macos::metal_renderer::MetalHeadlessRenderer::new()?,
        )))
    }

    #[cfg(target_os = "linux")]
    {
        gpui_wgpu::WgpuHeadlessRenderer::new()
            .map(|renderer| Some(Box::new(renderer) as Box<dyn gpui::PlatformHeadlessRenderer>))
    }

    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        Ok(None)
    }
}
