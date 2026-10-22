#![cfg_attr(target_family = "wasm", no_main)]

//! Custom WGSL shaders painted with `Window::paint_shader`: liquid metal, a dot matrix, chromatic
//! rings and an aurora, all animated by the time the view passes in.
//!
//! Press space to freeze and resume the animation. A frozen view requests no frames. Set
//! `PROCEDURAL_TIME` to a number of seconds to start frozen at that time, which makes the frame
//! reproducible.

#[path = "example_support/fonts.rs"]
mod example_support;

use gpui::{
    App, Bounds, Context, Corners, FocusHandle, Hsla, KeyDownEvent, LiquidGlass, Pixels, Render,
    Shader, ShaderInstance, ShaderSource, Window, WindowBounds, WindowOptions, canvas, div,
    prelude::*, px, rgb, rgba, size,
};
use gpui_platform::application;
use std::time::Instant;

/// The four example shaders, registered once.
#[derive(Clone, Copy)]
struct Shaders {
    liquid_metal: Shader,
    dot_matrix: Shader,
    rings: Shader,
    aurora: Shader,
}

impl Shaders {
    fn register(cx: &mut App) -> Self {
        let mut register = |label: &'static str, wgsl: &'static str| {
            cx.register_shader(ShaderSource::new(label, wgsl))
                .unwrap_or_else(|error| panic!("{error}"))
        };
        Self {
            liquid_metal: register("liquid_metal", include_str!("procedural/liquid_metal.wgsl")),
            dot_matrix: register("dot_matrix", include_str!("procedural/dot_matrix.wgsl")),
            rings: register("rings", include_str!("procedural/rings.wgsl")),
            aurora: register("aurora", include_str!("procedural/aurora.wgsl")),
        }
    }
}

struct Procedural {
    shaders: Shaders,
    focus_handle: FocusHandle,
    start: Instant,
    /// The time the animation is held at, when it is frozen.
    frozen: Option<f32>,
    /// How far the primary button is pressed in, eased towards whether it is hovered.
    pressure: f32,
    hovered: bool,
    last_frame: Option<Instant>,
}

impl Procedural {
    fn time(&self) -> f32 {
        self.frozen
            .unwrap_or_else(|| self.start.elapsed().as_secs_f32())
    }

    fn toggle_freeze(&mut self, cx: &mut Context<Self>) {
        self.frozen = match self.frozen {
            Some(time) => {
                self.start = Instant::now() - std::time::Duration::from_secs_f32(time);
                None
            }
            None => Some(self.time()),
        };
        cx.notify();
    }
}

/// An element that fills its parent with one shader instance.
fn fill(shader: Shader, instance: ShaderInstance, radius: Pixels) -> impl IntoElement {
    canvas(
        |_, _, _| {},
        move |bounds: Bounds<Pixels>, _, window, _| {
            window.paint_shader(bounds, Corners::all(radius), &shader, instance)
        },
    )
    .absolute()
    .top_0()
    .left_0()
    .size_full()
}

fn hsla(hex: u32) -> Hsla {
    rgb(hex).into()
}

fn label(text: &'static str) -> impl IntoElement {
    div().text_xs().text_color(rgba(0xe8ecf4a0)).child(text)
}

impl Render for Procedural {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let time = self.time();
        let now = Instant::now();
        let dt = self
            .last_frame
            .map_or(0.0, |last| now.duration_since(last).as_secs_f32());
        self.last_frame = Some(now);
        let target = if self.hovered { 1.0 } else { 0.0 };
        self.pressure += (target - self.pressure) * (dt * 8.0).min(1.0);
        let settling = (target - self.pressure).abs() > 0.001;
        if self.frozen.is_none() || settling {
            window.request_animation_frame();
        }

        let shaders = self.shaders;
        let aurora = ShaderInstance::at(time)
            .params(0, &[1.0, 760.0, 0.7, 0.7, 4.0])
            .color(0, hsla(0x05060f))
            .color(1, hsla(0x161a4d))
            .color(2, hsla(0x0c3b4c))
            .color(3, hsla(0x3b1d58));

        let rings = ShaderInstance::at(time)
            .params(0, &[-0.05, 1.25, 0.3, 70.0])
            .params(4, &[1.7, 2.6, 0.785, 64.0])
            .params(8, &[0.5, 2.4, 900.0])
            .color(0, hsla(0x1a130f))
            .color(1, hsla(0xf4f7ff))
            .color(2, hsla(0xffa14f));

        let wave = ShaderInstance::at(time)
            .params(0, &[4.0, 0.86, 0.0, 3.0])
            .params(4, &[0.0, 0.52, 1.0, 0.42])
            .params(8, &[0.5, 0.5, 0.8])
            .color(0, hsla(0x020c1d))
            .color(1, hsla(0xff0a7a))
            .color(2, hsla(0xff5cb0));

        let progress = (time * 0.12).fract();
        let progress_bar = ShaderInstance::at(time)
            .params(0, &[4.0, 0.75, 1.0, 2.0])
            .params(4, &[2.0, progress, 1.0, 0.035])
            .params(8, &[0.0, 0.0, 0.6])
            .color(0, rgba(0x0b0d1480))
            .color(1, hsla(0xf2f4f8))
            .color(2, hsla(0xff4d3d))
            .color(3, rgba(0xffffff1a));

        let shimmer = ShaderInstance::at(time)
            .params(0, &[3.0, 0.8, 1.0, 3.0])
            .params(4, &[4.0, 1.0, 1.0, 0.22])
            .params(8, &[0.0, 0.0, 0.0])
            .color(0, rgba(0x0b0d1480))
            .color(1, hsla(0x8fd3ff))
            .color(3, rgba(0xffffff12));

        let ripple = ShaderInstance::at(time)
            .params(0, &[5.0, 0.8, 1.0, 4.0])
            .params(4, &[1.0, 0.5 + 0.5 * (time * 0.4).sin().abs(), 1.0, 0.25])
            .params(8, &[0.5, 0.5, 0.7])
            .color(0, hsla(0x050608))
            .color(1, hsla(0xd8f3ff))
            .color(2, hsla(0x6ae1ff))
            .color(3, rgba(0xffffff0d));

        let scan = ShaderInstance::at(time)
            .params(0, &[3.0, 0.7, 1.0, 3.0])
            .params(4, &[4.0, 0.45, 0.5, 0.16])
            .color(0, hsla(0x07080c))
            .color(1, rgba(0x5fd4ffb0))
            .color(3, rgba(0xffffff0a));

        let sparkle = ShaderInstance::at(time)
            .params(0, &[3.0, 0.8, 1.0, 2.0])
            .params(4, &[3.0, 0.9, 1.0, 0.2])
            .color(0, rgba(0x0b0d1480))
            .color(1, hsla(0xffd166))
            .color(3, rgba(0xffffff10));

        let pressure = self.pressure;
        let metal = |rim: f32, interior: Hsla| {
            ShaderInstance::at(time)
                .params(0, &[rim, 1.0, -0.5, 1.05])
                .params(4, &[0.8, pressure])
                .color(0, hsla(0x0b0b0d))
                .color(1, hsla(0xf4f6fb))
                .color(2, interior)
        };
        let body = hsla(0x0c0c0f);
        let glass = LiquidGlass {
            blur: px(5.),
            refraction: px(26.),
            thickness: px(36.),
            chroma: 0.1,
            specular: 0.9,
            light_angle: -0.6,
            saturation: 0.45,
            tint: rgba(0xffffff0d).into(),
        };

        div()
            .id("procedural")
            .track_focus(&self.focus_handle)
            .on_key_down(cx.listener(|this, event: &KeyDownEvent, _, cx| {
                if event.keystroke.key == "space" {
                    this.toggle_freeze(cx);
                }
            }))
            .relative()
            .size_full()
            .overflow_hidden()
            .font_family("IBM Plex Sans")
            .text_color(rgb(0xf4f6fb))
            .child(fill(shaders.aurora, aurora, px(0.)))
            .child(
                div()
                    .absolute()
                    .top_0()
                    .left_0()
                    .size_full()
                    .p(px(44.))
                    .flex()
                    .flex_col()
                    .gap(px(28.))
                    .child(
                        div()
                            .flex()
                            .flex_col()
                            .gap_1()
                            .child(
                                div()
                                    .text_2xl()
                                    .font_weight(gpui::FontWeight::SEMIBOLD)
                                    .child("Procedural shaders"),
                            )
                            .child(label(
                                "Four WGSL sources, painted by one primitive. Space freezes time.",
                            )),
                    )
                    .child(
                        div()
                            .flex()
                            .gap(px(24.))
                            .h(px(320.))
                            .child(
                                div()
                                    .relative()
                                    .w(px(600.))
                                    .h_full()
                                    .child(fill(shaders.rings, rings, px(28.)))
                                    .child(
                                        div()
                                            .absolute()
                                            .bottom(px(20.))
                                            .left(px(24.))
                                            .child(label("rings · slats at 45°")),
                                    ),
                            )
                            .child(
                                div()
                                    .relative()
                                    .flex_1()
                                    .h_full()
                                    .flex()
                                    .items_center()
                                    .justify_center()
                                    .child(fill(shaders.dot_matrix, wave, px(28.)))
                                    .child(
                                        div()
                                            .relative()
                                            .text_3xl()
                                            .font_weight(gpui::FontWeight::SEMIBOLD)
                                            .child("Wave"),
                                    ),
                            ),
                    )
                    .child(
                        div()
                            .flex()
                            .gap(px(24.))
                            .items_center()
                            .child(
                                div()
                                    .id("primary")
                                    .relative()
                                    .w(px(220.))
                                    .h(px(56.))
                                    .flex()
                                    .items_center()
                                    .justify_center()
                                    .cursor_pointer()
                                    .on_hover(cx.listener(|this, hovered: &bool, _, cx| {
                                        this.hovered = *hovered;
                                        cx.notify();
                                    }))
                                    .child(fill(shaders.liquid_metal, metal(2.0, body), px(28.)))
                                    .child(
                                        div()
                                            .relative()
                                            .text_color(rgb(0xa9adb8))
                                            .child("Get Started"),
                                    ),
                            )
                            .child(
                                div()
                                    .relative()
                                    .size(px(56.))
                                    .flex()
                                    .items_center()
                                    .justify_center()
                                    .child(fill(shaders.liquid_metal, metal(2.0, body), px(28.)))
                                    .child(div().relative().text_color(rgb(0xa9adb8)).child("✦")),
                            )
                            .child(div().relative().w(px(150.)).h(px(56.)).child(fill(
                                shaders.liquid_metal,
                                metal(0.0, body).param(3, 0.7),
                                px(18.),
                            )))
                            .child(div().relative().size(px(120.)).child(fill(
                                shaders.dot_matrix,
                                ripple,
                                px(60.),
                            )))
                            .child(
                                div()
                                    .relative()
                                    .flex_1()
                                    .h(px(120.))
                                    .p(px(20.))
                                    .flex()
                                    .flex_col()
                                    .justify_end()
                                    .child(fill(shaders.dot_matrix, scan, px(24.)))
                                    .child(fill(
                                        shaders.liquid_metal,
                                        metal(1.4, gpui::transparent_black()),
                                        px(24.),
                                    ))
                                    .child(div().relative().child("Rim mode"))
                                    .child(div().relative().child(label(
                                        "a metal border over another surface",
                                    ))),
                            ),
                    )
                    .child(
                        div()
                            .flex()
                            .gap(px(24.))
                            .items_end()
                            .child(
                                div()
                                    .flex()
                                    .flex_col()
                                    .gap_2()
                                    .child(label("syncing"))
                                    .child(div().relative().w(px(420.)).h(px(28.)).child(fill(
                                        shaders.dot_matrix,
                                        progress_bar,
                                        px(8.),
                                    ))),
                            )
                            .child(
                                div()
                                    .flex()
                                    .flex_col()
                                    .gap_2()
                                    .child(label("thinking"))
                                    .child(div().relative().w(px(180.)).h(px(28.)).child(fill(
                                        shaders.dot_matrix,
                                        shimmer,
                                        px(14.),
                                    ))),
                            )
                            .child(
                                div()
                                    .flex()
                                    .flex_col()
                                    .gap_2()
                                    .child(label("sparkle"))
                                    .child(div().relative().w(px(140.)).h(px(28.)).child(fill(
                                        shaders.dot_matrix,
                                        sparkle,
                                        px(14.),
                                    ))),
                            ),
                    ),
            )
            .child(
                div()
                    .absolute()
                    .left(px(500.))
                    .top(px(282.))
                    .w(px(320.))
                    .h(px(190.))
                    .rounded(px(34.))
                    .liquid_glass(glass)
                    .p(px(24.))
                    .flex()
                    .flex_col()
                    .justify_end()
                    .gap_1()
                    .child(
                        div()
                            .text_lg()
                            .font_weight(gpui::FontWeight::SEMIBOLD)
                            .child("Liquid glass"),
                    )
                    .child(label("blur, lens rim, prism and highlight")),
            )
            .child(
                div()
                    .absolute()
                    .left(px(1080.))
                    .top(px(196.))
                    .size(px(104.))
                    .rounded(px(52.))
                    .liquid_glass(LiquidGlass {
                        blur: px(0.),
                        refraction: px(30.),
                        thickness: px(52.),
                        ..glass
                    }),
            )
    }
}

fn run_example() {
    application().run(|cx: &mut App| {
        if !example_support::load_fonts(cx) {
            return;
        }
        let shaders = Shaders::register(cx);
        let frozen = std::env::var("PROCEDURAL_TIME")
            .ok()
            .and_then(|time| time.parse().ok());
        let bounds = Bounds::centered(None, size(px(1280.), px(800.)), cx);
        cx.open_window(
            WindowOptions {
                window_bounds: Some(WindowBounds::Windowed(bounds)),
                focus: true,
                ..Default::default()
            },
            |window, cx| {
                cx.new(|cx| {
                    let focus_handle = cx.focus_handle();
                    window.focus(&focus_handle, cx);
                    Procedural {
                        shaders,
                        focus_handle,
                        start: Instant::now(),
                        frozen,
                        pressure: 0.0,
                        hovered: false,
                        last_frame: None,
                    }
                })
            },
        )
        .unwrap();
        cx.activate(true);
    });
}

#[cfg(not(target_family = "wasm"))]
fn main() {
    run_example();
}

#[cfg(target_family = "wasm")]
#[wasm_bindgen::prelude::wasm_bindgen(start)]
pub fn start() {
    gpui_platform::web_init();
    run_example();
}
