//! Scenes every renderer draws in its own tests, each with a check on a few pixels.
//!
//! The wgpu, Metal and DirectX render tests include this file and hand [`run`] a closure that
//! draws a scene at a size. With `GPUI_RENDER_FIXTURE_OUT` set to a folder, every case is also
//! written there as `<case>.png`, so the images of two backends or two commits can be diffed.

use gpui::{
    Backdrop, Background, BorderStyle, Bounds, ContentMask, Corners, DevicePixels, Edges, Filter,
    Hsla, PaintedPath, PathBuilder, Quad, ScaledPixels, Scene, Shader, ShaderQuad, ShaderSource,
    Shadow, Size, point, px, size,
};

/// The width and height of every case's frame, in device pixels.
const SIDE: f32 = 64.;

/// What a case's check found wrong, or nothing.
type Verdict = Result<(), String>;

/// One scene the renderers must draw alike and what a few of its pixels must look like.
pub struct Case {
    /// The case's name, which is also the name of its PNG.
    pub name: String,
    /// The size of the frame the scene is drawn into.
    pub size: Size<DevicePixels>,
    /// The scene, finished and ready to draw.
    pub scene: Scene,
    check: Box<dyn Fn(&image::RgbaImage) -> Verdict>,
}

impl Case {
    fn new(
        name: impl Into<String>,
        scene: Scene,
        check: impl Fn(&image::RgbaImage) -> Verdict + 'static,
    ) -> Self {
        Self {
            name: name.into(),
            size: size(DevicePixels(SIDE as i32), DevicePixels(SIDE as i32)),
            scene,
            check: Box::new(check),
        }
    }
}

/// Every case, in a fixed order.
pub fn cases() -> Vec<Case> {
    let mut cases = Vec::new();
    for blur in [3., 12.] {
        for depth in [0, 1, 2] {
            for inside in [false, true] {
                let place = match inside {
                    true => "inside",
                    false => "outside",
                };
                cases.push(Case::new(
                    format!("nested_backdrop_blur{blur}_depth{depth}_{place}"),
                    nested_backdrop_scene(depth, blur, inside),
                    nested_backdrop_check,
                ));
            }
        }
    }
    cases.push(layer_blur("layer_blur_gaussian", 3.));
    cases.push(layer_blur("layer_blur_pyramid", 12.));
    cases.push(mask_fade());
    cases.push(layer_scale_translate());
    cases.push(custom_shader());
    cases.push(opaque_rectangles());
    cases.push(path_in_blurred_layer());
    cases.push(shadows());
    cases.push(overscan());
    cases
}

/// Draws every case with `render`, writes its PNG when `GPUI_RENDER_FIXTURE_OUT` is set and
/// checks it. `suffix` is appended to each PNG's name so a backend can write several variants.
/// `known_failures` names the cases the backend is known to draw wrong. Those must still fail
/// their check, so fixing the backend shows up as an error until the name is removed. Every case
/// is drawn even after one fails, and the error lists them all.
pub fn run(
    suffix: &str,
    known_failures: &[&str],
    mut render: impl FnMut(&Scene, Size<DevicePixels>) -> anyhow::Result<image::RgbaImage>,
) -> anyhow::Result<()> {
    let out = std::env::var_os("GPUI_RENDER_FIXTURE_OUT").map(std::path::PathBuf::from);
    if let Some(out) = &out {
        std::fs::create_dir_all(out)?;
    }
    let mut failures = Vec::new();
    for case in cases() {
        let image = match render(&case.scene, case.size) {
            Ok(image) => image,
            Err(error) => {
                failures.push(format!("{}: {error:#}", case.name));
                continue;
            }
        };
        if let Some(out) = &out {
            image.save(out.join(format!("{}{suffix}.png", case.name)))?;
        }
        let known = known_failures.contains(&case.name.as_str());
        match ((case.check)(&image), known) {
            (Ok(()), false) => {}
            (Err(message), false) => failures.push(format!("{}: {message}", case.name)),
            (Ok(()), true) => failures.push(format!(
                "{}: passes now, remove it from the known failures",
                case.name
            )),
            (Err(message), true) => eprintln!("{}: known failure: {message}", case.name),
        }
    }
    anyhow::ensure!(failures.is_empty(), "{}", failures.join("\n"));
    Ok(())
}

/// A split red and blue background behind one or two scaled backdrop layers.
fn nested_backdrop_scene(depth: usize, blur: f32, background_in_layer: bool) -> Scene {
    let mut scene = Scene::default();
    let full = frame();
    for level in 0..depth {
        if level == 0 && !background_in_layer {
            split_background(&mut scene);
        }
        scene.push_filter(
            full,
            full.center(),
            full,
            Filter {
                scale: 0.9,
                ..Filter::default()
            },
        );
        if level == 0 && background_in_layer {
            split_background(&mut scene);
        }
    }
    if depth == 0 {
        split_background(&mut scene);
    }
    scene.insert_primitive(backdrop(rect(16., 16., 32., 32.), blur, full));
    for _ in 0..depth {
        scene.pop_filter();
    }
    scene.finish();
    scene
}

/// The backdrop sampled both sides of the ancestor's color boundary.
fn nested_backdrop_check(image: &image::RgbaImage) -> Verdict {
    expect(image, 31, 32, "a blur of red and blue", |[r, _, b, a]| {
        r > 20 && b > 20 && a > 245
    })?;
    expect_color(image, 8, 32, RED)?;
    expect_color(image, 56, 32, BLUE)
}

/// A red square blurred inside a layer on white, which runs the gaussian at a sigma of 3 and
/// the pyramid at 12.
fn layer_blur(name: &str, blur: f32) -> Case {
    let mut scene = white_scene();
    scene.push_filter(
        frame(),
        frame().center(),
        frame(),
        Filter {
            blur,
            ..Filter::default()
        },
    );
    scene.insert_primitive(quad(rect(16., 16., 32., 32.), RED));
    scene.pop_filter();
    scene.finish();
    Case::new(name, scene, move |image| {
        expect(image, 32, 32, "red at the centre", |[r, g, b, _]| {
            r > 245 && g < 120 && b < 120
        })?;
        expect(image, 14, 32, "a soft left edge", |[r, g, _, _]| {
            r > 245 && g > 60 && g < 235
        })?;
        expect(image, 32, 49, "a soft bottom edge", |[r, g, _, _]| {
            r > 245 && g > 60 && g < 235
        })?;
        expect(image, 0, 0, "white in the corner", |[r, g, b, _]| {
            r > 240 && g > 225 && b > 225
        })
    })
}

/// A red layer on white that fades out over its top and bottom 16 pixels.
fn mask_fade() -> Case {
    let mut scene = white_scene();
    scene.push_filter(
        frame(),
        frame().center(),
        frame(),
        Filter {
            fade_top: 16.,
            fade_bottom: 16.,
            ..Filter::default()
        },
    );
    scene.insert_primitive(quad(frame(), RED));
    scene.pop_filter();
    scene.finish();
    Case::new("mask_fade", scene, |image| {
        expect_color(image, 32, 32, RED)?;
        expect(image, 32, 0, "faded out at the top", |[r, g, _, _]| {
            r > 245 && g > 200
        })?;
        expect(image, 32, 63, "faded out at the bottom", |[r, g, _, _]| {
            r > 245 && g > 200
        })?;
        expect(image, 32, 8, "half faded at the top", |[r, g, _, _]| {
            r > 245 && g > 50 && g < 205
        })?;
        expect(image, 32, 55, "half faded at the bottom", |[r, g, _, _]| {
            r > 245 && g > 50 && g < 205
        })
    })
}

/// A red layer halved around its centre and moved 8 pixels right and down, so it lands on
/// 32..48 on both axes.
fn layer_scale_translate() -> Case {
    let mut scene = white_scene();
    let source = rect(16., 16., 32., 32.);
    scene.push_filter(
        source,
        source.center(),
        frame(),
        Filter {
            scale: 0.5,
            translate: point(ScaledPixels(8.), ScaledPixels(8.)),
            ..Filter::default()
        },
    );
    scene.insert_primitive(quad(source, RED));
    scene.pop_filter();
    scene.finish();
    Case::new("layer_scale_translate", scene, |image| {
        expect_color(image, 40, 40, RED)?;
        expect_color(image, 33, 33, RED)?;
        expect_color(image, 46, 46, RED)?;
        expect_color(image, 30, 40, WHITE)?;
        expect_color(image, 40, 30, WHITE)?;
        expect_color(image, 49, 40, WHITE)?;
        expect_color(image, 40, 49, WHITE)?;
        expect_color(image, 20, 20, WHITE)
    })
}

/// Custom shaders between quads, an opaque one and two translucent ones, with a path in the
/// same frame so a path's globals cannot be mistaken for the shader's alpha mode.
fn custom_shader() -> Case {
    let shader = Shader::new(ShaderSource::new(
        "fixture_flat",
        "fn shade(input: ShaderInput) -> vec4<f32> { return input.colors[0]; }",
    ))
    .expect("the fixture shader is valid")
    .id();
    let instance = |bounds: Bounds<ScaledPixels>, color: [f32; 4], opacity: f32| ShaderQuad {
        order: 0,
        shader,
        opacity,
        time: 0.,
        bounds,
        content_mask: ContentMask { bounds: frame() },
        corner_radii: Corners::default(),
        scale_factor: 1.,
        pad: [0; 3],
        params: [0.; gpui::SHADER_PARAM_COUNT],
        colors: [color, [0.; 4], [0.; 4], [0.; 4]],
    };
    let mut scene = white_scene();
    scene.insert_primitive(quad(rect(0., 0., 32., 32.), RED));
    scene.insert_primitive(instance(rect(0., 0., 32., 32.), [0., 0., 1., 1.], 1.));
    scene.insert_primitive(quad(rect(16., 0., 16., 32.), GREEN));
    scene.insert_primitive(instance(rect(32., 0., 32., 16.), [0., 0., 1., 1.], 0.5));
    scene.insert_primitive(instance(rect(32., 16., 32., 16.), [0., 0., 1., 0.5], 1.));
    scene.insert_primitive(triangle(0., 32., 32., RED));
    scene.finish();
    Case::new("custom_shader", scene, |image| {
        expect_color(image, 8, 16, BLUE)?;
        expect_color(image, 24, 16, GREEN)?;
        expect(image, 48, 8, "blue at half opacity", half_blue)?;
        expect(image, 48, 24, "blue at half alpha", half_blue)?;
        expect_color(image, 4, 36, RED)?;
        expect_color(image, 28, 60, WHITE)?;
        expect_color(image, 48, 48, WHITE)
    })
}

/// An opaque rectangle under a backdrop, one above it and one inside a scaled layer, which a
/// depth pre-pass that culls opaque quads without drawing them would lose.
fn opaque_rectangles() -> Case {
    let mut scene = white_scene();
    scene.insert_primitive(quad(rect(4., 4., 24., 24.), RED));
    scene.insert_primitive(backdrop(rect(0., 0., 64., 36.), 2., frame()));
    scene.insert_primitive(quad(rect(36., 8., 20., 20.), BLUE));
    let layer = rect(16., 40., 32., 20.);
    scene.push_filter(
        layer,
        layer.center(),
        frame(),
        Filter {
            scale: 0.9,
            ..Filter::default()
        },
    );
    scene.insert_primitive(quad(layer, GREEN));
    scene.pop_filter();
    scene.finish();
    Case::new("opaque_rectangles", scene, |image| {
        expect(image, 16, 16, "red under the backdrop", |[r, g, b, _]| {
            r > 245 && g < 40 && b < 40
        })?;
        expect(
            image,
            4,
            16,
            "the backdrop blurring red's edge",
            |[r, g, _, _]| r > 245 && g > 40 && g < 215,
        )?;
        expect_color(image, 46, 18, BLUE)?;
        expect_color(image, 37, 9, BLUE)?;
        expect_color(image, 32, 50, GREEN)?;
        expect_color(image, 2, 62, WHITE)
    })
}

/// A filled red square drawn as a path inside a layer blurred at a sigma of 3.
fn path_in_blurred_layer() -> Case {
    let mut scene = white_scene();
    scene.push_filter(
        frame(),
        frame().center(),
        frame(),
        Filter {
            blur: 3.,
            ..Filter::default()
        },
    );
    scene.insert_primitive(polygon(
        &[(16., 16.), (48., 16.), (48., 48.), (16., 48.)],
        RED,
    ));
    scene.pop_filter();
    scene.finish();
    Case::new("path_in_blurred_layer", scene, |image| {
        expect(image, 32, 32, "red at the centre", |[r, g, b, _]| {
            r > 245 && g < 60 && b < 60
        })?;
        expect(image, 15, 32, "a soft left edge", |[r, g, _, _]| {
            r > 245 && g > 60 && g < 235
        })?;
        expect(image, 32, 48, "a soft bottom edge", |[r, g, _, _]| {
            r > 245 && g > 60 && g < 235
        })?;
        expect_color(image, 2, 2, WHITE)
    })
}

/// A drop shadow with a spread around a red square on the left, and an inset shadow with a
/// spread inside a green square on the right.
fn shadows() -> Case {
    let mut scene = white_scene();
    // A drop shadow paints its shape and three blur radii around it, an inset one its element.
    let shadow = |element: Bounds<ScaledPixels>, shape: Bounds<ScaledPixels>, inset: bool| Shadow {
        order: 0,
        blur_radius: ScaledPixels(2.),
        bounds: match inset {
            true => element,
            false => shape.dilate(ScaledPixels(6.)),
        },
        corner_radii: Corners::default(),
        content_mask: ContentMask { bounds: frame() },
        color: BLACK,
        shape_bounds: shape,
        shape_corner_radii: Corners::default(),
        inset: inset.into(),
        element_corner_smoothing: 0.,
        element_bounds: element,
        element_corner_radii: Corners::default(),
    };
    let dropped = rect(10., 24., 16., 16.);
    scene.insert_primitive(shadow(dropped, dropped.dilate(ScaledPixels(4.)), false));
    scene.insert_primitive(quad(dropped, RED));
    let inset = rect(40., 22., 20., 20.);
    scene.insert_primitive(quad(inset, GREEN));
    scene.insert_primitive(shadow(inset, inset.dilate(ScaledPixels(-4.)), true));
    scene.finish();
    Case::new("shadows", scene, |image| {
        expect_color(image, 18, 32, RED)?;
        expect(image, 8, 32, "the spread shadow", |[r, g, b, _]| {
            r < 90 && g < 90 && b < 90
        })?;
        expect(image, 18, 22, "the spread shadow above", |[r, g, b, _]| {
            r < 90 && g < 90 && b < 90
        })?;
        expect(image, 1, 32, "no shadow past the blur", |[r, g, b, _]| {
            r > 230 && g > 230 && b > 230
        })?;
        expect(image, 41, 32, "the inset shadow", |[r, g, b, _]| {
            r < 40 && g < 90 && b < 40
        })?;
        expect_color(image, 50, 32, GREEN)?;
        expect_color(image, 38, 32, WHITE)?;
        expect_color(image, 62, 32, WHITE)?;
        expect_color(image, 50, 20, WHITE)
    })
}

/// A red frame with blue drawn past its right edge and green past its bottom edge, under a
/// backdrop in the bottom right corner. With the overscan honoured the backdrop blurs blue and
/// green in at the edges, without it the edges stay red.
fn overscan() -> Case {
    let mut scene = Scene::default();
    let past = rect(0., 0., SIDE + 32., SIDE + 32.);
    scene.overscan = size(DevicePixels(32), DevicePixels(32));
    let with_mask = |bounds: Bounds<ScaledPixels>, color: Hsla| Quad {
        content_mask: ContentMask { bounds: past },
        ..quad(bounds, color)
    };
    scene.insert_primitive(with_mask(frame(), RED));
    scene.insert_primitive(with_mask(rect(SIDE, 0., 32., SIDE + 32.), BLUE));
    scene.insert_primitive(with_mask(rect(0., SIDE, SIDE, 32.), GREEN));
    scene.insert_primitive(backdrop(rect(32., 32., 32., 32.), 4., past));
    scene.finish();
    Case::new("overscan", scene, |image| {
        expect_color(image, 8, 8, RED)?;
        expect_color(image, 63, 8, RED)?;
        expect(
            image,
            63,
            44,
            "blue blurred in from the right",
            |[_, _, b, _]| b > 30,
        )?;
        expect(
            image,
            44,
            63,
            "green blurred in from below",
            |[_, g, _, _]| g > 30,
        )
    })
}

const RED: Hsla = gpui::red();
const BLUE: Hsla = gpui::blue();
const WHITE: Hsla = gpui::white();
const BLACK: Hsla = gpui::black();
const GREEN: Hsla = Hsla {
    h: 1. / 3.,
    s: 1.,
    l: 0.5,
    a: 1.,
};

/// The whole frame.
fn frame() -> Bounds<ScaledPixels> {
    rect(0., 0., SIDE, SIDE)
}

fn rect(x: f32, y: f32, width: f32, height: f32) -> Bounds<ScaledPixels> {
    Bounds::new(
        point(ScaledPixels(x), ScaledPixels(y)),
        size(ScaledPixels(width), ScaledPixels(height)),
    )
}

/// A solid quad with square corners and no border, clipped to its own bounds.
fn quad(bounds: Bounds<ScaledPixels>, color: Hsla) -> Quad {
    Quad {
        order: 0,
        border_style: BorderStyle::Solid,
        bounds,
        content_mask: ContentMask { bounds },
        background: color.into(),
        border_color: color,
        corner_radii: Corners::default(),
        border_widths: Edges::default(),
        corner_smoothing: 0.,
        pad: 0,
    }
}

fn backdrop(bounds: Bounds<ScaledPixels>, blur: f32, mask: Bounds<ScaledPixels>) -> Backdrop {
    Backdrop {
        order: 0,
        corner_smoothing: 0.,
        blur,
        opacity: 1.,
        bounds,
        content_mask: ContentMask { bounds: mask },
        corner_radii: Corners::default(),
    }
}

/// A filled right triangle with its right angle at the top left, as a path.
fn triangle(x: f32, y: f32, side: f32, color: Hsla) -> PaintedPath {
    polygon(&[(x, y), (x + side, y), (x, y + side)], color)
}

/// The polygon through `corners`, filled with `color` and clipped to the frame.
fn polygon(corners: &[(f32, f32)], color: Hsla) -> PaintedPath {
    let mut builder = PathBuilder::fill();
    for (index, &(x, y)) in corners.iter().enumerate() {
        match index {
            0 => builder.move_to(point(px(x), px(y))),
            _ => builder.line_to(point(px(x), px(y))),
        }
    }
    builder.close();
    let path = builder.build().expect("a polygon builds");
    let frame = Bounds::new(point(px(0.), px(0.)), size(px(SIDE), px(SIDE)));
    PaintedPath::new(
        path,
        1.,
        ContentMask { bounds: frame },
        Background::from(color),
    )
}

/// A scene that starts with an opaque white frame.
fn white_scene() -> Scene {
    let mut scene = Scene::default();
    scene.insert_primitive(quad(frame(), WHITE));
    scene
}

/// Paints a high contrast boundary so a missing or unblurred backdrop fails visibly.
fn split_background(scene: &mut Scene) {
    scene.insert_primitive(quad(rect(0., 0., 32., 64.), RED));
    scene.insert_primitive(quad(rect(32., 0., 32., 64.), BLUE));
}

/// Blue at half strength over white, within a few levels.
fn half_blue([r, g, b, _]: [u8; 4]) -> bool {
    (115..=140).contains(&r) && (115..=140).contains(&g) && b > 245
}

/// Checks that pixel `x`, `y` satisfies `test`, describing it as `what` if not.
fn expect(
    image: &image::RgbaImage,
    x: u32,
    y: u32,
    what: &str,
    test: impl Fn([u8; 4]) -> bool,
) -> Verdict {
    let pixel = image.get_pixel(x, y).0;
    match test(pixel) {
        true => Ok(()),
        false => Err(format!("({x}, {y}) is {pixel:?}, expected {what}")),
    }
}

/// Checks that pixel `x`, `y` is opaque `color` within three levels per channel.
fn expect_color(image: &image::RgbaImage, x: u32, y: u32, color: Hsla) -> Verdict {
    let gpui::Rgba { r, g, b, a } = color.into();
    let expected = [r, g, b, a].map(|channel| (channel * 255.).round() as u8);
    expect(image, x, y, &format!("{expected:?}"), |pixel| {
        pixel
            .iter()
            .zip(expected)
            .all(|(actual, expected)| actual.abs_diff(expected) <= 3)
    })
}
