use std::sync::Arc;

use crate::{
    App, Bounds, DevicePixels, Element, ElementId, GlobalElementId, InspectorElementId,
    IntoElement, LayoutId, ObjectFit, Pixels, Size, Style, StyleRefinement, Styled, Window, size,
};
use anyhow::{Result, ensure};
#[cfg(target_os = "macos")]
use core_video::pixel_buffer::CVPixelBuffer;
use refineable::Refineable;

/// A source of a surface's content.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SurfaceSource {
    /// A macOS image buffer from CoreVideo, in the bi-planar 4:2:0 full range layout.
    #[cfg(target_os = "macos")]
    Surface(CVPixelBuffer),
    /// A frame in system memory, drawn on every platform.
    Nv12(Nv12Frame),
}

#[cfg(target_os = "macos")]
impl From<CVPixelBuffer> for SurfaceSource {
    fn from(value: CVPixelBuffer) -> Self {
        SurfaceSource::Surface(value)
    }
}

impl From<Nv12Frame> for SurfaceSource {
    fn from(value: Nv12Frame) -> Self {
        SurfaceSource::Nv12(value)
    }
}

impl SurfaceSource {
    /// The size of the picture, in pixels of the source.
    pub fn size(&self) -> Size<DevicePixels> {
        match self {
            #[cfg(target_os = "macos")]
            SurfaceSource::Surface(buffer) => size(
                DevicePixels(buffer.get_width() as i32),
                DevicePixels(buffer.get_height() as i32),
            ),
            SurfaceSource::Nv12(frame) => size(
                DevicePixels(frame.width as i32),
                DevicePixels(frame.height as i32),
            ),
        }
    }
}

/// Which values a frame's samples span. Video decoders usually hand out the narrower video
/// range, where black is 16 and white 235.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Nv12Range {
    /// Luma and chroma use the whole 0 to 255 span.
    Full,
    /// Luma spans 16 to 235 and chroma 16 to 240.
    Video,
}

/// One video frame in the NV12 layout most decoders produce: a full size plane of luma, then a
/// half size plane of interleaved Cb and Cr. Rows are tightly packed. Cloning only clones the
/// handles to the planes.
#[derive(Clone)]
pub struct Nv12Frame {
    width: u32,
    height: u32,
    range: Nv12Range,
    y: Arc<[u8]>,
    cb_cr: Arc<[u8]>,
}

impl Nv12Frame {
    /// A frame of `width` by `height` pixels. The luma plane must hold `width * height` bytes
    /// and the chroma plane two bytes for every pixel of a half size image, rounded up.
    pub fn new(
        width: u32,
        height: u32,
        range: Nv12Range,
        y: impl Into<Arc<[u8]>>,
        cb_cr: impl Into<Arc<[u8]>>,
    ) -> Result<Self> {
        let (y, cb_cr) = (y.into(), cb_cr.into());
        ensure!(width > 0 && height > 0, "an nv12 frame cannot be empty");
        ensure!(
            y.len() == width as usize * height as usize,
            "an nv12 luma plane of {width}x{height} needs {} bytes, not {}",
            width as usize * height as usize,
            y.len()
        );
        let chroma = (width.div_ceil(2) as usize) * (height.div_ceil(2) as usize) * 2;
        ensure!(
            cb_cr.len() == chroma,
            "an nv12 chroma plane of {width}x{height} needs {chroma} bytes, not {}",
            cb_cr.len()
        );
        Ok(Self {
            width,
            height,
            range,
            y,
            cb_cr,
        })
    }

    /// The width of the frame, and of its luma plane.
    pub fn width(&self) -> u32 {
        self.width
    }

    /// The height of the frame, and of its luma plane.
    pub fn height(&self) -> u32 {
        self.height
    }

    /// The width of the chroma plane, in Cb and Cr pairs.
    pub fn chroma_width(&self) -> u32 {
        self.width.div_ceil(2)
    }

    /// The height of the chroma plane.
    pub fn chroma_height(&self) -> u32 {
        self.height.div_ceil(2)
    }

    /// Which values the samples span.
    pub fn range(&self) -> Nv12Range {
        self.range
    }

    /// The luma plane, one byte a pixel.
    pub fn y(&self) -> &[u8] {
        &self.y
    }

    /// The chroma plane, a Cb and a Cr byte for every two by two block of pixels.
    pub fn cb_cr(&self) -> &[u8] {
        &self.cb_cr
    }
}

impl PartialEq for Nv12Frame {
    fn eq(&self, other: &Self) -> bool {
        self.width == other.width
            && self.height == other.height
            && self.range == other.range
            && Arc::ptr_eq(&self.y, &other.y)
            && Arc::ptr_eq(&self.cb_cr, &other.cb_cr)
    }
}

impl Eq for Nv12Frame {}

impl std::fmt::Debug for Nv12Frame {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Nv12Frame")
            .field("width", &self.width)
            .field("height", &self.height)
            .field("range", &self.range)
            .finish_non_exhaustive()
    }
}

/// A surface element.
pub struct Surface {
    source: SurfaceSource,
    object_fit: ObjectFit,
    style: StyleRefinement,
}

/// Create a new surface element.
pub fn surface(source: impl Into<SurfaceSource>) -> Surface {
    Surface {
        source: source.into(),
        object_fit: ObjectFit::Contain,
        style: Default::default(),
    }
}

impl Surface {
    /// Set the object fit for the image.
    pub fn object_fit(mut self, object_fit: ObjectFit) -> Self {
        self.object_fit = object_fit;
        self
    }
}

impl Element for Surface {
    type RequestLayoutState = ();
    type PrepaintState = ();

    fn id(&self) -> Option<ElementId> {
        None
    }

    fn source_location(&self) -> Option<&'static core::panic::Location<'static>> {
        None
    }

    fn request_layout(
        &mut self,
        _global_id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        window: &mut Window,
        cx: &mut App,
    ) -> (LayoutId, Self::RequestLayoutState) {
        let mut style = Style::default();
        style.refine(&self.style);
        let layout_id = window.request_layout(style, [], cx);
        (layout_id, ())
    }

    fn prepaint(
        &mut self,
        _global_id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        _bounds: Bounds<Pixels>,
        _request_layout: &mut Self::RequestLayoutState,
        _window: &mut Window,
        _cx: &mut App,
    ) -> Self::PrepaintState {
    }

    fn paint(
        &mut self,
        _global_id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        bounds: Bounds<Pixels>,
        _: &mut Self::RequestLayoutState,
        _: &mut Self::PrepaintState,
        window: &mut Window,
        _: &mut App,
    ) {
        let new_bounds = self.object_fit.get_bounds(bounds, self.source.size());
        let mut style = Style::default();
        style.refine(&self.style);
        let corner_radii = style.corner_radii.to_pixels(window.rem_size());
        window.paint_surface(bounds, new_bounds, corner_radii, self.source.clone());
    }
}

impl IntoElement for Surface {
    type Element = Self;

    fn into_element(self) -> Self::Element {
        self
    }
}

impl Styled for Surface {
    fn style(&mut self) -> &mut StyleRefinement {
        &mut self.style
    }
}
