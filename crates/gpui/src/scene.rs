// todo("windows"): remove
#![cfg_attr(windows, allow(dead_code))]

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::{
    AtlasTextureId, AtlasTile, Background, Bounds, ContentMask, Corners, DevicePixels, Edges, Hsla,
    Pixels, Point, Radians, ScaledPixels, ShaderId, Size, bounds_tree::BoundsTree, point,
};
use collections::FxHashMap;
use std::{
    fmt::Debug,
    iter::Peekable,
    ops::{Add, Range, Sub},
    slice,
};

#[allow(non_camel_case_types, unused)]
#[expect(missing_docs)]
pub type PathVertex_ScaledPixels = PathVertex<ScaledPixels>;

#[expect(missing_docs)]
pub type DrawOrder = u32;

/// Names a run of paint operations, so a later frame can find what an element drew and draw it
/// again.
pub type SpanId = u64;

/// The multiplier of the Fx hash, which folds what the bounds tree was told into
/// [`Scene::history`].
const HISTORY_SEED: u64 = 0x51_7c_c1_b7_27_22_0a_95;

/// What a scene tells its bounds tree, as folded into [`Scene::history`].
#[derive(Clone, Copy)]
#[repr(u64)]
enum TreeEvent {
    Insert = 1,
    PushLayer,
    PopLayer,
    Open,
    Close,
}

/// Where a named span sits among a scene's paint operations, with the bounds it was given.
struct PaintSpan {
    operations: Range<usize>,
    bounds: Bounds<Pixels>,
}

/// A boolean stored as a `u32` so that GPU-facing structs contain no
/// compiler-inserted padding bytes, which would be undefined behavior to
/// reinterpret as `&[u8]` when writing instance buffers. Guaranteed to be
/// `0` or `1` by construction; shaders read it as a `u32`/`uint`.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
#[repr(transparent)]
pub struct PaddedBool32(u32);

impl From<bool> for PaddedBool32 {
    fn from(value: bool) -> Self {
        PaddedBool32(value as u32)
    }
}

#[derive(Default)]
#[expect(missing_docs)]
pub struct Scene {
    pub(crate) paint_operations: Vec<PaintOperation>,
    primitive_bounds: BoundsTree<ScaledPixels>,
    /// A running hash of everything the bounds tree was told this frame, in order, along with
    /// every layer pop. Two scenes with the same history hand out the same orders from there on.
    history: u64,
    /// The history at chosen paint operation indices, in rising index order, so a later frame
    /// that replays from one of them can tell whether its orders still hold.
    marks: Vec<(usize, u64)>,
    layer_stack: Vec<DrawOrder>,
    spans: FxHashMap<SpanId, PaintSpan>,
    span_stack: Vec<(SpanId, usize, Bounds<Pixels>)>,
    pub shadows: Vec<Shadow>,
    pub backdrops: Vec<Backdrop>,
    pub effects: Vec<LayerEffect>,
    effect_stack: Vec<usize>,
    pub quads: Vec<Quad>,
    pub shaders: Vec<ShaderQuad>,
    pub paths: Vec<Path<ScaledPixels>>,
    pub underlines: Vec<Underline>,
    pub monochrome_sprites: Vec<MonochromeSprite>,
    pub subpixel_sprites: Vec<SubpixelSprite>,
    pub polychrome_sprites: Vec<PolychromeSprite>,
    pub surfaces: Vec<PaintSurface>,
    /// How far past the window's right and bottom edges the frame is drawn, in device pixels,
    /// so a backdrop blur near those edges reads what lies there instead of nothing. The
    /// renderer draws the frame this much larger and shows only the window's part of it.
    pub overscan: Size<DevicePixels>,
}

#[expect(missing_docs)]
impl Scene {
    pub fn clear(&mut self) {
        self.paint_operations.clear();
        self.primitive_bounds.clear();
        self.history = 0;
        self.marks.clear();
        self.layer_stack.clear();
        self.spans.clear();
        self.span_stack.clear();
        self.paths.clear();
        self.shadows.clear();
        self.backdrops.clear();
        self.effects.clear();
        self.effect_stack.clear();
        self.quads.clear();
        self.shaders.clear();
        self.underlines.clear();
        self.monochrome_sprites.clear();
        self.subpixel_sprites.clear();
        self.polychrome_sprites.clear();
        self.surfaces.clear();
        self.overscan = Size::default();
    }

    pub fn len(&self) -> usize {
        self.paint_operations.len()
    }

    /// Notes the history at the current paint operation index, so a later frame can replay
    /// from here with the orders this frame gives out.
    pub(crate) fn mark(&mut self) {
        let index = self.paint_operations.len();
        if self.marks.last().is_none_or(|(last, _)| *last < index) {
            self.marks.push((index, self.history));
        }
    }

    pub fn push_layer(&mut self, bounds: Bounds<ScaledPixels>) {
        self.record(TreeEvent::PushLayer, bounds);
        let order = self.primitive_bounds.insert(bounds);
        self.layer_stack.push(order);
        self.paint_operations
            .push(PaintOperation::StartLayer(bounds, order));
    }

    pub fn pop_layer(&mut self) {
        self.record(TreeEvent::PopLayer, Bounds::default());
        self.layer_stack.pop();
        self.paint_operations.push(PaintOperation::EndLayer);
    }

    /// Starts a layer that is drawn offscreen, put through the given filter, and composited back.
    ///
    /// The layer claims a range of orders nothing outside it can share, above everything painted
    /// before it, while its contents order among themselves the way they would anywhere else, so
    /// a quad still covers an image painted before it. Only what is painted later over the area
    /// the layer lands on goes above the range. The rest keeps the order it would have had
    /// without the layer, so a backdrop beside a pressed button blurs what it always did.
    pub fn push_filter(
        &mut self,
        source_bounds: Bounds<ScaledPixels>,
        transform_origin: Point<ScaledPixels>,
        content_mask: Bounds<ScaledPixels>,
        filter: Filter,
    ) {
        self.record(TreeEvent::Open, Bounds::default());
        let start = self.primitive_bounds.open();
        let mut destination_bounds =
            scale_bounds_around(source_bounds, transform_origin, filter.scale);
        destination_bounds.origin += filter.translate;
        self.effects.push(LayerEffect {
            start,
            end: DrawOrder::MAX,
            source_bounds,
            transform_origin,
            destination_bounds,
            content_mask,
            filter,
            parent: self.effect_stack.last().copied(),
        });
        self.effect_stack.push(self.effects.len() - 1);
        self.paint_operations.push(PaintOperation::StartFilter(
            source_bounds,
            transform_origin,
            content_mask,
            filter,
        ));
    }

    pub fn pop_filter(&mut self) {
        if let Some(index) = self.effect_stack.pop() {
            let footprint = self.effects[index].destination_clip();
            self.record(TreeEvent::Close, footprint);
            self.effects[index].end = self.primitive_bounds.close(footprint);
        }
        self.paint_operations.push(PaintOperation::EndFilter);
    }

    /// Starts a named span. Everything painted until [`Scene::pop_span`] can be found again under
    /// `id`, in this scene and in the next one that replays it.
    pub fn push_span(&mut self, id: SpanId, bounds: Bounds<Pixels>) {
        self.paint_operations
            .push(PaintOperation::StartSpan(id, bounds));
        self.span_stack
            .push((id, self.paint_operations.len(), bounds));
    }

    pub fn pop_span(&mut self) {
        if let Some((id, start, bounds)) = self.span_stack.pop() {
            let operations = start..self.paint_operations.len();
            self.spans.insert(id, PaintSpan { operations, bounds });
        }
        self.paint_operations.push(PaintOperation::EndSpan);
    }

    /// The bounds the span `id` was given, when this scene holds one.
    pub fn span(&self, id: SpanId) -> Option<Bounds<Pixels>> {
        self.spans.get(&id).map(|span| span.bounds)
    }

    /// Paints what `prev_scene` painted in its span `id` again, under the same name, with the
    /// alpha of every colour scaled by `fade`. Spans nested in it lose their names. Returns
    /// whether `prev_scene` held the span.
    pub fn replay_span(&mut self, id: SpanId, prev_scene: &Scene, fade: f32) -> bool {
        let Some(span) = prev_scene.spans.get(&id) else {
            return false;
        };
        self.push_span(id, span.bounds);
        // A span pushed by hand can open a layer or a filter it does not close, or close one
        // it did not open. The replay keeps count, so it never closes the caller's and never
        // leaves one of its own open.
        let (mut layers, mut filters) = (0usize, 0usize);
        for operation in &prev_scene.paint_operations[span.operations.clone()] {
            match operation {
                PaintOperation::Primitive(primitive) => {
                    self.insert_primitive(primitive.clone().faded(fade))
                }
                PaintOperation::StartLayer(bounds, _) => {
                    layers += 1;
                    self.push_layer(*bounds);
                }
                PaintOperation::EndLayer if layers > 0 => {
                    layers -= 1;
                    self.pop_layer();
                }
                PaintOperation::StartFilter(bounds, transform_origin, content_mask, filter) => {
                    filters += 1;
                    self.push_filter(*bounds, *transform_origin, *content_mask, *filter);
                }
                PaintOperation::EndFilter if filters > 0 => {
                    filters -= 1;
                    self.pop_filter();
                }
                PaintOperation::EndLayer
                | PaintOperation::EndFilter
                | PaintOperation::StartSpan(..)
                | PaintOperation::EndSpan => {}
            }
        }
        for _ in 0..filters {
            self.pop_filter();
        }
        for _ in 0..layers {
            self.pop_layer();
        }
        self.pop_span();
        true
    }

    /// The draw order of a batch's first primitive.
    pub fn batch_order(&self, batch: &PrimitiveBatch) -> DrawOrder {
        match batch {
            PrimitiveBatch::Shadows(range) => self.shadows[range.start].order,
            PrimitiveBatch::Backdrops(range) => self.backdrops[range.start].order,
            PrimitiveBatch::Quads(range) => self.quads[range.start].order,
            PrimitiveBatch::Shaders { range, .. } => self.shaders[range.start].order,
            PrimitiveBatch::Paths(range) => self.paths[range.start].order,
            PrimitiveBatch::Underlines(range) => self.underlines[range.start].order,
            PrimitiveBatch::MonochromeSprites { range, .. } => {
                self.monochrome_sprites[range.start].order
            }
            PrimitiveBatch::SubpixelSprites { range, .. } => {
                self.subpixel_sprites[range.start].order
            }
            PrimitiveBatch::PolychromeSprites { range, .. } => {
                self.polychrome_sprites[range.start].order
            }
            PrimitiveBatch::Surfaces(range) => self.surfaces[range.start].order,
        }
    }

    /// The innermost layer a primitive drawn at the given order belongs to, if any.
    ///
    /// Membership is a range rather than an exact order: elements open stacking layers of their
    /// own inside a filtered subtree, and their primitives carry those orders instead.
    pub fn filtered(&self, order: DrawOrder) -> Option<usize> {
        let after = self.effects.partition_point(|layer| layer.start <= order);
        (0..after)
            .rev()
            .find(|index| self.effects[*index].end > order)
    }

    /// A layer and everything that encloses it, outermost first.
    pub fn filter_chain(&self, index: usize) -> Vec<usize> {
        let mut chain = vec![index];
        let mut walk = self.effects[index].parent;
        while let Some(parent) = walk {
            chain.push(parent);
            walk = self.effects[parent].parent;
        }
        chain.reverse();
        chain
    }

    pub fn insert_primitive(&mut self, primitive: impl Into<Primitive>) {
        let primitive = primitive.into();
        let clipped_bounds = primitive
            .bounds()
            .intersect(&primitive.content_mask().bounds);

        if clipped_bounds.is_empty() {
            return;
        }

        let order = match self.layer_stack.last() {
            Some(order) => *order,
            None => self.order(clipped_bounds),
        };
        self.push_primitive(primitive, order);
    }

    /// Inserts a primitive replayed from a scene whose history matched this one's where the
    /// replay began, with the order it was drawn at there.
    fn keep_primitive(&mut self, primitive: Primitive) {
        let order = primitive.order();
        if self.layer_stack.is_empty() {
            let clipped_bounds = primitive
                .bounds()
                .intersect(&primitive.content_mask().bounds);
            self.record(TreeEvent::Insert, clipped_bounds);
            self.primitive_bounds.insert_known(clipped_bounds, order);
        }
        self.push_primitive(primitive, order);
    }

    fn push_primitive(&mut self, mut primitive: Primitive, order: DrawOrder) {
        match &mut primitive {
            Primitive::Shadow(shadow) => {
                shadow.order = order;
                self.shadows.push(*shadow);
            }
            Primitive::Backdrop(backdrop) => {
                backdrop.order = order;
                self.backdrops.push(*backdrop);
            }
            Primitive::Quad(quad) => {
                quad.order = order;
                self.quads.push(*quad);
            }
            Primitive::Shader(shader) => {
                shader.order = order;
                self.shaders.push(*shader);
            }
            Primitive::Path(path) => {
                path.order = order;
                path.id = PathId(self.paths.len());
                self.paths.push(path.clone());
            }
            Primitive::Underline(underline) => {
                underline.order = order;
                self.underlines.push(*underline);
            }
            Primitive::MonochromeSprite(sprite) => {
                sprite.order = order;
                self.monochrome_sprites.push(*sprite);
            }
            Primitive::SubpixelSprite(sprite) => {
                sprite.order = order;
                self.subpixel_sprites.push(*sprite);
            }
            Primitive::PolychromeSprite(sprite) => {
                sprite.order = order;
                self.polychrome_sprites.push(*sprite);
            }
            Primitive::Surface(surface) => {
                surface.order = order;
                self.surfaces.push(surface.clone());
            }
        }
        self.paint_operations
            .push(PaintOperation::Primitive(primitive));
    }

    /// Paints the operations `range` of `prev_scene` again. When this scene's history matches
    /// the one `prev_scene` marked at `range.start`, every primitive keeps the order it had
    /// there, which is the order a fresh insert would give it, and the bounds tree skips its
    /// searches. The history covers open filters and layers too, so a match holds inside them.
    pub fn replay(&mut self, range: Range<usize>, prev_scene: &Scene) {
        let first_mark = prev_scene
            .marks
            .partition_point(|(index, _)| *index < range.start);
        let keep = prev_scene.marks.get(first_mark) == Some(&(range.start, self.history));
        let mut marks = prev_scene.marks[first_mark..]
            .iter()
            .map(|(index, _)| *index)
            .peekable();
        for (index, operation) in (range.start..).zip(&prev_scene.paint_operations[range]) {
            // A cached view nested in this range may be replayed alone next frame, so the
            // marks inside it move along with what they mark.
            if marks.next_if_eq(&index).is_some() {
                self.mark();
            }
            match operation {
                PaintOperation::Primitive(primitive) => match keep {
                    true => self.keep_primitive(primitive.clone()),
                    false => self.insert_primitive(primitive.clone()),
                },
                PaintOperation::StartLayer(bounds, order) => match keep {
                    true => {
                        self.record(TreeEvent::PushLayer, *bounds);
                        self.primitive_bounds.insert_known(*bounds, *order);
                        self.layer_stack.push(*order);
                        self.paint_operations
                            .push(PaintOperation::StartLayer(*bounds, *order));
                    }
                    false => self.push_layer(*bounds),
                },
                PaintOperation::EndLayer => self.pop_layer(),
                PaintOperation::StartFilter(bounds, transform_origin, content_mask, filter) => {
                    self.push_filter(*bounds, *transform_origin, *content_mask, *filter)
                }
                PaintOperation::EndFilter => self.pop_filter(),
                PaintOperation::StartSpan(id, bounds) => self.push_span(*id, *bounds),
                PaintOperation::EndSpan => self.pop_span(),
            }
        }
    }

    /// Asks the bounds tree for the order of `bounds` and folds the request into the history.
    fn order(&mut self, bounds: Bounds<ScaledPixels>) -> DrawOrder {
        self.record(TreeEvent::Insert, bounds);
        self.primitive_bounds.insert(bounds)
    }

    /// Folds one event into the history.
    fn record(&mut self, event: TreeEvent, bounds: Bounds<ScaledPixels>) {
        let pair = |a: ScaledPixels, b: ScaledPixels| {
            (u64::from(a.0.to_bits()) << 32) | u64::from(b.0.to_bits())
        };
        for word in [
            event as u64,
            pair(bounds.origin.x, bounds.origin.y),
            pair(bounds.size.width, bounds.size.height),
        ] {
            self.history = (self.history.rotate_left(5) ^ word).wrapping_mul(HISTORY_SEED);
        }
    }

    pub fn finish(&mut self) {
        self.span_stack.clear();
        // A layer that never closed would otherwise swallow everything painted after it.
        for index in self.effect_stack.drain(..) {
            log::warn!("scene: a filtered layer was left open");
            self.effects[index].end = self.effects[index].start;
        }
        self.shadows.sort_by_key(|shadow| shadow.order);
        self.backdrops.sort_by_key(|backdrop| backdrop.order);
        self.quads.sort_by_key(|quad| quad.order);
        self.shaders
            .sort_by_key(|shader| (shader.order, shader.shader));
        self.paths.sort_by_key(|path| path.order);
        self.underlines.sort_by_key(|underline| underline.order);
        self.monochrome_sprites
            .sort_by_key(|sprite| (sprite.order, sprite.tile.texture_id.index));
        self.subpixel_sprites
            .sort_by_key(|sprite| (sprite.order, sprite.tile.texture_id.index));
        self.polychrome_sprites
            .sort_by_key(|sprite| (sprite.order, sprite.tile.texture_id.index));
        self.surfaces.sort_by_key(|surface| surface.order);
    }

    #[cfg_attr(
        all(
            any(target_os = "linux", target_os = "freebsd"),
            not(any(feature = "x11", feature = "wayland"))
        ),
        allow(dead_code)
    )]
    pub fn batches(&self) -> impl Iterator<Item = PrimitiveBatch> + '_ {
        BatchIterator {
            layered: {
                // A layer whose contents were clipped away holds nothing to redirect, so it does
                // not get to split a batch.
                let mut edges: Vec<DrawOrder> = self
                    .effects
                    .iter()
                    .filter(|layer| layer.end > layer.start + 1)
                    .flat_map(|layer| [layer.start, layer.end])
                    .collect();
                edges.sort_unstable();
                edges
            },
            shadows_start: 0,
            shadows_iter: self.shadows.iter().peekable(),
            backdrops_start: 0,
            backdrops_iter: self.backdrops.iter().peekable(),
            quads_start: 0,
            quads_iter: self.quads.iter().peekable(),
            shaders_start: 0,
            shaders_iter: self.shaders.iter().peekable(),
            paths_start: 0,
            paths_iter: self.paths.iter().peekable(),
            underlines_start: 0,
            underlines_iter: self.underlines.iter().peekable(),
            monochrome_sprites_start: 0,
            monochrome_sprites_iter: self.monochrome_sprites.iter().peekable(),
            subpixel_sprites_start: 0,
            subpixel_sprites_iter: self.subpixel_sprites.iter().peekable(),
            polychrome_sprites_start: 0,
            polychrome_sprites_iter: self.polychrome_sprites.iter().peekable(),
            surfaces_start: 0,
            surfaces_iter: self.surfaces.iter().peekable(),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Default)]
#[cfg_attr(
    all(
        any(target_os = "linux", target_os = "freebsd"),
        not(any(feature = "x11", feature = "wayland"))
    ),
    allow(dead_code)
)]
pub(crate) enum PrimitiveKind {
    Shadow,
    Backdrop,
    #[default]
    Quad,
    Shader,
    Path,
    Underline,
    MonochromeSprite,
    SubpixelSprite,
    PolychromeSprite,
    Surface,
}

pub(crate) enum PaintOperation {
    Primitive(Primitive),
    /// A layer's bounds and the order everything in it is drawn at.
    StartLayer(Bounds<ScaledPixels>, DrawOrder),
    EndLayer,
    StartFilter(
        Bounds<ScaledPixels>,
        Point<ScaledPixels>,
        Bounds<ScaledPixels>,
        Filter,
    ),
    EndFilter,
    StartSpan(SpanId, Bounds<Pixels>),
    EndSpan,
}

/// A run of primitives that is drawn offscreen, put through [`Effect`], and composited back.
#[derive(Clone, Copy, Debug, PartialEq)]
#[expect(missing_docs)]
pub struct LayerEffect {
    /// The order the layer opened at. Everything it contains is drawn between this and `end`.
    pub start: DrawOrder,
    /// The order the layer closed at.
    pub end: DrawOrder,
    /// The area of the untransformed offscreen source, including blur reach.
    pub source_bounds: Bounds<ScaledPixels>,
    /// The center of the element's original bounds, before adding blur reach.
    pub transform_origin: Point<ScaledPixels>,
    /// The source bounds after applying the layer scale and translation.
    pub destination_bounds: Bounds<ScaledPixels>,
    /// The parent content mask that was in force when the layer opened.
    pub content_mask: Bounds<ScaledPixels>,
    pub filter: Filter,
    /// The enclosing layer, if this one is nested.
    pub parent: Option<usize>,
}

impl LayerEffect {
    /// The transformed destination area after clipping to the parent content mask.
    pub fn destination_clip(&self) -> Bounds<ScaledPixels> {
        self.destination_bounds.intersect(&self.content_mask)
    }

    /// The destination span rounded out to whole pixels when a transform made it fractional.
    pub fn destination_scissor_bounds(
        &self,
        destination_span: Bounds<ScaledPixels>,
    ) -> Bounds<ScaledPixels> {
        if !self.filter.transforms() {
            return destination_span;
        }

        Bounds::from_corners(
            point(
                ScaledPixels(destination_span.origin.x.0.floor()),
                ScaledPixels(destination_span.origin.y.0.floor()),
            ),
            point(
                ScaledPixels(destination_span.bottom_right().x.0.ceil()),
                ScaledPixels(destination_span.bottom_right().y.0.ceil()),
            ),
        )
    }

    /// The source area the blur passes need to update.
    ///
    /// Unscaled layers retain the previous clipped processing area. A transformed layer must keep
    /// its whole source because the destination clip is in a different coordinate space.
    pub fn blur_bounds(&self, destination_span: Bounds<ScaledPixels>) -> Bounds<ScaledPixels> {
        if self.filter.transforms() {
            self.source_bounds
        } else {
            destination_span
        }
    }

    /// Whether adjacent sibling layers may share one composite operation.
    pub fn can_merge_with(&self, next: &Self) -> bool {
        self.filter == next.filter
            && self.parent == next.parent
            && !next.filter.fades()
            && !self.filter.transforms()
    }
}

/// What a layer does to its contents on the way back into the frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Filter {
    /// The standard deviation of the blur, in scaled pixels.
    pub blur: f32,
    /// How far the contents fade out at the top edge, in scaled pixels.
    pub fade_top: f32,
    /// How far the contents fade out at the bottom edge, in scaled pixels.
    pub fade_bottom: f32,
    /// How far the contents fade out at the left edge, in scaled pixels.
    pub fade_left: f32,
    /// How far the contents fade out at the right edge, in scaled pixels.
    pub fade_right: f32,
    /// Uniform compositor scale around the layer's transform origin.
    pub scale: f32,
    /// Compositor translation in scaled pixels.
    pub translate: Point<ScaledPixels>,
}

/// The widest blur, as a standard deviation in scaled pixels, that runs as a gaussian at full
/// resolution. Anything wider goes through [`Pyramid`], whose first level halves the image.
pub const GAUSSIAN_SIGMA: f32 = 4.;

/// How far a gaussian pass reads to either side, in texels, before its weights stop mattering.
pub const GAUSSIAN_REACH: f32 = 3.;

/// How many times a pyramid blur can halve the image. The gaussian at the deepest level spreads
/// up to four of its texels, so six reach a standard deviation of about 250 scaled pixels.
pub const BLUR_LEVELS: usize = 6;

/// The spread a gaussian at the deepest level of a pyramid starts from, in its texels. It takes
/// the fewest levels that leave the gaussian at least this wide, since a narrower one leaves the
/// level's texel grid showing through the passes back up.
const PYRAMID_SPREAD: f32 = 2.;

/// The variance a pass down adds, in squared texels of the level it reads: four bilinear taps,
/// one on each corner around the texel it writes, average the four by four texels under it.
const DOWN_VARIANCE: f32 = 1.25;

/// The variance a pass up adds, in squared texels of the level it reads: four bilinear taps half
/// a texel out on each diagonal, a box of one texel on top of the bilinear tent.
const UP_VARIANCE: f32 = 1. / 6. + 1. / 4.;

/// How a renderer blurs by a given standard deviation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BlurPasses {
    /// Two separable gaussian passes at full resolution, across and then down, with this
    /// standard deviation in scaled pixels.
    Gaussian(f32),
    /// A gaussian at a level of a pyramid.
    Pyramid(Pyramid),
}

impl BlurPasses {
    /// The source margin needed around a clipped blur.
    pub fn source_margin(self) -> ScaledPixels {
        ScaledPixels(match self {
            Self::Gaussian(sigma) => (sigma * GAUSSIAN_REACH).ceil() + 1.,
            Self::Pyramid(pyramid) => pyramid.source_margin(),
        })
    }

    /// The blur that spreads a point as far as a gaussian of standard deviation `sigma`, in
    /// scaled pixels.
    pub fn for_sigma(sigma: f32) -> Self {
        match sigma <= GAUSSIAN_SIGMA {
            true => Self::Gaussian(sigma.max(0.)),
            false => Self::Pyramid(Pyramid::for_sigma(sigma)),
        }
    }
}

/// A wide blur. It halves the image `levels` times, blurs the deepest level with a separable
/// gaussian, and grows it back level by level to full size, so its cost barely grows with the
/// radius. Every pass is separable, so the blur stays round, and the gaussian carries most of
/// the spread, so it keeps the gaussian's soft falloff. A dual Kawase blur spread a point into
/// a square or a diamond depending on its radius, and stopping a level short of full size showed
/// the deepest level's texels as a grid of soft squares.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pyramid {
    /// How many times the image is halved, at least one.
    pub levels: usize,
    /// The standard deviation of the gaussian at the deepest level, in its texels. The passes
    /// down and up add the rest of the spread.
    pub sigma: f32,
}

impl Pyramid {
    /// The passes that spread a point as far as a gaussian of standard deviation `sigma`, in
    /// scaled pixels. A radius past the deepest level is capped there.
    pub fn for_sigma(sigma: f32) -> Self {
        let sigma = sigma.max(GAUSSIAN_SIGMA);
        let levels = ((sigma / PYRAMID_SPREAD).log2().floor() as usize).clamp(1, BLUR_LEVELS);
        // Every pass adds its variance, scaled by the size of the texels it reads. What the
        // gaussian leaves to them it does not have to spread itself.
        let added: f32 = (1..=levels)
            .map(|level| {
                DOWN_VARIANCE * (1u32 << (2 * (level - 1))) as f32
                    + UP_VARIANCE * (1u32 << (2 * level)) as f32
            })
            .sum();
        let deepest = (1u32 << levels) as f32;
        let rest = (sigma * sigma - added).max(0.);
        Self {
            levels,
            sigma: (rest.sqrt() / deepest).min(PYRAMID_SPREAD * 2.),
        }
    }

    /// How far the gaussian reads past a texel, in texels of the deepest level.
    fn spread(&self) -> f32 {
        (self.sigma * GAUSSIAN_REACH).ceil() + 1.
    }

    /// How far around the clip, in scaled pixels of the frame, the pass up into `level` has to
    /// write, so the passes up after it read only texels it wrote. Level zero gets a pixel more
    /// than the clip for the sample that composites it.
    pub fn up_margin(&self, level: usize) -> f32 {
        // the pass into level `k - 1` reads two texels of level `k` out
        1. + 2. * ((2u32 << level) - 2) as f32
    }

    /// How far around the clip the pass down the deepest level, the second half of the gaussian,
    /// has to write.
    pub fn along_margin(&self) -> f32 {
        let deepest = (1u32 << self.levels) as f32;
        self.up_margin(self.levels - 1) + 2. * deepest
    }

    /// How far around the clip the pass across the deepest level, the first half of the
    /// gaussian, has to write.
    pub fn across_margin(&self) -> f32 {
        self.along_margin() + self.spread() * (1u32 << self.levels) as f32
    }

    /// How far around the clip the pass down into `level` has to write. The deepest level feeds
    /// the gaussian, and every shallower one the pass down below it. The passes up write into
    /// the same levels afterwards, but only once the passes down are done with them.
    pub fn down_margin(&self, level: usize) -> f32 {
        // the gaussian reads its spread out of the deepest level, and the pass into level
        // `k + 1` two texels of level `k`
        let deepest = (1u32 << self.levels) as f32;
        let below: f32 = (level + 1..=self.levels)
            .map(|deeper| (1u32 << deeper) as f32)
            .sum();
        self.across_margin() + self.spread() * deepest + below
    }

    /// How far around the clip the source has to hold what lies under it, which is what the
    /// first pass down reads.
    pub fn source_margin(&self) -> f32 {
        self.down_margin(0)
    }
}

impl Default for Filter {
    fn default() -> Self {
        Self {
            blur: 0.0,
            fade_top: 0.0,
            fade_bottom: 0.0,
            fade_left: 0.0,
            fade_right: 0.0,
            scale: 1.0,
            translate: Point::default(),
        }
    }
}

impl Filter {
    /// Whether the layer changes its contents at all.
    pub fn is_noop(&self) -> bool {
        self.blur <= 0. && !self.fades() && !self.transforms()
    }

    /// Whether the layer has to go through the blur passes.
    pub fn blurs(&self) -> bool {
        self.blur > 0.
    }

    /// Whether the layer is masked on its way back.
    pub fn fades(&self) -> bool {
        self.fade_top > 0. || self.fade_bottom > 0. || self.fade_left > 0. || self.fade_right > 0.
    }

    /// Whether the layer is transformed while it is composited.
    pub fn scales(&self) -> bool {
        self.scale != 1.0
    }

    /// Whether the layer is translated while it is composited.
    pub fn translates(&self) -> bool {
        self.translate != Point::default()
    }

    /// Whether the layer is transformed while it is composited.
    pub fn transforms(&self) -> bool {
        self.scales() || self.translates()
    }
}

pub(crate) fn scale_bounds_around(
    bounds: Bounds<ScaledPixels>,
    origin: Point<ScaledPixels>,
    scale: f32,
) -> Bounds<ScaledPixels> {
    let scaled_x = |value: ScaledPixels| ScaledPixels(origin.x.0 + (value.0 - origin.x.0) * scale);
    let scaled_y = |value: ScaledPixels| ScaledPixels(origin.y.0 + (value.0 - origin.y.0) * scale);

    Bounds::from_corners(
        point(scaled_x(bounds.origin.x), scaled_y(bounds.origin.y)),
        point(
            scaled_x(bounds.bottom_right().x),
            scaled_y(bounds.bottom_right().y),
        ),
    )
}

#[cfg(test)]
mod layer_filter_tests {
    use super::*;
    use crate::{bounds, size};

    fn rect(x: f32, y: f32, width: f32, height: f32) -> Bounds<ScaledPixels> {
        bounds(
            point(ScaledPixels(x), ScaledPixels(y)),
            size(ScaledPixels(width), ScaledPixels(height)),
        )
    }

    fn origin(x: f32, y: f32) -> Point<ScaledPixels> {
        point(ScaledPixels(x), ScaledPixels(y))
    }

    fn filter(scale: f32) -> Filter {
        Filter {
            scale,
            ..Default::default()
        }
    }

    fn assert_close(actual: f32, expected: f32) {
        assert!(
            (actual - expected).abs() < 0.0001,
            "expected {expected}, got {actual}"
        );
    }

    fn assert_bounds_close(actual: Bounds<ScaledPixels>, expected: Bounds<ScaledPixels>) {
        assert_close(actual.origin.x.0, expected.origin.x.0);
        assert_close(actual.origin.y.0, expected.origin.y.0);
        assert_close(actual.size.width.0, expected.size.width.0);
        assert_close(actual.size.height.0, expected.size.height.0);
    }

    #[test]
    fn filter_scale_participates_in_noop_detection() {
        assert!(Filter::default().is_noop());
        assert!(filter(1.0).is_noop());
        assert!(!filter(0.99).is_noop());

        let translated = Filter {
            translate: origin(0.25, -0.75),
            ..Default::default()
        };
        assert!(!translated.is_noop());
        assert!(translated.transforms());
    }

    #[test]
    fn fractional_translation_moves_only_the_destination() {
        let source = rect(10.0, 20.0, 100.0, 40.0);
        let mut scene = Scene::default();
        scene.push_filter(
            source,
            origin(60.0, 40.0),
            rect(0.0, 0.0, 200.0, 100.0),
            Filter {
                translate: origin(0.25, -0.75),
                ..Default::default()
            },
        );
        scene.pop_filter();

        assert_eq!(scene.effects[0].source_bounds, source);
        assert_bounds_close(
            scene.effects[0].destination_bounds,
            rect(10.25, 19.25, 100.0, 40.0),
        );
    }

    #[test]
    fn center_based_scaling_keeps_the_origin_fixed() {
        let source = rect(10.0, 20.0, 100.0, 40.0);
        let transform_origin = origin(60.0, 40.0);
        let destination = scale_bounds_around(source, transform_origin, 0.99);

        assert_bounds_close(destination, rect(10.5, 20.2, 99.0, 39.6));
        assert_bounds_close(
            Bounds::centered_at(destination.center(), destination.size),
            destination,
        );
        assert_eq!(destination.center(), transform_origin);
    }

    #[test]
    fn fractional_scale_preserves_subpixel_destination_bounds() {
        let destination =
            scale_bounds_around(rect(4.25, 8.5, 137.0, 53.0), origin(72.75, 35.0), 0.9973);

        assert_close(destination.origin.x.0, 4.43495);
        assert_close(destination.origin.y.0, 8.57155);
        assert_close(destination.size.width.0, 136.6301);
        assert_close(destination.size.height.0, 52.8569);
    }

    #[test]
    fn layer_effect_separates_source_destination_and_parent_clip() {
        let source = rect(10.0, 20.0, 100.0, 40.0);
        let content_mask = rect(20.0, 10.0, 80.0, 80.0);
        let mut scene = Scene::default();
        scene.push_filter(source, origin(60.0, 40.0), content_mask, filter(1.1));
        scene.pop_filter();

        let effect = scene.effects[0];
        assert_eq!(effect.source_bounds, source);
        assert_eq!(effect.transform_origin, origin(60.0, 40.0));
        assert_bounds_close(effect.destination_bounds, rect(5.0, 18.0, 110.0, 44.0));
        assert_bounds_close(effect.destination_clip(), rect(20.0, 18.0, 80.0, 44.0));
        assert_eq!(effect.blur_bounds(effect.destination_clip()), source);

        let fractional_span = rect(20.25, 18.75, 79.5, 43.5);
        assert_eq!(
            effect.destination_scissor_bounds(fractional_span),
            rect(20.0, 18.0, 80.0, 45.0)
        );
    }

    #[test]
    fn nested_effects_keep_their_parent_and_geometry() {
        let mut scene = Scene::default();
        scene.push_filter(
            rect(0.0, 0.0, 200.0, 100.0),
            origin(100.0, 50.0),
            rect(0.0, 0.0, 300.0, 200.0),
            filter(0.99),
        );
        scene.push_filter(
            rect(20.0, 10.0, 50.0, 30.0),
            origin(45.0, 25.0),
            rect(0.0, 0.0, 200.0, 100.0),
            filter(0.9973),
        );
        scene.pop_filter();
        scene.pop_filter();

        assert_eq!(scene.effects[0].parent, None);
        assert_eq!(scene.effects[1].parent, Some(0));
        assert_eq!(scene.filter_chain(1), vec![0, 1]);
        assert_eq!(scene.effects[1].transform_origin, origin(45.0, 25.0));
    }

    #[test]
    fn scene_replay_restores_scaled_layer_effects() {
        let mut previous = Scene::default();
        previous.push_filter(
            rect(10.0, 20.0, 100.0, 40.0),
            origin(60.0, 40.0),
            rect(0.0, 0.0, 120.0, 80.0),
            Filter {
                blur: 3.0,
                fade_top: 2.0,
                scale: 0.9973,
                ..Default::default()
            },
        );
        previous.pop_filter();

        let mut replayed = Scene::default();
        replayed.replay(0..previous.len(), &previous);

        assert_eq!(replayed.effects, previous.effects);
    }

    #[test]
    fn transformed_sibling_layers_cannot_merge() {
        let bounds = rect(0.0, 0.0, 100.0, 50.0);
        let content_mask = rect(0.0, 0.0, 500.0, 500.0);
        let mut scene = Scene::default();
        scene.push_filter(bounds, origin(50.0, 25.0), content_mask, filter(0.99));
        scene.pop_filter();
        scene.push_filter(bounds, origin(150.0, 25.0), content_mask, filter(0.99));
        scene.pop_filter();

        assert!(!scene.effects[0].can_merge_with(&scene.effects[1]));

        let mut unscaled = scene.effects;
        unscaled[0].filter.scale = 1.0;
        unscaled[1].filter.scale = 1.0;
        assert!(unscaled[0].can_merge_with(&unscaled[1]));
    }

    #[test]
    fn a_closed_layer_lifts_only_what_is_painted_over_it() {
        let content_mask = rect(0.0, 0.0, 500.0, 500.0);
        let button = rect(100.0, 60.0, 20.0, 20.0);
        let mut scene = Scene::default();
        let row = scene.primitive_bounds.insert(rect(0.0, 50.0, 500.0, 40.0));
        scene.push_filter(button, origin(110.0, 70.0), content_mask, filter(0.9));
        scene.primitive_bounds.insert(button);
        scene.pop_filter();
        let layer = scene.effects[0];

        // A table head painted after the pressed row, above it, keeps the order it has with the
        // button at rest, so its backdrop still blurs before the row is drawn.
        let head = scene.primitive_bounds.insert(rect(0.0, 10.0, 500.0, 30.0));
        assert!(head <= row, "head at {head}, row at {row}");
        assert!(scene.filtered(head).is_none());

        // What stacks on the head climbs past the layer's orders rather than into them.
        let label = scene.primitive_bounds.insert(rect(10.0, 15.0, 40.0, 20.0));
        assert!(
            label > layer.end,
            "label at {label}, layer ends at {}",
            layer.end
        );

        // What is painted over the button still lands above its layer.
        let over = scene.primitive_bounds.insert(rect(105.0, 65.0, 5.0, 5.0));
        assert!(
            over > layer.end,
            "over at {over}, layer ends at {}",
            layer.end
        );
    }

    fn quad(bounds: Bounds<ScaledPixels>) -> Quad {
        Quad {
            order: 0,
            border_style: BorderStyle::default(),
            bounds,
            content_mask: ContentMask {
                bounds: rect(0.0, 0.0, 500.0, 500.0),
            },
            background: Background::default(),
            border_color: Hsla::default(),
            corner_radii: Corners::default(),
            border_widths: Edges::default(),
            corner_smoothing: 0.0,
            pad: 0,
        }
    }

    /// Paints three views the way a window does, marking the scene around each one, and
    /// returns their paint operation ranges. A view is painted live when `prev` is `None`,
    /// and otherwise replayed from the matching range of `prev`. The first view's middle
    /// quad is drawn at `moved`.
    fn paint_views(
        scene: &mut Scene,
        prev: Option<(&Scene, &[Range<usize>])>,
        moved: Bounds<ScaledPixels>,
        replay_first: bool,
    ) -> Vec<Range<usize>> {
        let content_mask = rect(0.0, 0.0, 500.0, 500.0);
        let views: [&dyn Fn(&mut Scene); 3] = [
            &|scene| {
                scene.insert_primitive(quad(rect(0.0, 0.0, 200.0, 100.0)));
                scene.insert_primitive(quad(moved));
                scene.insert_primitive(quad(rect(150.0, 50.0, 100.0, 100.0)));
            },
            &|scene| {
                scene.push_layer(rect(100.0, 80.0, 120.0, 60.0));
                scene.insert_primitive(quad(rect(100.0, 80.0, 120.0, 60.0)));
                scene.insert_primitive(quad(rect(110.0, 90.0, 20.0, 20.0)));
                scene.pop_layer();
                scene.insert_primitive(quad(rect(300.0, 0.0, 50.0, 50.0)));
            },
            &|scene| {
                scene.push_filter(
                    rect(120.0, 100.0, 80.0, 80.0),
                    origin(160.0, 140.0),
                    content_mask,
                    Filter {
                        blur: 4.0,
                        ..Default::default()
                    },
                );
                scene.insert_primitive(quad(rect(120.0, 100.0, 80.0, 80.0)));
                scene.insert_primitive(quad(rect(130.0, 110.0, 10.0, 10.0)));
                scene.pop_filter();
                scene.insert_primitive(quad(rect(0.0, 120.0, 400.0, 20.0)));
            },
        ];
        let mut ranges = Vec::new();
        for (index, view) in views.iter().enumerate() {
            scene.mark();
            let start = scene.len();
            match prev {
                Some((prev, ranges)) if index > 0 || replay_first => {
                    scene.replay(ranges[index].clone(), prev)
                }
                _ => view(scene),
            }
            ranges.push(start..scene.len());
        }
        scene.mark();
        ranges
    }

    /// The order of every primitive and layer, in paint order.
    fn orders(scene: &Scene) -> Vec<DrawOrder> {
        scene
            .paint_operations
            .iter()
            .filter_map(|operation| match operation {
                PaintOperation::Primitive(primitive) => Some(primitive.order()),
                PaintOperation::StartLayer(_, order) => Some(*order),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn a_replay_after_the_same_history_keeps_its_orders() {
        let at_rest = rect(10.0, 10.0, 30.0, 30.0);
        let mut painted = Scene::default();
        let ranges = paint_views(&mut painted, None, at_rest, false);

        let mut replayed = Scene::default();
        paint_views(&mut replayed, Some((&painted, &ranges)), at_rest, true);

        assert_eq!(orders(&replayed), orders(&painted));
        assert_eq!(replayed.effects, painted.effects);
        assert_eq!(replayed.history, painted.history);
        assert_eq!(replayed.marks, painted.marks);
        assert_eq!(replayed.primitive_bounds.searches, 0);
        assert_eq!(replayed.primitive_bounds.known_inserts, 8);
    }

    #[test]
    fn a_replay_after_a_changed_history_orders_afresh() {
        let at_rest = rect(10.0, 10.0, 30.0, 30.0);
        let moved = rect(105.0, 85.0, 100.0, 100.0);
        let mut painted = Scene::default();
        let ranges = paint_views(&mut painted, None, at_rest, false);

        let mut fresh = Scene::default();
        paint_views(&mut fresh, None, moved, false);
        assert_ne!(orders(&fresh), orders(&painted));

        let mut replayed = Scene::default();
        paint_views(&mut replayed, Some((&painted, &ranges)), moved, false);

        assert_eq!(orders(&replayed), orders(&fresh));
        assert_eq!(replayed.effects, fresh.effects);
        assert_eq!(replayed.marks, fresh.marks);
        assert_eq!(replayed.primitive_bounds.known_inserts, 0);
    }

    #[test]
    fn a_replay_after_a_quad_where_a_layer_was_orders_afresh() {
        let bounds = rect(40.0, 40.0, 100.0, 100.0);
        let paint = |scene: &mut Scene, prev: Option<(&Scene, Range<usize>)>, layer: bool| {
            match layer {
                true => scene.push_layer(bounds),
                false => scene.insert_primitive(quad(bounds)),
            }
            scene.mark();
            let start = scene.len();
            match prev {
                Some((prev, range)) => scene.replay(range, prev),
                None => scene.insert_primitive(quad(rect(60.0, 60.0, 20.0, 20.0))),
            }
            let view = start..scene.len();
            scene.mark();
            if layer {
                scene.pop_layer();
            }
            view
        };
        let mut painted = Scene::default();
        let view = paint(&mut painted, None, true);

        let mut fresh = Scene::default();
        paint(&mut fresh, None, false);

        let mut replayed = Scene::default();
        paint(&mut replayed, Some((&painted, view)), false);

        assert_eq!(orders(&replayed), orders(&fresh));
        assert_eq!(replayed.primitive_bounds.known_inserts, 0);
    }

    #[test]
    fn a_replay_inside_an_open_filter_keeps_its_orders() {
        let content_mask = rect(0.0, 0.0, 500.0, 500.0);
        let paint = |scene: &mut Scene, prev: Option<(&Scene, Range<usize>)>| {
            scene.insert_primitive(quad(rect(0.0, 0.0, 300.0, 300.0)));
            scene.push_filter(
                rect(20.0, 20.0, 200.0, 200.0),
                origin(120.0, 120.0),
                content_mask,
                Filter {
                    blur: 4.0,
                    ..Default::default()
                },
            );
            scene.insert_primitive(quad(rect(20.0, 20.0, 200.0, 200.0)));
            scene.mark();
            let start = scene.len();
            match prev {
                Some((prev, range)) => scene.replay(range, prev),
                None => {
                    scene.push_layer(rect(40.0, 40.0, 100.0, 100.0));
                    scene.insert_primitive(quad(rect(40.0, 40.0, 50.0, 50.0)));
                    scene.pop_layer();
                    scene.insert_primitive(quad(rect(60.0, 60.0, 100.0, 100.0)));
                }
            }
            let view = start..scene.len();
            scene.mark();
            scene.pop_filter();
            scene.insert_primitive(quad(rect(100.0, 100.0, 300.0, 50.0)));
            view
        };
        let mut painted = Scene::default();
        let view = paint(&mut painted, None);

        let mut replayed = Scene::default();
        paint(&mut replayed, Some((&painted, view)));

        assert_eq!(orders(&replayed), orders(&painted));
        assert_eq!(replayed.effects, painted.effects);
        assert_eq!(replayed.history, painted.history);
        assert_eq!(replayed.primitive_bounds.known_inserts, 2);
    }

    #[test]
    fn a_replayed_closed_layer_lifts_only_what_is_painted_over_it() {
        let content_mask = rect(0.0, 0.0, 500.0, 500.0);
        let button = rect(100.0, 60.0, 20.0, 20.0);
        let paint_layer = |scene: &mut Scene| {
            scene.push_filter(button, origin(110.0, 70.0), content_mask, filter(0.9));
            scene.insert_primitive(quad(button));
            scene.pop_filter();
        };
        let mut painted = Scene::default();
        painted.insert_primitive(quad(rect(0.0, 50.0, 500.0, 40.0)));
        painted.mark();
        let start = painted.len();
        paint_layer(&mut painted);
        let layer = start..painted.len();
        painted.mark();

        let mut scene = Scene::default();
        scene.insert_primitive(quad(rect(0.0, 50.0, 500.0, 40.0)));
        scene.mark();
        scene.replay(layer, &painted);
        scene.mark();
        assert_eq!(scene.primitive_bounds.known_inserts, 1);
        assert_eq!(orders(&scene), orders(&painted));
        let row = scene.quads[0].order;
        let layer = scene.effects[0];

        let head = scene.primitive_bounds.insert(rect(0.0, 10.0, 500.0, 30.0));
        assert!(head <= row, "head at {head}, row at {row}");
        assert!(scene.filtered(head).is_none());

        let label = scene.primitive_bounds.insert(rect(10.0, 15.0, 40.0, 20.0));
        assert!(
            label > layer.end,
            "label at {label}, layer ends at {}",
            layer.end
        );

        let over = scene.primitive_bounds.insert(rect(105.0, 65.0, 5.0, 5.0));
        assert!(
            over > layer.end,
            "over at {over}, layer ends at {}",
            layer.end
        );
    }

    #[test]
    fn backdrops_of_another_radius_blur_in_a_batch_of_their_own() {
        let content_mask = rect(0.0, 0.0, 500.0, 500.0);
        let mut scene = Scene::default();
        // A faint haze of thin strips along the top and a frosted menu further down, apart, so
        // nothing else orders between them.
        for (bounds, blur) in [
            (rect(0.0, 0.0, 500.0, 10.0), 1.0),
            (rect(0.0, 10.0, 500.0, 10.0), 1.0),
            (rect(300.0, 200.0, 150.0, 200.0), 8.0),
        ] {
            scene.insert_primitive(Backdrop {
                blur,
                opacity: 1.0,
                bounds,
                content_mask: ContentMask {
                    bounds: content_mask,
                },
                ..Default::default()
            });
        }
        scene.finish();

        let batches: Vec<_> = scene
            .batches()
            .map(|batch| match batch {
                PrimitiveBatch::Backdrops(range) => scene.backdrops[range]
                    .iter()
                    .map(|backdrop| backdrop.blur)
                    .collect::<Vec<_>>(),
                other => panic!("expected only backdrops, got {other:?}"),
            })
            .collect();
        assert_eq!(batches, vec![vec![1.0, 1.0], vec![8.0]]);
    }
}

#[derive(Clone)]
#[expect(missing_docs)]
pub enum Primitive {
    Shadow(Shadow),
    Backdrop(Backdrop),
    Quad(Quad),
    Shader(ShaderQuad),
    Path(Path<ScaledPixels>),
    Underline(Underline),
    MonochromeSprite(MonochromeSprite),
    SubpixelSprite(SubpixelSprite),
    PolychromeSprite(PolychromeSprite),
    Surface(PaintSurface),
}

#[expect(missing_docs)]
impl Primitive {
    /// The primitive with the alpha of its colours scaled by `fade`. A surface has no colour
    /// of its own and comes back as it was.
    pub fn faded(mut self, fade: f32) -> Self {
        let fade = fade.clamp(0., 1.);
        match &mut self {
            Primitive::Shadow(shadow) => shadow.color = shadow.color.opacity(fade),
            Primitive::Backdrop(backdrop) => backdrop.opacity *= fade,
            Primitive::Quad(quad) => {
                quad.background = quad.background.opacity(fade);
                quad.border_color = quad.border_color.opacity(fade);
            }
            Primitive::Shader(shader) => shader.opacity *= fade,
            Primitive::Path(path) => path.color = path.color.opacity(fade),
            Primitive::Underline(underline) => underline.color = underline.color.opacity(fade),
            Primitive::MonochromeSprite(sprite) => sprite.color = sprite.color.opacity(fade),
            Primitive::SubpixelSprite(sprite) => sprite.color = sprite.color.opacity(fade),
            Primitive::PolychromeSprite(sprite) => sprite.opacity *= fade,
            Primitive::Surface(_) => {}
        }
        self
    }

    pub fn bounds(&self) -> &Bounds<ScaledPixels> {
        match self {
            Primitive::Shadow(shadow) => &shadow.bounds,
            Primitive::Backdrop(backdrop) => &backdrop.bounds,
            Primitive::Quad(quad) => &quad.bounds,
            Primitive::Shader(shader) => &shader.bounds,
            Primitive::Path(path) => &path.bounds,
            Primitive::Underline(underline) => &underline.bounds,
            Primitive::MonochromeSprite(sprite) => &sprite.bounds,
            Primitive::SubpixelSprite(sprite) => &sprite.bounds,
            Primitive::PolychromeSprite(sprite) => &sprite.bounds,
            Primitive::Surface(surface) => &surface.bounds,
        }
    }

    pub fn content_mask(&self) -> &ContentMask<ScaledPixels> {
        match self {
            Primitive::Shadow(shadow) => &shadow.content_mask,
            Primitive::Backdrop(backdrop) => &backdrop.content_mask,
            Primitive::Quad(quad) => &quad.content_mask,
            Primitive::Shader(shader) => &shader.content_mask,
            Primitive::Path(path) => &path.content_mask,
            Primitive::Underline(underline) => &underline.content_mask,
            Primitive::MonochromeSprite(sprite) => &sprite.content_mask,
            Primitive::SubpixelSprite(sprite) => &sprite.content_mask,
            Primitive::PolychromeSprite(sprite) => &sprite.content_mask,
            Primitive::Surface(surface) => &surface.content_mask,
        }
    }

    /// The order the primitive was drawn at, once a scene has inserted it.
    pub(crate) fn order(&self) -> DrawOrder {
        match self {
            Primitive::Shadow(shadow) => shadow.order,
            Primitive::Backdrop(backdrop) => backdrop.order,
            Primitive::Quad(quad) => quad.order,
            Primitive::Shader(shader) => shader.order,
            Primitive::Path(path) => path.order,
            Primitive::Underline(underline) => underline.order,
            Primitive::MonochromeSprite(sprite) => sprite.order,
            Primitive::SubpixelSprite(sprite) => sprite.order,
            Primitive::PolychromeSprite(sprite) => sprite.order,
            Primitive::Surface(surface) => surface.order,
        }
    }
}

#[cfg_attr(
    all(
        any(target_os = "linux", target_os = "freebsd"),
        not(any(feature = "x11", feature = "wayland"))
    ),
    allow(dead_code)
)]
struct BatchIterator<'a> {
    /// Where filtered layers open and close, sorted. A batch never spans one of these.
    layered: Vec<DrawOrder>,
    shadows_start: usize,
    shadows_iter: Peekable<slice::Iter<'a, Shadow>>,
    backdrops_start: usize,
    backdrops_iter: Peekable<slice::Iter<'a, Backdrop>>,
    quads_start: usize,
    quads_iter: Peekable<slice::Iter<'a, Quad>>,
    shaders_start: usize,
    shaders_iter: Peekable<slice::Iter<'a, ShaderQuad>>,
    paths_start: usize,
    paths_iter: Peekable<slice::Iter<'a, Path<ScaledPixels>>>,
    underlines_start: usize,
    underlines_iter: Peekable<slice::Iter<'a, Underline>>,
    monochrome_sprites_start: usize,
    monochrome_sprites_iter: Peekable<slice::Iter<'a, MonochromeSprite>>,
    subpixel_sprites_start: usize,
    subpixel_sprites_iter: Peekable<slice::Iter<'a, SubpixelSprite>>,
    polychrome_sprites_start: usize,
    polychrome_sprites_iter: Peekable<slice::Iter<'a, PolychromeSprite>>,
    surfaces_start: usize,
    surfaces_iter: Peekable<slice::Iter<'a, PaintSurface>>,
}

impl<'a> Iterator for BatchIterator<'a> {
    type Item = PrimitiveBatch;

    fn next(&mut self) -> Option<Self::Item> {
        let mut orders_and_kinds = [
            (
                self.shadows_iter.peek().map(|s| s.order),
                PrimitiveKind::Shadow,
            ),
            (
                self.backdrops_iter.peek().map(|b| b.order),
                PrimitiveKind::Backdrop,
            ),
            (self.quads_iter.peek().map(|q| q.order), PrimitiveKind::Quad),
            (
                self.shaders_iter.peek().map(|s| s.order),
                PrimitiveKind::Shader,
            ),
            (self.paths_iter.peek().map(|q| q.order), PrimitiveKind::Path),
            (
                self.underlines_iter.peek().map(|u| u.order),
                PrimitiveKind::Underline,
            ),
            (
                self.monochrome_sprites_iter.peek().map(|s| s.order),
                PrimitiveKind::MonochromeSprite,
            ),
            (
                self.subpixel_sprites_iter.peek().map(|s| s.order),
                PrimitiveKind::SubpixelSprite,
            ),
            (
                self.polychrome_sprites_iter.peek().map(|s| s.order),
                PrimitiveKind::PolychromeSprite,
            ),
            (
                self.surfaces_iter.peek().map(|s| s.order),
                PrimitiveKind::Surface,
            ),
        ];
        orders_and_kinds.sort_by_key(|(order, kind)| (order.unwrap_or(u32::MAX), *kind));

        let first = orders_and_kinds[0];
        let second = orders_and_kinds[1];
        let (batch_kind, max_order_and_kind) = if first.0.is_some() {
            (first.1, (second.0.unwrap_or(u32::MAX), second.1))
        } else {
            return None;
        };

        let first_order = orders_and_kinds[0].0.unwrap_or_default();
        // A batch may not cross into or out of a filtered layer, or the layer's primitives could
        // not be redirected to its own target. Folding the nearest boundary into the threshold the
        // run already compares against keeps that free for every primitive that has no filter.
        let edge = self.layered.partition_point(|order| *order <= first_order);
        let limit = self.layered.get(edge).copied().unwrap_or(DrawOrder::MAX);
        let max_order_and_kind = max_order_and_kind.min((limit, PrimitiveKind::Shadow));

        match batch_kind {
            PrimitiveKind::Shadow => {
                let shadows_start = self.shadows_start;
                let mut shadows_end = shadows_start + 1;
                self.shadows_iter.next();
                while self
                    .shadows_iter
                    .next_if(|shadow| (shadow.order, batch_kind) < max_order_and_kind)
                    .is_some()
                {
                    shadows_end += 1;
                }
                self.shadows_start = shadows_end;
                Some(PrimitiveBatch::Shadows(shadows_start..shadows_end))
            }
            PrimitiveKind::Backdrop => {
                // The renderer blurs a batch once at a single radius, so a backdrop of another
                // radius starts a batch of its own rather than taking on its neighbours' blur.
                let backdrops_start = self.backdrops_start;
                let mut backdrops_end = backdrops_start + 1;
                let blur = self
                    .backdrops_iter
                    .next()
                    .map_or(0., |backdrop| backdrop.blur);
                while self
                    .backdrops_iter
                    .next_if(|backdrop| {
                        (backdrop.order, batch_kind) < max_order_and_kind && backdrop.blur == blur
                    })
                    .is_some()
                {
                    backdrops_end += 1;
                }
                self.backdrops_start = backdrops_end;
                Some(PrimitiveBatch::Backdrops(backdrops_start..backdrops_end))
            }
            PrimitiveKind::Quad => {
                let quads_start = self.quads_start;
                let mut quads_end = quads_start + 1;
                self.quads_iter.next();
                while self
                    .quads_iter
                    .next_if(|quad| (quad.order, batch_kind) < max_order_and_kind)
                    .is_some()
                {
                    quads_end += 1;
                }
                self.quads_start = quads_end;
                Some(PrimitiveBatch::Quads(quads_start..quads_end))
            }
            PrimitiveKind::Shader => {
                let shader = self.shaders_iter.peek().unwrap().shader;
                let shaders_start = self.shaders_start;
                let mut shaders_end = shaders_start + 1;
                self.shaders_iter.next();
                while self
                    .shaders_iter
                    .next_if(|instance| {
                        (instance.order, batch_kind) < max_order_and_kind
                            && instance.shader == shader
                    })
                    .is_some()
                {
                    shaders_end += 1;
                }
                self.shaders_start = shaders_end;
                Some(PrimitiveBatch::Shaders {
                    shader,
                    range: shaders_start..shaders_end,
                })
            }
            PrimitiveKind::Path => {
                let paths_start = self.paths_start;
                let mut paths_end = paths_start + 1;
                self.paths_iter.next();
                while self
                    .paths_iter
                    .next_if(|path| (path.order, batch_kind) < max_order_and_kind)
                    .is_some()
                {
                    paths_end += 1;
                }
                self.paths_start = paths_end;
                Some(PrimitiveBatch::Paths(paths_start..paths_end))
            }
            PrimitiveKind::Underline => {
                let underlines_start = self.underlines_start;
                let mut underlines_end = underlines_start + 1;
                self.underlines_iter.next();
                while self
                    .underlines_iter
                    .next_if(|underline| (underline.order, batch_kind) < max_order_and_kind)
                    .is_some()
                {
                    underlines_end += 1;
                }
                self.underlines_start = underlines_end;
                Some(PrimitiveBatch::Underlines(underlines_start..underlines_end))
            }
            PrimitiveKind::MonochromeSprite => {
                let texture_id = self.monochrome_sprites_iter.peek().unwrap().tile.texture_id;
                let sprites_start = self.monochrome_sprites_start;
                let mut sprites_end = sprites_start + 1;
                self.monochrome_sprites_iter.next();
                while self
                    .monochrome_sprites_iter
                    .next_if(|sprite| {
                        (sprite.order, batch_kind) < max_order_and_kind
                            && sprite.tile.texture_id == texture_id
                    })
                    .is_some()
                {
                    sprites_end += 1;
                }
                self.monochrome_sprites_start = sprites_end;
                Some(PrimitiveBatch::MonochromeSprites {
                    texture_id,
                    range: sprites_start..sprites_end,
                })
            }
            PrimitiveKind::SubpixelSprite => {
                let texture_id = self.subpixel_sprites_iter.peek().unwrap().tile.texture_id;
                let sprites_start = self.subpixel_sprites_start;
                let mut sprites_end = sprites_start + 1;
                self.subpixel_sprites_iter.next();
                while self
                    .subpixel_sprites_iter
                    .next_if(|sprite| {
                        (sprite.order, batch_kind) < max_order_and_kind
                            && sprite.tile.texture_id == texture_id
                    })
                    .is_some()
                {
                    sprites_end += 1;
                }
                self.subpixel_sprites_start = sprites_end;
                Some(PrimitiveBatch::SubpixelSprites {
                    texture_id,
                    range: sprites_start..sprites_end,
                })
            }
            PrimitiveKind::PolychromeSprite => {
                let texture_id = self.polychrome_sprites_iter.peek().unwrap().tile.texture_id;
                let sprites_start = self.polychrome_sprites_start;
                let mut sprites_end = sprites_start + 1;
                self.polychrome_sprites_iter.next();
                while self
                    .polychrome_sprites_iter
                    .next_if(|sprite| {
                        (sprite.order, batch_kind) < max_order_and_kind
                            && sprite.tile.texture_id == texture_id
                    })
                    .is_some()
                {
                    sprites_end += 1;
                }
                self.polychrome_sprites_start = sprites_end;
                Some(PrimitiveBatch::PolychromeSprites {
                    texture_id,
                    range: sprites_start..sprites_end,
                })
            }
            PrimitiveKind::Surface => {
                let surfaces_start = self.surfaces_start;
                let mut surfaces_end = surfaces_start + 1;
                self.surfaces_iter.next();
                while self
                    .surfaces_iter
                    .next_if(|surface| (surface.order, batch_kind) < max_order_and_kind)
                    .is_some()
                {
                    surfaces_end += 1;
                }
                self.surfaces_start = surfaces_end;
                Some(PrimitiveBatch::Surfaces(surfaces_start..surfaces_end))
            }
        }
    }
}

#[derive(Debug)]
#[cfg_attr(
    all(
        any(target_os = "linux", target_os = "freebsd"),
        not(any(feature = "x11", feature = "wayland"))
    ),
    allow(dead_code)
)]
#[allow(missing_docs)]
pub enum PrimitiveBatch {
    Shadows(Range<usize>),
    Backdrops(Range<usize>),
    Quads(Range<usize>),
    /// Instances of one custom shader.
    Shaders {
        shader: ShaderId,
        range: Range<usize>,
    },
    Paths(Range<usize>),
    Underlines(Range<usize>),
    MonochromeSprites {
        texture_id: AtlasTextureId,
        range: Range<usize>,
    },
    #[cfg_attr(target_os = "macos", allow(dead_code))]
    SubpixelSprites {
        texture_id: AtlasTextureId,
        range: Range<usize>,
    },
    PolychromeSprites {
        texture_id: AtlasTextureId,
        range: Range<usize>,
    },
    Surfaces(Range<usize>),
}

impl PrimitiveBatch {
    #[expect(missing_docs)]
    pub fn label(&self) -> String {
        match self {
            Self::Shadows(range) => format!("shadows ({})", range.len()),
            Self::Backdrops(range) => format!("backdrops ({})", range.len()),
            Self::Quads(range) => format!("quads ({})", range.len()),
            Self::Shaders { shader, range } => {
                format!("shader {} ({})", shader.0, range.len())
            }
            Self::Paths(range) => format!("paths ({})", range.len()),
            Self::Underlines(range) => format!("underlines ({})", range.len()),
            Self::MonochromeSprites { texture_id, range } => {
                format!(
                    "monochrome sprites ({}) on atlas {}",
                    range.len(),
                    texture_id.index
                )
            }
            Self::SubpixelSprites { texture_id, range } => {
                format!(
                    "subpixel sprites ({}) on atlas {}",
                    range.len(),
                    texture_id.index
                )
            }
            Self::PolychromeSprites { texture_id, range } => {
                format!(
                    "polychrome sprites ({}) on atlas {}",
                    range.len(),
                    texture_id.index
                )
            }
            Self::Surfaces(range) => format!("surfaces ({})", range.len()),
        }
    }
}

#[derive(Default, Debug, Copy, Clone)]
#[repr(C)]
#[expect(missing_docs)]
pub struct Backdrop {
    pub order: DrawOrder,
    /// How far the corners are smoothed from circular arcs towards a squircle, from 0 to 1.
    pub corner_smoothing: f32,
    /// The standard deviation of the blur, in scaled pixels.
    pub blur: f32,
    /// How much of the blurred backdrop shows through.
    pub opacity: f32,
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    pub corner_radii: Corners<ScaledPixels>,
}

impl From<Backdrop> for Primitive {
    fn from(backdrop: Backdrop) -> Self {
        Primitive::Backdrop(backdrop)
    }
}

#[derive(Debug, Copy, Clone)]
#[repr(C)]
#[expect(missing_docs)]
pub struct Quad {
    pub order: DrawOrder,
    pub border_style: BorderStyle,
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    pub background: Background,
    pub border_color: Hsla,
    pub corner_radii: Corners<ScaledPixels>,
    pub border_widths: Edges<ScaledPixels>,
    /// How far the corners are smoothed from circular arcs towards a squircle, from 0 to 1.
    pub corner_smoothing: f32,
    pub pad: u32, // align to 8 bytes
}

impl From<Quad> for Primitive {
    fn from(quad: Quad) -> Self {
        Primitive::Quad(quad)
    }
}

/// One instance of a custom shader, drawn over a rounded rectangle. Its layout is the
/// `GpuiShaderInstance` struct of the shader prelude, so the renderers upload it as is.
#[derive(Debug, Copy, Clone)]
#[repr(C)]
pub struct ShaderQuad {
    /// Where the instance falls among the frame's primitives.
    pub order: DrawOrder,
    /// The shader that colours it.
    pub shader: ShaderId,
    /// The element opacity it is faded by.
    pub opacity: f32,
    /// The time the app painted it with, in seconds.
    pub time: f32,
    /// Its bounds in device pixels.
    pub bounds: Bounds<ScaledPixels>,
    /// The area it is clipped to.
    pub content_mask: ContentMask<ScaledPixels>,
    /// The radii of its corners in device pixels.
    pub corner_radii: Corners<ScaledPixels>,
    /// Device pixels per logical pixel.
    pub scale_factor: f32,
    /// Keeps `params` on the 16 byte boundary the shader reads it at.
    pub pad: [u32; 3],
    /// The parameters the app painted it with.
    pub params: [f32; crate::SHADER_PARAM_COUNT],
    /// The colours the app painted it with, as straight alpha sRGB.
    pub colors: [[f32; 4]; crate::SHADER_COLOR_COUNT],
}

impl From<ShaderQuad> for Primitive {
    fn from(shader: ShaderQuad) -> Self {
        Primitive::Shader(shader)
    }
}

#[derive(Debug, Copy, Clone)]
#[repr(C)]
#[expect(missing_docs)]
pub struct Underline {
    pub order: DrawOrder,
    pub pad: u32, // align to 8 bytes
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    pub color: Hsla,
    pub thickness: ScaledPixels,
    pub wavy: PaddedBool32,
}

impl From<Underline> for Primitive {
    fn from(underline: Underline) -> Self {
        Primitive::Underline(underline)
    }
}

#[derive(Debug, Copy, Clone)]
#[repr(C)]
#[expect(missing_docs)]
pub struct Shadow {
    pub order: DrawOrder,
    pub blur_radius: ScaledPixels,
    pub bounds: Bounds<ScaledPixels>,
    pub corner_radii: Corners<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    pub color: Hsla,
    pub element_bounds: Bounds<ScaledPixels>,
    pub element_corner_radii: Corners<ScaledPixels>,
    /// 0 = drop shadow (rendered outside the element), 1 = inset shadow (rendered inside).
    pub inset: u32,
    /// The corner smoothing of the element's own outline, which clips the shadow.
    pub element_corner_smoothing: f32,
}

impl From<Shadow> for Primitive {
    fn from(shadow: Shadow) -> Self {
        Primitive::Shadow(shadow)
    }
}

/// The style of a border.
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[repr(C)]
pub enum BorderStyle {
    /// A solid border.
    #[default]
    Solid = 0,
    /// A dashed border.
    Dashed = 1,
}

/// A data type representing a 2 dimensional transformation that can be applied to an element.
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct TransformationMatrix {
    /// 2x2 matrix containing rotation and scale,
    /// stored row-major
    pub rotation_scale: [[f32; 2]; 2],
    /// translation vector
    pub translation: [f32; 2],
}

impl Eq for TransformationMatrix {}

impl TransformationMatrix {
    /// The unit matrix, has no effect.
    pub fn unit() -> Self {
        Self {
            rotation_scale: [[1.0, 0.0], [0.0, 1.0]],
            translation: [0.0, 0.0],
        }
    }

    /// Move the origin by a given point
    pub fn translate(self, point: Point<ScaledPixels>) -> Self {
        self.compose(Self {
            rotation_scale: [[1.0, 0.0], [0.0, 1.0]],
            translation: [point.x.0, point.y.0],
        })
    }

    /// Clockwise rotation in radians around the origin
    pub fn rotate(self, angle: Radians) -> Self {
        self.compose(Self {
            rotation_scale: [
                [angle.0.cos(), -angle.0.sin()],
                [angle.0.sin(), angle.0.cos()],
            ],
            translation: [0.0, 0.0],
        })
    }

    /// Scale around the origin
    pub fn scale(self, size: Size<f32>) -> Self {
        self.compose(Self {
            rotation_scale: [[size.width, 0.0], [0.0, size.height]],
            translation: [0.0, 0.0],
        })
    }

    /// Perform matrix multiplication with another transformation
    /// to produce a new transformation that is the result of
    /// applying both transformations: first, `other`, then `self`.
    #[inline]
    pub fn compose(self, other: TransformationMatrix) -> TransformationMatrix {
        if other == Self::unit() {
            return self;
        }
        // Perform matrix multiplication
        TransformationMatrix {
            rotation_scale: [
                [
                    self.rotation_scale[0][0] * other.rotation_scale[0][0]
                        + self.rotation_scale[0][1] * other.rotation_scale[1][0],
                    self.rotation_scale[0][0] * other.rotation_scale[0][1]
                        + self.rotation_scale[0][1] * other.rotation_scale[1][1],
                ],
                [
                    self.rotation_scale[1][0] * other.rotation_scale[0][0]
                        + self.rotation_scale[1][1] * other.rotation_scale[1][0],
                    self.rotation_scale[1][0] * other.rotation_scale[0][1]
                        + self.rotation_scale[1][1] * other.rotation_scale[1][1],
                ],
            ],
            translation: [
                self.translation[0]
                    + self.rotation_scale[0][0] * other.translation[0]
                    + self.rotation_scale[0][1] * other.translation[1],
                self.translation[1]
                    + self.rotation_scale[1][0] * other.translation[0]
                    + self.rotation_scale[1][1] * other.translation[1],
            ],
        }
    }

    /// Apply transformation to a point, mainly useful for debugging
    pub fn apply(&self, point: Point<Pixels>) -> Point<Pixels> {
        let input = [point.x.0, point.y.0];
        let mut output = self.translation;
        for (i, output_cell) in output.iter_mut().enumerate() {
            for (k, input_cell) in input.iter().enumerate() {
                *output_cell += self.rotation_scale[i][k] * *input_cell;
            }
        }
        Point::new(output[0].into(), output[1].into())
    }
}

impl Default for TransformationMatrix {
    fn default() -> Self {
        Self::unit()
    }
}

#[derive(Copy, Clone, Debug)]
#[repr(C)]
#[expect(missing_docs)]
pub struct MonochromeSprite {
    pub order: DrawOrder,
    pub pad: u32,
    pub tile: AtlasTile,
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    pub color: Hsla,
    pub transformation: TransformationMatrix,
}

impl From<MonochromeSprite> for Primitive {
    fn from(sprite: MonochromeSprite) -> Self {
        Primitive::MonochromeSprite(sprite)
    }
}

#[derive(Copy, Clone, Debug)]
#[repr(C)]
#[expect(missing_docs)]
pub struct SubpixelSprite {
    pub order: DrawOrder,
    pub pad: u32, // align to 8 bytes
    pub tile: AtlasTile,
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    pub color: Hsla,
    pub transformation: TransformationMatrix,
}

impl From<SubpixelSprite> for Primitive {
    fn from(sprite: SubpixelSprite) -> Self {
        Primitive::SubpixelSprite(sprite)
    }
}

#[derive(Copy, Clone, Debug)]
#[repr(C)]
#[expect(missing_docs)]
pub struct PolychromeSprite {
    pub order: DrawOrder,
    pub pad: u32,
    pub tile: AtlasTile,
    pub grayscale: PaddedBool32,
    pub opacity: f32,
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    pub corner_radii: Corners<ScaledPixels>,
}

impl From<PolychromeSprite> for Primitive {
    fn from(sprite: PolychromeSprite) -> Self {
        Primitive::PolychromeSprite(sprite)
    }
}

#[derive(Clone, Debug)]
#[allow(missing_docs)]
pub struct PaintSurface {
    pub order: DrawOrder,
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    pub image_buffer: objc2_core_foundation::CFRetained<objc2_core_video::CVPixelBuffer>,
}

impl From<PaintSurface> for Primitive {
    fn from(surface: PaintSurface) -> Self {
        Primitive::Surface(surface)
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[expect(missing_docs)]
pub struct PathId(pub usize);

/// A line made up of a series of vertices and control points.
#[derive(Clone, Debug)]
#[expect(missing_docs)]
pub struct Path<P: Clone + Debug + Default + PartialEq> {
    pub id: PathId,
    pub order: DrawOrder,
    pub bounds: Bounds<P>,
    pub content_mask: ContentMask<P>,
    pub vertices: Vec<PathVertex<P>>,
    pub color: Background,
    start: Point<P>,
    current: Point<P>,
    contour_count: usize,
}

impl Path<Pixels> {
    /// Create a new path with the given starting point.
    pub fn new(start: Point<Pixels>) -> Self {
        Self {
            id: PathId(0),
            order: DrawOrder::default(),
            vertices: Vec::new(),
            start,
            current: start,
            bounds: Bounds {
                origin: start,
                size: Default::default(),
            },
            content_mask: Default::default(),
            color: Default::default(),
            contour_count: 0,
        }
    }

    /// Scale this path by the given factor.
    pub fn scale(&self, factor: f32) -> Path<ScaledPixels> {
        Path {
            id: self.id,
            order: self.order,
            bounds: self.bounds.scale(factor),
            content_mask: self.content_mask.scale(factor),
            vertices: self
                .vertices
                .iter()
                .map(|vertex| vertex.scale(factor))
                .collect(),
            start: self.start.map(|start| start.scale(factor)),
            current: self.current.scale(factor),
            contour_count: self.contour_count,
            color: self.color,
        }
    }

    /// Move the start, current point to the given point.
    pub fn move_to(&mut self, to: Point<Pixels>) {
        self.contour_count += 1;
        self.start = to;
        self.current = to;
    }

    /// Draw a straight line from the current point to the given point.
    pub fn line_to(&mut self, to: Point<Pixels>) {
        self.contour_count += 1;
        if self.contour_count > 1 {
            self.push_triangle(
                (self.start, self.current, to),
                (point(0., 1.), point(0., 1.), point(0., 1.)),
            );
        }
        self.current = to;
    }

    /// Draw a curve from the current point to the given point, using the given control point.
    pub fn curve_to(&mut self, to: Point<Pixels>, ctrl: Point<Pixels>) {
        self.contour_count += 1;
        if self.contour_count > 1 {
            self.push_triangle(
                (self.start, self.current, to),
                (point(0., 1.), point(0., 1.), point(0., 1.)),
            );
        }

        self.push_triangle(
            (self.current, ctrl, to),
            (point(0., 0.), point(0.5, 0.), point(1., 1.)),
        );
        self.current = to;
    }

    /// Push a triangle to the Path.
    pub fn push_triangle(
        &mut self,
        xy: (Point<Pixels>, Point<Pixels>, Point<Pixels>),
        st: (Point<f32>, Point<f32>, Point<f32>),
    ) {
        self.bounds = self
            .bounds
            .union(&Bounds {
                origin: xy.0,
                size: Default::default(),
            })
            .union(&Bounds {
                origin: xy.1,
                size: Default::default(),
            })
            .union(&Bounds {
                origin: xy.2,
                size: Default::default(),
            });

        self.vertices.push(PathVertex {
            xy_position: xy.0,
            st_position: st.0,
            content_mask: Default::default(),
        });
        self.vertices.push(PathVertex {
            xy_position: xy.1,
            st_position: st.1,
            content_mask: Default::default(),
        });
        self.vertices.push(PathVertex {
            xy_position: xy.2,
            st_position: st.2,
            content_mask: Default::default(),
        });
    }
}

impl<T> Path<T>
where
    T: Clone + Debug + Default + PartialEq + PartialOrd + Add<T, Output = T> + Sub<Output = T>,
{
    #[allow(unused)]
    #[expect(missing_docs)]
    pub fn clipped_bounds(&self) -> Bounds<T> {
        self.bounds.intersect(&self.content_mask.bounds)
    }
}

impl From<Path<ScaledPixels>> for Primitive {
    fn from(path: Path<ScaledPixels>) -> Self {
        Primitive::Path(path)
    }
}

#[derive(Clone, Debug)]
#[repr(C)]
#[expect(missing_docs)]
pub struct PathVertex<P: Clone + Debug + Default + PartialEq> {
    pub xy_position: Point<P>,
    pub st_position: Point<f32>,
    pub content_mask: ContentMask<P>,
}

#[expect(missing_docs)]
impl PathVertex<Pixels> {
    pub fn scale(&self, factor: f32) -> PathVertex<ScaledPixels> {
        PathVertex {
            xy_position: self.xy_position.scale(factor),
            st_position: self.st_position,
            content_mask: self.content_mask.scale(factor),
        }
    }
}
