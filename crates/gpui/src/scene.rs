// todo("windows"): remove
#![cfg_attr(windows, allow(dead_code))]

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::{
    AtlasTextureId, AtlasTile, Background, Bounds, ContentMask, Corners, DevicePixels, Edges, Hsla,
    Path, PathTiles, Pixels, Point, Radians, ScaledPixels, ShaderId, Size, bounds_tree::BoundsTree,
    point,
};
use collections::FxHashMap;
use smallvec::SmallVec;
use std::{collections::VecDeque, fmt::Debug, iter::Peekable, ops::Range, rc::Rc, slice};

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
enum TreeEvent {
    /// Bounds going in with the kind they take and the highest kind they may share an order
    /// with, which together decide the order they get.
    Insert(PrimitiveKind, PrimitiveKind),
    PushLayer,
    PopLayer,
    Open,
    Close,
}

impl TreeEvent {
    fn word(self) -> u64 {
        match self {
            Self::Insert(kind, shared) => 1 | (kind as u64) << 8 | (shared as u64) << 16,
            Self::PushLayer => 2,
            Self::PopLayer => 3,
            Self::Open => 4,
            Self::Close => 5,
        }
    }
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

impl PaddedBool32 {
    /// The wrapped boolean.
    pub fn as_bool(self) -> bool {
        self.0 != 0
    }
}

/// A paint layer, whose primitives get one order between them once the outermost open layer
/// closes.
struct Layer {
    /// The paint operation that opened the layer, which keeps the layer's key once it closes.
    start: usize,
    primitives: SmallVec<[Range<usize>; 1]>,
    bounds: Bounds<ScaledPixels>,
    max_kind: PrimitiveKind,
    max_shared_kind: PrimitiveKind,
    /// The order an earlier frame gave this layer, when a replay that keeps its orders opened it
    /// along with every layer around it. Its bounds and kinds are then that frame's too, so its
    /// primitives go in without adding to them.
    known: Option<DrawOrder>,
    /// Whether its primitives go straight into the scene at the known order. Only an outermost
    /// layer can, since a nested layer's primitives follow all of its parent's.
    direct: bool,
}

/// How a closed paint layer went into the bounds tree, kept with the paint operation that opened
/// it so that a replay can open it again settled.
#[derive(Clone, Copy)]
pub(crate) struct LayerKey {
    bounds: Bounds<ScaledPixels>,
    order: DrawOrder,
    max_kind: PrimitiveKind,
    max_shared_kind: PrimitiveKind,
}

#[derive(Default)]
#[expect(missing_docs)]
pub struct Scene {
    pub(crate) paint_operations: Vec<PaintOperation>,
    primitive_bounds: BoundsTree<ScaledPixels>,
    /// A running hash of every insert into the bounds tree this frame, a primitive's or a closed
    /// paint layer's, and of every layer and filter it opened and closed, in order. Two scenes
    /// with the same history hand out the same orders from there on.
    history: u64,
    /// The history at chosen paint operation indices, in rising index order, so a later frame
    /// that replays from one of them can tell whether its orders still hold.
    marks: Vec<(usize, u64)>,
    layer_stack: Vec<usize>,
    layers: VecDeque<Layer>,
    spans: FxHashMap<SpanId, PaintSpan>,
    span_stack: Vec<(SpanId, usize, Bounds<Pixels>)>,
    pub shadows: Vec<Shadow>,
    pub backdrops: Vec<Backdrop>,
    pub effects: Vec<LayerEffect>,
    effect_stack: Vec<usize>,
    pub quads: Vec<Quad>,
    pub shaders: Vec<ShaderQuad>,
    pub paths: Vec<PaintedPath>,
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
        self.layers.clear();
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

    pub fn push_layer(&mut self) {
        self.open_layer(None);
    }

    /// Opens a paint layer, settled on `key` when a replay that keeps its orders knows how the
    /// layer went into the tree in an earlier frame.
    fn open_layer(&mut self, key: Option<LayerKey>) {
        self.record(TreeEvent::PushLayer, Bounds::default());
        let direct = key.is_some() && self.layer_stack.is_empty();
        self.layer_stack.push(self.layers.len());
        let start = self.paint_operations.len();
        self.layers.push_back(match key {
            Some(key) => Layer {
                start,
                primitives: SmallVec::new(),
                bounds: key.bounds,
                max_kind: key.max_kind,
                max_shared_kind: key.max_shared_kind,
                known: Some(key.order),
                direct,
            },
            None => Layer {
                start,
                primitives: SmallVec::new(),
                bounds: Bounds::default(),
                max_kind: PrimitiveKind::Shadow,
                max_shared_kind: PrimitiveKind::Surface,
                known: None,
                direct,
            },
        });
        self.paint_operations.push(PaintOperation::StartLayer(None));
    }

    pub fn pop_layer(&mut self) {
        self.record(TreeEvent::PopLayer, Bounds::default());
        self.layer_stack.pop();
        if self.layer_stack.is_empty() {
            while let Some(layer) = self.layers.pop_front() {
                self.insert_layer(layer);
            }
        }
        self.paint_operations.push(PaintOperation::EndLayer);
    }

    /// Gives a closed layer its order and its primitives that order, in the scene and in the
    /// paint operations a later frame may replay with their orders kept.
    fn insert_layer(&mut self, layer: Layer) {
        if layer.primitives.is_empty() {
            return;
        }
        self.record(
            TreeEvent::Insert(layer.max_kind, layer.max_shared_kind),
            layer.bounds,
        );
        let order = match layer.known {
            Some(order) => {
                self.primitive_bounds
                    .insert_known(layer.bounds, order, layer.max_kind);
                order
            }
            None => {
                self.primitive_bounds
                    .insert(layer.bounds, layer.max_kind, layer.max_shared_kind)
            }
        };
        self.paint_operations[layer.start] = PaintOperation::StartLayer(Some(LayerKey {
            bounds: layer.bounds,
            order,
            max_kind: layer.max_kind,
            max_shared_kind: layer.max_shared_kind,
        }));
        if layer.direct {
            return;
        }
        for index in layer.primitives.into_iter().flatten() {
            if let PaintOperation::Primitive(primitive) = &mut self.paint_operations[index] {
                primitive.set_order(order);
                let primitive = primitive.clone();
                self.push_primitive(primitive, order);
            }
        }
    }

    /// Starts a layer that is drawn offscreen, put through the given filter, and composited back.
    ///
    /// The layer claims a range of orders nothing outside it can share, above everything painted
    /// before it, while its contents order among themselves the way they would anywhere else, so
    /// a quad still covers an image painted before it. Only what is painted later over the area
    /// the layer lands on goes above the range. The rest keeps the order it would have had
    /// without the layer, so a backdrop beside a pressed button blurs what it always did.
    ///
    /// No paint layer may be open, since its primitives get their orders only once it closes,
    /// after this layer. gpui opens paint layers only around a line of text.
    pub fn push_filter(
        &mut self,
        source_bounds: Bounds<ScaledPixels>,
        transform_origin: Point<ScaledPixels>,
        content_mask: Bounds<ScaledPixels>,
        filter: Filter,
    ) {
        debug_assert!(
            self.layer_stack.is_empty(),
            "a filtered layer opened inside a paint layer"
        );
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
                PaintOperation::StartLayer(_) => {
                    layers += 1;
                    self.push_layer();
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
        let mut primitive = primitive.into();
        let clipped_bounds = primitive.clipped_bounds();
        if clipped_bounds.is_empty() {
            return;
        }

        let kind = primitive.kind();
        if !self.add_to_layer(clipped_bounds, kind) {
            self.record(
                TreeEvent::Insert(kind, kind.max_shared_kind()),
                clipped_bounds,
            );
            let order = self
                .primitive_bounds
                .insert(clipped_bounds, kind, kind.max_shared_kind());
            primitive.set_order(order);
            self.push_primitive(primitive.clone(), order);
        }
        self.paint_operations
            .push(PaintOperation::Primitive(primitive));
    }

    /// Inserts a primitive replayed from a scene whose history matched this one's where the
    /// replay began, with the order it was drawn at there.
    fn keep_primitive(&mut self, primitive: Primitive) {
        let index = self.paint_operations.len();
        if let Some(layer) = self
            .layer_stack
            .last()
            .and_then(|index| self.layers.get_mut(*index))
            && let Some(order) = layer.known
        {
            let direct = layer.direct;
            Self::extend_layer(layer, index);
            if direct {
                self.push_primitive(primitive.clone(), order);
            }
            self.paint_operations
                .push(PaintOperation::Primitive(primitive));
            return;
        }
        let clipped_bounds = primitive.clipped_bounds();
        let kind = primitive.kind();
        let order = primitive.order();
        if !self.add_to_layer(clipped_bounds, kind) {
            self.record(
                TreeEvent::Insert(kind, kind.max_shared_kind()),
                clipped_bounds,
            );
            self.primitive_bounds
                .insert_known(clipped_bounds, order, kind);
            self.push_primitive(primitive.clone(), order);
        }
        self.paint_operations
            .push(PaintOperation::Primitive(primitive));
    }

    /// Adds the primitive that becomes the next paint operation to the innermost open paint
    /// layer, or returns false when none is open.
    fn add_to_layer(&mut self, clipped_bounds: Bounds<ScaledPixels>, kind: PrimitiveKind) -> bool {
        let index = self.paint_operations.len();
        let Some(layer) = self
            .layer_stack
            .last()
            .and_then(|index| self.layers.get_mut(*index))
        else {
            return false;
        };
        layer.bounds = layer.bounds.join(&clipped_bounds);
        layer.max_kind = layer.max_kind.max(kind);
        layer.max_shared_kind = layer.max_shared_kind.min(kind.max_shared_kind());
        Self::extend_layer(layer, index);
        true
    }

    /// Counts the paint operation at `index` among the layer's primitives.
    fn extend_layer(layer: &mut Layer, index: usize) {
        match layer.primitives.last_mut() {
            Some(range) if range.end == index => range.end += 1,
            _ => layer.primitives.push(index..index + 1),
        }
    }

    fn push_primitive(&mut self, primitive: Primitive, order: DrawOrder) {
        match primitive {
            Primitive::Shadow(shadow) => self.shadows.push(Shadow { order, ..shadow }),
            Primitive::Backdrop(backdrop) => self.backdrops.push(Backdrop { order, ..backdrop }),
            Primitive::Quad(quad) => self.quads.push(Quad { order, ..quad }),
            Primitive::Shader(shader) => self.shaders.push(ShaderQuad { order, ..shader }),
            Primitive::Path(mut path) => {
                path.order = order;
                self.paths.push(path);
            }
            Primitive::Underline(underline) => {
                self.underlines.push(Underline { order, ..underline })
            }
            Primitive::MonochromeSprite(sprite) => self
                .monochrome_sprites
                .push(MonochromeSprite { order, ..sprite }),
            Primitive::SubpixelSprite(sprite) => self
                .subpixel_sprites
                .push(SubpixelSprite { order, ..sprite }),
            Primitive::PolychromeSprite(sprite) => self
                .polychrome_sprites
                .push(PolychromeSprite { order, ..sprite }),
            Primitive::Surface(mut surface) => {
                surface.order = order;
                self.surfaces.push(surface);
            }
        }
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
        // A paint layer gets its order when the outermost one closes, so a layer keeps its
        // order only when the replay opens every layer around it and closes it again.
        let layers_kept = keep
            && self.layer_stack.is_empty()
            && closes_its_layers(&prev_scene.paint_operations[range.clone()]);
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
                PaintOperation::StartLayer(key) => match layers_kept {
                    true => self.open_layer(*key),
                    false => self.push_layer(),
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

    /// Folds one event into the history.
    fn record(&mut self, event: TreeEvent, bounds: Bounds<ScaledPixels>) {
        let pair = |a: ScaledPixels, b: ScaledPixels| {
            (u64::from(a.0.to_bits()) << 32) | u64::from(b.0.to_bits())
        };
        for word in [
            event.word(),
            pair(bounds.origin.x, bounds.origin.y),
            pair(bounds.size.width, bounds.size.height),
        ] {
            self.history = (self.history.rotate_left(5) ^ word).wrapping_mul(HISTORY_SEED);
        }
    }

    pub fn finish(&mut self) {
        debug_assert!(self.layer_stack.is_empty(), "unbalanced scene layers");
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

impl PrimitiveKind {
    /// We guarantee that overlapping primitives are drawn in paint order.
    /// But, to improve GPU draw batching, the renderer sorts primitives that
    /// share a draw order by kind, then by atlas texture, and only then in
    /// paint order. So, a primitive can only share the draw order of the
    /// topmost primitive it overlaps if that primitive's kind is at most the
    /// one returned here. Otherwise, it needs a new draw order. A backdrop shares only
    /// with shadows and other backdrops, so every quad it overlaps is drawn before it blurs.
    /// A custom shader shares at most with a quad, since shaders sharing an order are sorted by
    /// their id rather than in paint order.
    fn max_shared_kind(self) -> Self {
        match self {
            Self::Shader => Self::Quad,
            Self::MonochromeSprite => Self::Underline,
            Self::SubpixelSprite => Self::MonochromeSprite,
            Self::PolychromeSprite => Self::SubpixelSprite,
            kind => kind,
        }
    }
}

pub(crate) enum PaintOperation {
    Primitive(Primitive),
    /// A paint layer opens, with how it went into the tree once it closed.
    StartLayer(Option<LayerKey>),
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

/// Whether `operations` close every paint layer they open and no other.
fn closes_its_layers(operations: &[PaintOperation]) -> bool {
    let mut depth = 0usize;
    for operation in operations {
        match operation {
            PaintOperation::StartLayer(_) => depth += 1,
            PaintOperation::EndLayer => match depth.checked_sub(1) {
                Some(outer) => depth = outer,
                None => return false,
            },
            _ => {}
        }
    }
    depth == 0
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
        let row = quad_order(&mut scene, rect(0.0, 50.0, 500.0, 40.0));
        scene.push_filter(button, origin(110.0, 70.0), content_mask, filter(0.9));
        quad_order(&mut scene, button);
        scene.pop_filter();
        let layer = scene.effects[0];

        // A table head painted after the pressed row, above it, keeps the order it has with the
        // button at rest, so its backdrop still blurs before the row is drawn.
        let head = quad_order(&mut scene, rect(0.0, 10.0, 500.0, 30.0));
        assert!(head <= row, "head at {head}, row at {row}");
        assert!(scene.filtered(head).is_none());

        // What stacks on the head without sharing its order climbs past the layer's orders
        // rather than into them.
        let label = scene.primitive_bounds.insert(
            rect(10.0, 15.0, 40.0, 20.0),
            PrimitiveKind::Shadow,
            PrimitiveKind::Shadow,
        );
        assert!(
            label > layer.end,
            "label at {label}, layer ends at {}",
            layer.end
        );

        // What is painted over the button still lands above its layer.
        let over = quad_order(&mut scene, rect(105.0, 65.0, 5.0, 5.0));
        assert!(
            over > layer.end,
            "over at {over}, layer ends at {}",
            layer.end
        );
    }

    /// Asks the scene's bounds tree for the order of a quad at `bounds`.
    fn quad_order(scene: &mut Scene, bounds: Bounds<ScaledPixels>) -> DrawOrder {
        scene
            .primitive_bounds
            .insert(bounds, PrimitiveKind::Quad, PrimitiveKind::Quad)
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

    fn underline(bounds: Bounds<ScaledPixels>) -> Underline {
        Underline {
            order: 0,
            pad: 0,
            bounds,
            content_mask: ContentMask {
                bounds: rect(0.0, 0.0, 500.0, 500.0),
            },
            color: Hsla::default(),
            thickness: ScaledPixels(1.0),
            wavy: false.into(),
        }
    }

    /// Paints three views the way a window does, marking the scene around each one, and
    /// returns their paint operation ranges. A view is painted live when `prev` is `None`,
    /// and otherwise replayed from the matching range of `prev`. The first view's middle
    /// primitive is an underline drawn at `moved`, so a quad over it takes a new order.
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
                scene.insert_primitive(underline(moved));
                scene.insert_primitive(quad(rect(150.0, 50.0, 100.0, 100.0)));
            },
            &|scene| {
                scene.push_layer();
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

    /// The order of every primitive, in paint order.
    fn orders(scene: &Scene) -> Vec<DrawOrder> {
        scene
            .paint_operations
            .iter()
            .filter_map(|operation| match operation {
                PaintOperation::Primitive(primitive) => Some(primitive.order()),
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
                true => scene.push_layer(),
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
    fn a_kept_replay_of_nested_layers_draws_them_as_painted() {
        let view = |scene: &mut Scene| {
            scene.push_layer();
            scene.insert_primitive(quad(rect(0.0, 0.0, 10.0, 10.0)));
            scene.push_layer();
            scene.insert_primitive(quad(rect(5.0, 0.0, 10.0, 10.0)));
            scene.pop_layer();
            scene.insert_primitive(quad(rect(8.0, 0.0, 10.0, 10.0)));
            scene.pop_layer();
        };
        let mut painted = Scene::default();
        painted.mark();
        view(&mut painted);
        let range = 0..painted.len();
        painted.mark();

        let mut replayed = Scene::default();
        replayed.mark();
        replayed.replay(range, &painted);
        replayed.mark();

        // The nested layer shares its parent's order, so the quads draw in the order they went
        // into the scene, the parent's before the nested layer's.
        let quads = |scene: &Scene| {
            scene
                .quads
                .iter()
                .map(|quad| (quad.bounds, quad.order))
                .collect::<Vec<_>>()
        };
        assert_eq!(quads(&replayed), quads(&painted));
        assert_eq!(painted.quads[2].bounds, rect(5.0, 0.0, 10.0, 10.0));
        assert_eq!(replayed.history, painted.history);
        assert_eq!(replayed.primitive_bounds.searches, 0);
        assert_eq!(replayed.primitive_bounds.known_inserts, 2);
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
                    scene.push_layer();
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

        let head = quad_order(&mut scene, rect(0.0, 10.0, 500.0, 30.0));
        assert!(head <= row, "head at {head}, row at {row}");
        assert!(scene.filtered(head).is_none());

        let label = scene.primitive_bounds.insert(
            rect(10.0, 15.0, 40.0, 20.0),
            PrimitiveKind::Shadow,
            PrimitiveKind::Shadow,
        );
        assert!(
            label > layer.end,
            "label at {label}, layer ends at {}",
            layer.end
        );

        let over = quad_order(&mut scene, rect(105.0, 65.0, 5.0, 5.0));
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

    #[test]
    fn an_order_shared_inside_a_filter_stays_inside_its_range() {
        let content_mask = rect(0.0, 0.0, 500.0, 500.0);
        let mut scene = Scene::default();
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
        // Outside a filter this quad would share the order of the one under it.
        scene.insert_primitive(quad(rect(40.0, 40.0, 50.0, 50.0)));
        scene.pop_filter();
        // Nothing shares the order the layer ends at, so a quad over it goes above the layer.
        scene.insert_primitive(quad(rect(60.0, 60.0, 10.0, 10.0)));

        let layer = scene.effects[0];
        let [under, inside, over] = [0, 1, 2].map(|index| scene.quads[index].order);
        assert_eq!(scene.filtered(under), None);
        assert!(
            inside > layer.start,
            "inside at {inside}, layer at {layer:?}"
        );
        assert_eq!(scene.filtered(inside), Some(0));
        assert!(
            over > layer.end,
            "over at {over}, layer ends at {}",
            layer.end
        );
    }

    #[test]
    fn backdrops_and_shaders_take_their_place_among_the_kinds() {
        let content_mask = ContentMask {
            bounds: rect(0.0, 0.0, 500.0, 500.0),
        };
        let mut scene = Scene::default();
        scene.insert_primitive(quad(rect(0.0, 0.0, 200.0, 200.0)));
        // A backdrop blurs the quad under it, so it takes a new order.
        scene.insert_primitive(Backdrop {
            blur: 4.0,
            opacity: 1.0,
            bounds: rect(50.0, 50.0, 100.0, 100.0),
            content_mask,
            ..Default::default()
        });
        // A tint over the backdrop and a shader over the tint share its order and draw after it.
        scene.insert_primitive(quad(rect(50.0, 50.0, 100.0, 100.0)));
        scene.insert_primitive(ShaderQuad {
            order: 0,
            shader: ShaderId(7),
            opacity: 1.0,
            time: 0.0,
            bounds: rect(60.0, 60.0, 20.0, 20.0),
            content_mask,
            corner_radii: Corners::default(),
            scale_factor: 1.0,
            pad: [0; 3],
            params: [0.0; crate::SHADER_PARAM_COUNT],
            colors: [[0.0; 4]; crate::SHADER_COLOR_COUNT],
        });
        // Another shader over that one takes a new order, so it draws on top despite its lower id.
        scene.insert_primitive(ShaderQuad {
            order: 0,
            shader: ShaderId(3),
            opacity: 1.0,
            time: 0.0,
            bounds: rect(65.0, 65.0, 20.0, 20.0),
            content_mask,
            corner_radii: Corners::default(),
            scale_factor: 1.0,
            pad: [0; 3],
            params: [0.0; crate::SHADER_PARAM_COUNT],
            colors: [[0.0; 4]; crate::SHADER_COLOR_COUNT],
        });
        scene.finish();

        let labels: Vec<_> = scene.batches().map(|batch| batch.label()).collect();
        assert_eq!(
            labels,
            [
                "quads (1)",
                "backdrops (1)",
                "quads (1)",
                "shader 7 (1)",
                "shader 3 (1)"
            ]
        );
        assert!(scene.quads[0].order < scene.backdrops[0].order);
        assert_eq!(scene.quads[1].order, scene.backdrops[0].order);
        assert_eq!(scene.shaders[0].order, scene.backdrops[0].order);
        assert!(scene.shaders[1].order > scene.shaders[0].order);
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "a filtered layer opened inside a paint layer")]
    fn a_filter_cannot_open_inside_a_paint_layer() {
        let mut scene = Scene::default();
        scene.push_layer();
        scene.push_filter(
            rect(0.0, 0.0, 100.0, 100.0),
            origin(50.0, 50.0),
            rect(0.0, 0.0, 500.0, 500.0),
            Filter {
                blur: 4.0,
                ..Default::default()
            },
        );
    }
}

#[derive(Clone)]
#[expect(missing_docs)]
pub enum Primitive {
    Shadow(Shadow),
    Backdrop(Backdrop),
    Quad(Quad),
    Shader(ShaderQuad),
    Path(PaintedPath),
    Underline(Underline),
    MonochromeSprite(MonochromeSprite),
    SubpixelSprite(SubpixelSprite),
    PolychromeSprite(PolychromeSprite),
    Surface(PaintSurface),
}

#[expect(missing_docs)]
impl Primitive {
    pub(crate) fn kind(&self) -> PrimitiveKind {
        match self {
            Primitive::Shadow(_) => PrimitiveKind::Shadow,
            Primitive::Backdrop(_) => PrimitiveKind::Backdrop,
            Primitive::Quad(_) => PrimitiveKind::Quad,
            Primitive::Shader(_) => PrimitiveKind::Shader,
            Primitive::Path(_) => PrimitiveKind::Path,
            Primitive::Underline(_) => PrimitiveKind::Underline,
            Primitive::MonochromeSprite(_) => PrimitiveKind::MonochromeSprite,
            Primitive::SubpixelSprite(_) => PrimitiveKind::SubpixelSprite,
            Primitive::PolychromeSprite(_) => PrimitiveKind::PolychromeSprite,
            Primitive::Surface(_) => PrimitiveKind::Surface,
        }
    }

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
            Primitive::Path(path) => path.paint.color = path.paint.color.opacity(fade),
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

    fn transformation(&self) -> TransformationMatrix {
        match self {
            Primitive::MonochromeSprite(sprite) => sprite.transformation,
            Primitive::SubpixelSprite(sprite) => sprite.transformation,
            _ => TransformationMatrix::unit(),
        }
    }

    fn clipped_bounds(&self) -> Bounds<ScaledPixels> {
        self.transformation()
            .transform_bounds(*self.bounds())
            .intersect(&self.content_mask().bounds)
    }

    pub fn content_mask(&self) -> &ContentMask<ScaledPixels> {
        match self {
            Primitive::Shadow(shadow) => &shadow.content_mask,
            Primitive::Backdrop(backdrop) => &backdrop.content_mask,
            Primitive::Quad(quad) => &quad.content_mask,
            Primitive::Shader(shader) => &shader.content_mask,
            Primitive::Path(path) => &path.paint.content_mask,
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

    /// Stores the order the scene draws the primitive at.
    fn set_order(&mut self, order: DrawOrder) {
        match self {
            Primitive::Shadow(shadow) => shadow.order = order,
            Primitive::Backdrop(backdrop) => backdrop.order = order,
            Primitive::Quad(quad) => quad.order = order,
            Primitive::Shader(shader) => shader.order = order,
            Primitive::Path(path) => path.order = order,
            Primitive::Underline(underline) => underline.order = order,
            Primitive::MonochromeSprite(sprite) => sprite.order = order,
            Primitive::SubpixelSprite(sprite) => sprite.order = order,
            Primitive::PolychromeSprite(sprite) => sprite.order = order,
            Primitive::Surface(surface) => surface.order = order,
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
    paths_iter: Peekable<slice::Iter<'a, PaintedPath>>,
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
            (
                self.paths_iter.peek().map(|path| path.order),
                PrimitiveKind::Path,
            ),
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

#[derive(Default, Debug, Copy, Clone)]
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
    pub shape_bounds: Bounds<ScaledPixels>,
    pub shape_corner_radii: Corners<ScaledPixels>,
    pub inset: PaddedBool32,
    /// The corner smoothing of the element's own outline, which clips the shadow.
    pub element_corner_smoothing: f32,
    /// The element that clips the shadow: a drop shadow is drawn only outside it, an inset one
    /// only inside it.
    pub element_bounds: Bounds<ScaledPixels>,
    pub element_corner_radii: Corners<ScaledPixels>,
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

    /// The smallest bounds that contain `bounds` once it is transformed.
    pub(crate) fn transform_bounds(&self, bounds: Bounds<ScaledPixels>) -> Bounds<ScaledPixels> {
        if *self == Self::unit() {
            return bounds;
        }
        let [[xx, xy], [yx, yy]] = self.rotation_scale;
        let [translate_x, translate_y] = self.translation;
        let corners = [
            bounds.origin,
            bounds.top_right(),
            bounds.bottom_left(),
            bounds.bottom_right(),
        ]
        .map(|corner| {
            Point::new(
                ScaledPixels(xx * corner.x.0 + xy * corner.y.0 + translate_x),
                ScaledPixels(yx * corner.x.0 + yy * corner.y.0 + translate_y),
            )
        });
        let [first, rest @ ..] = corners;
        let top_left = rest.iter().fold(first, |min, corner| min.min(corner));
        let bottom_right = rest.iter().fold(first, |max, corner| max.max(corner));
        Bounds::from_corners(top_left, bottom_right)
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

#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(C)]
#[expect(missing_docs)]
pub struct PathCurve {
    pub p0: Point<f32>,
    pub p1: Point<f32>,
    pub a: Point<f32>,
    pub b: Point<f32>,
    pub sign: f32,
    pub pad: u32, // align to 8 bytes
}

#[derive(Clone, Copy, Debug)]
#[repr(C)]
#[expect(missing_docs)]
pub struct PathPaint {
    /// The path's bounds, in device pixels.
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    pub color: Background,
    pub even_odd: PaddedBool32,
    pub pad: u32, // align to 8 bytes
}

#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(C)]
#[expect(missing_docs)]
pub struct PathTile {
    pub paint: u32,
    pub curve_start: u32,
    pub curve_count: u32,
    pub backdrop: i32,
    pub top_left: Point<f32>,
    pub bottom_right: Point<f32>,
    pub reference: Point<f32>,
    pub curves_top_left: Point<f32>,
    pub curves_bottom_right: Point<f32>,
}

#[derive(Clone, Debug)]
#[expect(missing_docs)]
pub struct PaintedPath {
    order: DrawOrder,
    bounds: Bounds<ScaledPixels>,
    tiles: Rc<PathTiles>,
    paint: PathPaint,
}

#[expect(missing_docs)]
impl PaintedPath {
    /// Bins `path` into tiles, to be filled with `color` within `content_mask`.
    pub fn new(
        path: Path,
        scale_factor: f32,
        content_mask: ContentMask<Pixels>,
        color: Background,
    ) -> Self {
        let content_mask = content_mask.scale(scale_factor);
        let mut tiles = path.bin(scale_factor);
        tiles.retain_visible(content_mask.bounds);
        let paint = PathPaint {
            bounds: tiles.bounds(),
            content_mask,
            color,
            even_odd: path.even_odd().into(),
            pad: 0,
        };
        Self {
            order: 0,
            bounds: tiles.painted_bounds(),
            tiles: Rc::new(tiles),
            paint,
        }
    }

    pub fn tile_count(&self) -> usize {
        self.tiles.tiles().len()
    }

    pub fn append_to(
        &self,
        paints: &mut Vec<PathPaint>,
        curves: &mut Vec<PathCurve>,
        tiles: &mut Vec<PathTile>,
    ) {
        let paint_index = paints.len() as u32;
        let curve_start = curves.len() as u32;
        paints.push(self.paint);
        curves.extend_from_slice(self.tiles.curves());
        tiles.extend(self.tiles.tiles().iter().map(|tile| PathTile {
            paint: paint_index,
            curve_start: curve_start + tile.curve_start,
            ..*tile
        }));
    }
}

impl From<PaintedPath> for Primitive {
    fn from(path: PaintedPath) -> Self {
        Primitive::Path(path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{point, size};

    fn square(x: f32) -> Bounds<ScaledPixels> {
        Bounds {
            origin: point(ScaledPixels(x), ScaledPixels(0.)),
            size: size(ScaledPixels(10.), ScaledPixels(10.)),
        }
    }

    fn quad(x: f32) -> Quad {
        Quad {
            bounds: square(x),
            content_mask: ContentMask { bounds: square(x) },
            ..Default::default()
        }
    }

    fn underline(x: f32) -> Underline {
        Underline {
            order: 0,
            pad: 0,
            bounds: square(x),
            content_mask: ContentMask { bounds: square(x) },
            color: Hsla::default(),
            thickness: ScaledPixels(1.),
            wavy: false.into(),
        }
    }

    fn batch_labels(scene: &mut Scene) -> Vec<String> {
        scene.finish();
        scene.batches().map(|batch| batch.label()).collect()
    }

    #[test]
    fn layers_are_drawn_in_paint_order() {
        // A quad painted after a layer goes above it.
        let mut scene = Scene::default();
        scene.insert_primitive(quad(0.));
        scene.push_layer();
        scene.insert_primitive(underline(0.));
        scene.insert_primitive(underline(5.));
        scene.pop_layer();
        scene.insert_primitive(quad(5.));
        assert_eq!(
            batch_labels(&mut scene),
            ["quads (1)", "underlines (2)", "quads (1)"]
        );

        // A layer goes above the layer painted before it.
        let mut scene = Scene::default();
        scene.push_layer();
        scene.insert_primitive(underline(0.));
        scene.pop_layer();
        scene.push_layer();
        scene.insert_primitive(quad(5.));
        scene.pop_layer();
        assert_eq!(batch_labels(&mut scene), ["underlines (1)", "quads (1)"]);
    }

    #[test]
    fn nested_layers_are_drawn_above_their_parents() {
        // A layer has one draw order, so what it paints after a nested layer
        // still goes below that layer.
        let mut scene = Scene::default();
        scene.push_layer();
        scene.insert_primitive(underline(0.));
        scene.push_layer();
        scene.insert_primitive(quad(5.));
        scene.pop_layer();
        scene.insert_primitive(underline(5.));
        scene.pop_layer();
        assert_eq!(batch_labels(&mut scene), ["underlines (2)", "quads (1)"]);

        // The same holds when the nested layer shares its parent's order.
        let mut scene = Scene::default();
        scene.push_layer();
        scene.insert_primitive(quad(0.));
        scene.push_layer();
        scene.insert_primitive(quad(5.));
        scene.pop_layer();
        scene.insert_primitive(quad(8.));
        scene.pop_layer();
        assert_eq!(batch_labels(&mut scene), ["quads (3)"]);
        assert_eq!(
            scene
                .quads
                .iter()
                .map(|quad| quad.bounds)
                .collect::<Vec<_>>(),
            [square(0.), square(8.), square(5.)]
        );

        // A layer goes above the layers nested in the layer before it.
        let mut scene = Scene::default();
        scene.push_layer();
        scene.push_layer();
        scene.insert_primitive(underline(0.));
        scene.pop_layer();
        scene.pop_layer();
        scene.push_layer();
        scene.insert_primitive(quad(5.));
        scene.pop_layer();
        assert_eq!(batch_labels(&mut scene), ["underlines (1)", "quads (1)"]);
    }
}
