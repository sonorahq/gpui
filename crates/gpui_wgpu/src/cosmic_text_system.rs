use anyhow::{Context as _, Ok, Result};
use collections::HashMap;
use cosmic_text::{
    Attrs, AttrsList, Ellipsize, Family, Font as CosmicTextFont,
    FontFeatures as CosmicFontFeatures, FontSystem, ShapeBuffer, ShapeLine, Weight as CosmicWeight,
};
use gpui::{
    Bounds, DevicePixels, Font, FontCacheConfig, FontFallbacks, FontFeatures, FontId, FontMetrics,
    FontRun, GlyphId, LineLayout, Pixels, PlatformTextSystem, RenderGlyphParams, SUBPIXEL_VARIANTS_X,
    SUBPIXEL_VARIANTS_Y, ShapedGlyph, ShapedRun, SharedString, Size, TextRenderingMode, point,
    size,
};

use itertools::Itertools;
use parking_lot::RwLock;
use smallvec::SmallVec;
use std::{
    borrow::Cow,
    ops::Range,
    sync::Arc,
    time::{Duration, Instant},
};
use swash::{
    NormalizedCoord, Tag,
    scale::{Render, ScaleContext, Source, StrikeWith},
    tag_from_bytes,
    zeno::{Format, Vector},
};
use unicode_segmentation::UnicodeSegmentation;

pub struct CosmicTextSystem(RwLock<CosmicTextSystemState>);

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct FontKey {
    family: SharedString,
    features: FontFeatures,
    fallbacks: Option<FontFallbacks>,
    weight: CosmicWeight,
    style: gpui::FontStyle,
}

impl FontKey {
    fn new(
        family: SharedString,
        features: FontFeatures,
        fallbacks: Option<FontFallbacks>,
        weight: CosmicWeight,
        style: gpui::FontStyle,
    ) -> Self {
        Self {
            family,
            features,
            fallbacks,
            weight,
            style,
        }
    }
}

struct CosmicTextSystemState {
    font_system: FontSystem,
    scratch: ShapeBuffer,
    swash_scale_context: ScaleContext,
    /// Contains fonts that have been selected or requested by shaping. Indexed by `FontId`.
    loaded_fonts: Vec<LoadedFont>,
    /// Caches the `FontId`s associated with a specific family to avoid iterating the font database
    /// for every font face in a family.
    font_ids_by_family_cache: HashMap<FontKey, SmallVec<[FontId; 4]>>,
    /// Cosmic can resolve the same fallback face at several variable weights.
    /// Cache those resolutions so shaping does not linearly scan loaded fonts.
    font_ids_by_cosmic_id: HashMap<(cosmic_text::fontdb::ID, u16), FontId>,
    system_font_fallback: String,
    cache_config: FontCacheConfig,
    last_eviction_check: Instant,
}

struct LoadedFont {
    database_id: cosmic_text::fontdb::ID,
    font: Option<Arc<CosmicTextFont>>,
    cached_metrics: FontMetrics,
    features: CosmicFontFeatures,
    has_color_glyphs: bool,
    instantiated_weight: CosmicWeight,
    /// empty when not variable
    wght_coords: SmallVec<[NormalizedCoord; 4]>,
    /// resolved at load time so `layout_line` shares one chain across faces.
    /// `Arc` keeps clone cheap on the per-run hot path.
    user_fallback_chain: Arc<[(FontId, SharedString)]>,
    last_accessed: Instant,
    is_protected: bool,
}

const WGHT: Tag = tag_from_bytes(b"wght");

fn cosmic_weight(weight: gpui::FontWeight) -> CosmicWeight {
    CosmicWeight(weight.0.round().clamp(1.0, 1000.0) as u16)
}

fn compute_font_metrics(font: &CosmicTextFont, coords: &[NormalizedCoord]) -> FontMetrics {
    let metrics = font.as_swash().metrics(coords);
    FontMetrics {
        units_per_em: metrics.units_per_em as u32,
        ascent: metrics.ascent,
        descent: -metrics.descent,
        line_gap: metrics.leading,
        underline_position: metrics.underline_offset,
        underline_thickness: metrics.stroke_size,
        cap_height: metrics.cap_height,
        x_height: metrics.x_height,
        bounding_box: Bounds {
            origin: point(0.0, 0.0),
            size: size(metrics.max_width, metrics.ascent + metrics.descent),
        },
    }
}

// none when not variable
fn wght_instance(
    font: &CosmicTextFont,
    weight: CosmicWeight,
) -> Option<(CosmicWeight, SmallVec<[NormalizedCoord; 4]>)> {
    let variations = font.as_swash().variations();
    let axis = variations.find_by_tag(WGHT)?;
    let value = f32::from(weight.0).clamp(axis.min_value(), axis.max_value());
    let coords = variations.normalized_coords([(WGHT, value)]).collect();
    Some((CosmicWeight(value.round() as u16), coords))
}

impl CosmicTextSystem {
    pub fn new(system_font_fallback: &str) -> Self {
        let font_system = FontSystem::new();

        Self(RwLock::new(CosmicTextSystemState {
            font_system,
            scratch: ShapeBuffer::default(),
            swash_scale_context: ScaleContext::new(),
            loaded_fonts: Vec::new(),
            font_ids_by_family_cache: HashMap::default(),
            font_ids_by_cosmic_id: HashMap::default(),
            system_font_fallback: system_font_fallback.to_string(),
            cache_config: FontCacheConfig::default(),
            last_eviction_check: Instant::now(),
        }))
    }

    pub fn new_without_system_fonts(system_font_fallback: &str) -> Self {
        let font_system = FontSystem::new_with_locale_and_db(
            "en-US".to_string(),
            cosmic_text::fontdb::Database::new(),
        );

        Self(RwLock::new(CosmicTextSystemState {
            font_system,
            scratch: ShapeBuffer::default(),
            swash_scale_context: ScaleContext::new(),
            loaded_fonts: Vec::new(),
            font_ids_by_family_cache: HashMap::default(),
            font_ids_by_cosmic_id: HashMap::default(),
            system_font_fallback: system_font_fallback.to_string(),
            cache_config: FontCacheConfig::default(),
            last_eviction_check: Instant::now(),
        }))
    }

    pub fn set_cache_config(&self, config: FontCacheConfig) {
        self.0.write().cache_config = config;
    }

    pub fn evict_unused_fonts(&self, older_than: Option<Duration>) -> usize {
        self.0.write().evict_unused_fonts(older_than)
    }

    pub fn unload_font(&self, font_id: FontId) -> bool {
        self.0.write().unload_font(font_id)
    }

    pub fn active_font_count(&self) -> usize {
        self.0.read().active_font_count()
    }

    pub fn clear_font_cache(&self) -> usize {
        self.0.write().clear_font_cache()
    }
}

impl PlatformTextSystem for CosmicTextSystem {
    fn add_fonts(&self, fonts: Vec<Cow<'static, [u8]>>) -> Result<()> {
        self.0.write().add_fonts(fonts)
    }

    fn all_font_names(&self) -> Vec<String> {
        let mut result = self
            .0
            .read()
            .font_system
            .db()
            .faces()
            .filter_map(|face| face.families.first().map(|family| family.0.clone()))
            .collect_vec();
        result.sort_unstable();
        result.dedup();
        result
    }

    fn font_id(&self, font: &Font) -> Result<FontId> {
        let mut state = self.0.write();
        let weight = cosmic_weight(font.weight);
        let key = FontKey::new(
            font.family.clone(),
            font.features.clone(),
            font.fallbacks.clone(),
            weight,
            font.style,
        );
        let candidates = if let Some(font_ids) = state.font_ids_by_family_cache.get(&key) {
            font_ids.as_slice()
        } else {
            let font_ids = state.load_family(
                &font.family,
                &font.features,
                font.fallbacks.as_ref(),
                weight,
                font.style,
            )?;
            state.font_ids_by_family_cache.insert(key.clone(), font_ids);
            state.font_ids_by_family_cache[&key].as_ref()
        };

        let ix = find_best_match(font, candidates, &state)?;
        let font_id = candidates[ix];
        state.ensure_font_loaded(font_id)?;

        Ok(font_id)
    }

    fn font_metrics(&self, font_id: FontId) -> FontMetrics {
        let lock = self.0.read();
        lock.loaded_fonts[font_id.0].cached_metrics
    }

    fn typographic_bounds(&self, font_id: FontId, glyph_id: GlyphId) -> Result<Bounds<f32>> {
        self.0.write().typographic_bounds(font_id, glyph_id)
    }

    fn advance(&self, font_id: FontId, glyph_id: GlyphId) -> Result<Size<f32>> {
        self.0.write().advance(font_id, glyph_id)
    }

    fn glyph_for_char(&self, font_id: FontId, ch: char) -> Option<GlyphId> {
        self.0.write().glyph_for_char(font_id, ch)
    }

    fn glyph_raster_bounds(&self, params: &RenderGlyphParams) -> Result<Bounds<DevicePixels>> {
        self.0.write().raster_bounds(params)
    }

    fn rasterize_glyph(
        &self,
        params: &RenderGlyphParams,
        raster_bounds: Bounds<DevicePixels>,
    ) -> Result<(Size<DevicePixels>, Vec<u8>)> {
        self.0.write().rasterize_glyph(params, raster_bounds)
    }

    fn layout_line(&self, text: &str, font_size: Pixels, runs: &[FontRun]) -> LineLayout {
        self.0.write().layout_line(text, font_size, runs)
    }

    fn recommended_rendering_mode(
        &self,
        _font_id: FontId,
        _font_size: Pixels,
    ) -> TextRenderingMode {
        TextRenderingMode::Subpixel
    }

    fn set_font_cache_config(&self, config: FontCacheConfig) {
        self.0.write().cache_config = config;
    }

    fn evict_unused_fonts(&self, older_than: Option<Duration>) -> Result<usize> {
        Ok(self.0.write().evict_unused_fonts(older_than))
    }

    fn unload_font(&self, font_id: FontId) -> Result<bool> {
        Ok(self.0.write().unload_font(font_id))
    }

    fn active_font_count(&self) -> usize {
        self.0.read().active_font_count()
    }

    fn clear_font_cache(&self) -> Result<usize> {
        Ok(self.0.write().clear_font_cache())
    }
}

impl CosmicTextSystemState {
    fn loaded_font(&self, font_id: FontId) -> &LoadedFont {
        &self.loaded_fonts[font_id.0]
    }

    #[profiling::function]
    fn add_fonts(&mut self, fonts: Vec<Cow<'static, [u8]>>) -> Result<()> {
        {
            let db = self.font_system.db_mut();
            for bytes in fonts {
                db.load_font_source(cosmic_text::fontdb::Source::Binary(Arc::new(bytes)));
            }
        }
        // Adding a font can change the best face for a family that has already
        // been queried. Keep the cache from returning the old candidate set.
        self.font_ids_by_family_cache.clear();
        Ok(())
    }

    #[profiling::function]
    fn load_family(
        &mut self,
        name: &str,
        features: &FontFeatures,
        fallbacks: Option<&FontFallbacks>,
        weight: CosmicWeight,
        style: gpui::FontStyle,
    ) -> Result<SmallVec<[FontId; 4]>> {
        // recurse with `fallbacks = None` so a fallback family cannot pull in
        // another chain. missing fallback families are dropped so a typo in
        // settings still lets the primary family load.
        let user_fallback_chain: Arc<[(FontId, SharedString)]> = match fallbacks {
            Some(fallbacks) if !fallbacks.fallback_list().is_empty() => {
                let mut chain: Vec<(FontId, SharedString)> = Vec::new();
                for fallback_name in fallbacks.fallback_list() {
                    let fb_key = FontKey::new(
                        SharedString::from(fallback_name.clone()),
                        features.clone(),
                        None,
                        weight,
                        gpui::FontStyle::Normal,
                    );
                    let fb_ids = if let Some(cached) = self.font_ids_by_family_cache.get(&fb_key) {
                        cached.clone()
                    } else {
                        let loaded = self.load_family(
                            fallback_name,
                            features,
                            None,
                            weight,
                            gpui::FontStyle::Normal,
                        )?;
                        self.font_ids_by_family_cache
                            .insert(fb_key.clone(), loaded.clone());
                        loaded
                    };
                    // a family can hold several static weights, so match the requested
                    // one rather than taking whichever face the database lists first.
                    let wanted = Font {
                        family: SharedString::from(fallback_name.clone()),
                        features: features.clone(),
                        fallbacks: None,
                        weight: gpui::FontWeight(f32::from(weight.0)),
                        style: gpui::FontStyle::Normal,
                    };
                    let Some(&fb_id) = find_best_match(&wanted, &fb_ids, self)
                        .ok()
                        .and_then(|place| fb_ids.get(place))
                        .or_else(|| fb_ids.first())
                    else {
                        continue;
                    };
                    let db_id = self.loaded_fonts[fb_id.0].database_id;
                    if let Some(face) = self.font_system.db().face(db_id)
                        && let Some(family) = face.families.first()
                    {
                        chain.push((fb_id, SharedString::from(family.0.clone())));
                    }
                }
                Arc::from(chain)
            }
            _ => Arc::from(Vec::new()),
        };

        let name = gpui::font_name_with_fallbacks(name, &self.system_font_fallback);

        let mut families = self
            .font_system
            .db()
            .faces()
            .filter(|face| face.families.iter().any(|family| *name == family.0))
            .map(|face| face.id)
            .collect::<SmallVec<[_; 4]>>();

        let cosmic_features = cosmic_font_features(features)?;
        let wanted = Font {
            family: SharedString::from(name),
            features: features.clone(),
            fallbacks: None,
            weight: gpui::FontWeight(f32::from(weight.0)),
            style,
        };

        let mut loaded_font_ids = SmallVec::new();
        while !families.is_empty() {
            // Select the best face from database metadata before asking
            // cosmic-text to instantiate it. Loading every face in a family
            // is particularly expensive on Linux system font databases.
            let face_index = find_best_face(&wanted, &families, self)?;
            let database_id = families.remove(face_index);
            let face = self
                .font_system
                .db()
                .face(database_id)
                .context("font face not found in database")?;
            let postscript_name = face.post_script_name.clone();
            let face_weight = face.weight;

            // probes the wght axis
            let default_instance = self
                .font_system
                .get_font(database_id, CosmicWeight::NORMAL)
                .context("Could not load font")?;

            // HACK: To let the storybook run and render Windows caption icons. We should actually do better font fallback.
            let allowed_bad_font_names = [
                "SegoeFluentIcons", // NOTE: Segoe fluent icons postscript name is inconsistent
                "Segoe Fluent Icons",
            ];

            let has_color_glyphs = font_has_color_glyphs(&default_instance);

            if default_instance.as_swash().charmap().map('m') == 0
                && !has_color_glyphs
                && !allowed_bad_font_names.contains(&postscript_name.as_str())
            {
                self.font_system.db_mut().remove_face(default_instance.id());
                continue;
            };

            let (font, instantiated_weight, wght_coords) =
                match wght_instance(&default_instance, weight) {
                    Some((clamped, coords)) => {
                        let font = self
                            .font_system
                            .get_font(database_id, clamped)
                            .context("Could not load variable font instance")?;
                        (font, clamped, coords)
                    }
                    None => (default_instance, face_weight, SmallVec::new()),
                };

            let cached_metrics = compute_font_metrics(&font, &wght_coords);
            let font_id = FontId(self.loaded_fonts.len());
            loaded_font_ids.push(font_id);
            let cache_weight = if wght_coords.is_empty() {
                0
            } else {
                instantiated_weight.0
            };
            let is_protected = self.loaded_fonts.is_empty()
                || name == self.system_font_fallback
                || name == ".SystemUIFont";
            self.loaded_fonts.push(LoadedFont {
                database_id,
                font: Some(font),
                cached_metrics,
                features: cosmic_features.clone(),
                has_color_glyphs,
                instantiated_weight,
                wght_coords,
                user_fallback_chain: Arc::clone(&user_fallback_chain),
                last_accessed: Instant::now(),
                is_protected,
            });
            self.font_ids_by_cosmic_id
                .insert((database_id, cache_weight), font_id);
            if cache_weight != 0 {
                self.font_ids_by_cosmic_id
                    .insert((database_id, weight.0), font_id);
            }
            break;
        }

        self.maybe_evict();
        Ok(loaded_font_ids)
    }

    fn advance(&mut self, font_id: FontId, glyph_id: GlyphId) -> Result<Size<f32>> {
        let font = self.ensure_font_loaded(font_id)?;
        let glyph_metrics = font
            .as_swash()
            .glyph_metrics(&self.loaded_fonts[font_id.0].wght_coords);
        Ok(Size {
            width: glyph_metrics.advance_width(glyph_id.0 as u16),
            height: glyph_metrics.advance_height(glyph_id.0 as u16),
        })
    }

    fn glyph_for_char(&mut self, font_id: FontId, ch: char) -> Option<GlyphId> {
        let font = self.ensure_font_loaded(font_id).ok()?;
        let glyph_id = font.as_swash().charmap().map(ch);
        if glyph_id == 0 {
            None
        } else {
            Some(GlyphId(glyph_id.into()))
        }
    }

    fn typographic_bounds(&mut self, font_id: FontId, glyph_id: GlyphId) -> Result<Bounds<f32>> {
        let font = self.ensure_font_loaded(font_id)?;
        let loaded_font = &self.loaded_fonts[font_id.0];
        let glyph_metrics = font
            .as_swash()
            .glyph_metrics(&loaded_font.wght_coords);
        let glyph_id = glyph_id.0 as u16;
        Ok(Bounds {
            origin: point(0.0, 0.0),
            size: size(
                glyph_metrics.advance_width(glyph_id),
                glyph_metrics.advance_height(glyph_id),
            ),
        })
    }

    fn raster_bounds(&mut self, params: &RenderGlyphParams) -> Result<Bounds<DevicePixels>> {
        let image = self.render_glyph_image(params)?;
        Ok(Bounds {
            origin: point(image.placement.left.into(), (-image.placement.top).into()),
            size: size(image.placement.width.into(), image.placement.height.into()),
        })
    }

    #[profiling::function]
    fn rasterize_glyph(
        &mut self,
        params: &RenderGlyphParams,
        glyph_bounds: Bounds<DevicePixels>,
    ) -> Result<(Size<DevicePixels>, Vec<u8>)> {
        if glyph_bounds.size.width.0 == 0 || glyph_bounds.size.height.0 == 0 {
            anyhow::bail!("glyph bounds are empty");
        }

        let mut image = self.render_glyph_image(params)?;
        let bitmap_size = glyph_bounds.size;
        match image.content {
            swash::scale::image::Content::Color | swash::scale::image::Content::SubpixelMask => {
                // Convert from RGBA to BGRA.
                for pixel in image.data.chunks_exact_mut(4) {
                    pixel.swap(0, 2);
                }
                Ok((bitmap_size, image.data))
            }
            swash::scale::image::Content::Mask => {
                if params.subpixel_rendering {
                    // We must always return RGBA data when subpixel rendering is requested.
                    let expanded = image.data.iter().flat_map(|&a| [a, a, a, a]).collect();
                    Ok((bitmap_size, expanded))
                } else {
                    Ok((bitmap_size, image.data))
                }
            }
        }
    }

    fn render_glyph_image(
        &mut self,
        params: &RenderGlyphParams,
    ) -> Result<swash::scale::image::Image> {
        let font = self.ensure_font_loaded(params.font_id)?;
        let loaded_font = &self.loaded_fonts[params.font_id.0];
        let font_ref = font.as_swash();
        let pixel_size = f32::from(params.font_size);

        let subpixel_offset = Vector::new(
            params.subpixel_variant.x as f32 / SUBPIXEL_VARIANTS_X as f32 / params.scale_factor,
            params.subpixel_variant.y as f32 / SUBPIXEL_VARIANTS_Y as f32 / params.scale_factor,
        );

        // clears stale coords
        let mut scaler = self
            .swash_scale_context
            .builder(font_ref)
            .size(pixel_size * params.scale_factor)
            .hint(true)
            .normalized_coords(&loaded_font.wght_coords)
            .build();

        let sources: &[Source] = if params.is_emoji {
            &[
                Source::ColorOutline(0),
                Source::ColorBitmap(StrikeWith::BestFit),
                Source::Outline,
            ]
        } else {
            &[Source::Bitmap(StrikeWith::ExactSize), Source::Outline]
        };

        let mut renderer = Render::new(sources);
        if params.subpixel_rendering {
            // There seems to be a bug in Swash where the B and R values are swapped.
            renderer
                .format(Format::subpixel_bgra())
                .offset(subpixel_offset);
        } else {
            renderer.format(Format::Alpha).offset(subpixel_offset);
        }

        let glyph_id: u16 = params.glyph_id.0.try_into()?;
        renderer
            .render(&mut scaler, glyph_id)
            .with_context(|| format!("unable to render glyph via swash for {params:?}"))
    }

    /// This is used when cosmic_text has chosen a fallback font instead of using the requested
    /// font, typically to handle some unicode characters. When this happens, `loaded_fonts` may not
    /// yet have an entry for this fallback font, and so one is added.
    ///
    /// Note that callers shouldn't use this `FontId` somewhere that will retrieve the corresponding
    /// `LoadedFont.features`, as it will have an arbitrarily chosen or empty value. The only
    /// current use of this field is for the *input* of `layout_line`, and so it's fine to use
    /// `font_id_for_cosmic_id` when computing the *output* of `layout_line`.
    /// `weight` is the weight the glyph was shaped at, and is part of the identity
    /// of the entry: a variable face pinned to two weights needs two `FontId`s so
    /// rasterization picks up the matching `wght` coordinates.
    fn font_id_for_cosmic_id(
        &mut self,
        id: cosmic_text::fontdb::ID,
        weight: CosmicWeight,
    ) -> Result<FontId> {
        // Static faces use weight zero as a wildcard; variable faces are keyed
        // by the clamped weight at which they were instantiated.
        if let Some(&font_id) = self
            .font_ids_by_cosmic_id
            .get(&(id, weight.0))
            .or_else(|| self.font_ids_by_cosmic_id.get(&(id, 0)))
        {
            self.ensure_font_loaded(font_id)?;
            return Ok(font_id);
        }

        let default_instance = self
            .font_system
            .get_font(id, CosmicWeight::NORMAL)
            .context("failed to get fallback font from cosmic-text font system")?;
        let face = self
            .font_system
            .db()
            .face(id)
            .context("fallback font face not found in cosmic-text database")?;
        let face_weight = face.weight;
        let has_color_glyphs = font_has_color_glyphs(&default_instance);

            let (font, instantiated_weight, wght_coords) =
                match wght_instance(&default_instance, weight) {
                    Some((clamped, coords)) => {
                        let font = self
                            .font_system
                            .get_font(id, clamped)
                            .context("failed to get variable fallback font instance")?;
                        (font, clamped, coords)
                    }
                    None => (default_instance, face_weight, SmallVec::new()),
                };

        let cached_metrics = compute_font_metrics(&font, &wght_coords);
        let font_id = FontId(self.loaded_fonts.len());
        let cache_weight = if wght_coords.is_empty() {
            0
        } else {
            instantiated_weight.0
        };
        self.loaded_fonts.push(LoadedFont {
            database_id: id,
            font: Some(font),
            cached_metrics,
            features: CosmicFontFeatures::new(),
            has_color_glyphs,
            instantiated_weight,
            wght_coords,
            user_fallback_chain: Arc::from(Vec::new()),
            last_accessed: Instant::now(),
            is_protected: false,
        });
        self.font_ids_by_cosmic_id
            .insert((id, cache_weight), font_id);
        if cache_weight != 0 {
            self.font_ids_by_cosmic_id.insert((id, weight.0), font_id);
        }

        self.maybe_evict();
        Ok(font_id)
    }

    #[profiling::function]
    fn layout_line(&mut self, text: &str, font_size: Pixels, font_runs: &[FontRun]) -> LineLayout {
        let layout = if contains_paragraph_separator(text) {
            self.layout_line_with_separators(text, font_size, font_runs)
        } else {
            self.layout_line_no_separators(text, font_size, font_runs)
        };
        self.maybe_evict();
        layout
    }

    fn layout_line_with_separators(
        &mut self,
        text: &str,
        font_size: Pixels,
        font_runs: &[FontRun],
    ) -> LineLayout {
        let mut layout = LineLayout {
            font_size,
            len: text.len(),
            ..Default::default()
        };
        let mut paragraph_start = 0;

        for (separator_start, separator) in text
            .char_indices()
            .filter(|(_, character)| is_paragraph_separator(*character))
        {
            let separator_end = separator_start + separator.len_utf8();
            self.shape_segment(
                text,
                paragraph_start..separator_start,
                font_size,
                font_runs,
                &mut layout,
            );
            self.shape_segment(
                text,
                separator_start..separator_end,
                font_size,
                font_runs,
                &mut layout,
            );
            paragraph_start = separator_end;
        }

        self.shape_segment(
            text,
            paragraph_start..text.len(),
            font_size,
            font_runs,
            &mut layout,
        );

        layout
    }

    fn shape_segment(
        &mut self,
        text: &str,
        range: Range<usize>,
        font_size: Pixels,
        font_runs: &[FontRun],
        layout: &mut LineLayout,
    ) {
        if range.is_empty() {
            return;
        }

        let segment_font_runs = clip_font_runs(font_runs, range.clone());
        let segment =
            self.layout_line_no_separators(&text[range.clone()], font_size, &segment_font_runs);

        let mut segment_runs = segment.runs;
        for run in &mut segment_runs {
            for glyph in &mut run.glyphs {
                glyph.index += range.start;
                glyph.position.x += layout.width;
            }
        }

        for mut run in segment_runs {
            if let Some(same_run) = layout
                .runs
                .last_mut()
                .filter(|last| last.font_id == run.font_id)
            {
                same_run.glyphs.append(&mut run.glyphs);
            } else {
                layout.runs.push(run);
            }
        }

        layout.width += segment.width;
        layout.ascent = layout.ascent.max(segment.ascent);
        layout.descent = layout.descent.max(segment.descent);
    }

    fn layout_line_no_separators(
        &mut self,
        text: &str,
        font_size: Pixels,
        font_runs: &[FontRun],
    ) -> LineLayout {
        for run in font_runs {
            self.ensure_font_loaded(run.font_id).ok();
            if let Some(loaded_font) = self.loaded_fonts.get(run.font_id.0) {
                let chain = Arc::clone(&loaded_font.user_fallback_chain);
                for (fb_id, _) in chain.iter() {
                    self.ensure_font_loaded(*fb_id).ok();
                }
            }
        }

        let mut attrs_list = AttrsList::new(&Attrs::new());
        let mut offs = 0;
        for run in font_runs {
            let run_end = offs + run.len;

            let loaded_font = self.loaded_font(run.font_id);
            let Some(face) = self.font_system.db().face(loaded_font.database_id) else {
                log::warn!(
                    "font face not found in database for font_id {:?}",
                    run.font_id
                );
                offs = run_end;
                continue;
            };
            let Some(first_family) = face.families.first() else {
                log::warn!(
                    "font face has no family names for font_id {:?}",
                    run.font_id
                );
                offs = run_end;
                continue;
            };

            let primary_family_name: SharedString = first_family.0.clone().into();
            let primary_stretch = face.stretch;
            let primary_style = face.style;
            // pinned instance weight
            let primary_weight = loaded_font.instantiated_weight;
            let primary_features = loaded_font.features.clone();
            let fallback_chain = Arc::clone(&loaded_font.user_fallback_chain);

            // build one `Attrs` per slot up front. each clone of span attrs
            // would otherwise re-allocate the `font_features` Vec.
            let primary_attrs = Attrs::new()
                .metadata(run.font_id.0)
                .family(Family::Name(&primary_family_name))
                .stretch(primary_stretch)
                .style(primary_style)
                .weight(primary_weight)
                .font_features(primary_features.clone());
            let fallback_attrs: SmallVec<[Attrs<'_>; 4]> = fallback_chain
                .iter()
                .map(|(fb_id, fb_name)| {
                    Attrs::new()
                        .metadata(fb_id.0)
                        .family(Family::Name(fb_name))
                        .stretch(primary_stretch)
                        .style(primary_style)
                        .weight(primary_weight)
                        .font_features(primary_features.clone())
                })
                .collect();

            let spans = if fallback_chain.is_empty() {
                let mut spans = SmallVec::<[RunSpan; 4]>::new();
                spans.push(RunSpan {
                    start: offs,
                    end: run_end,
                    slot: None,
                    font_id: run.font_id,
                });
                spans
            } else {
                let loaded_fonts = &self.loaded_fonts;
                let covers = |id: FontId, ch: char| charmap_covers(loaded_fonts, id, ch);
                compute_run_spans(text, offs, run.len, run.font_id, &fallback_chain, &covers)
            };

            for span in spans {
                let attrs = match span.slot {
                    None => &primary_attrs,
                    Some(ix) => &fallback_attrs[ix],
                };
                attrs_list.add_span(span.start..span.end, attrs);
            }
            offs = run_end;
        }

        let line = ShapeLine::new(
            &mut self.font_system,
            text,
            &attrs_list,
            cosmic_text::Shaping::Advanced,
            4,
        );
        let mut layout_lines = Vec::with_capacity(1);
        line.layout_to_buffer(
            &mut self.scratch,
            f32::from(font_size),
            None, // We do our own wrapping
            cosmic_text::Wrap::None,
            Ellipsize::None,
            None,
            &mut layout_lines,
            None,
            cosmic_text::Hinting::Disabled,
        );

        let Some(layout) = layout_lines.first() else {
            return LineLayout {
                font_size,
                width: Pixels::ZERO,
                ascent: Pixels::ZERO,
                descent: Pixels::ZERO,
                runs: Vec::new(),
                len: text.len(),
            };
        };

        let mut runs: Vec<ShapedRun> = Vec::new();
        for glyph in &layout.glyphs {
            let mut font_id = FontId(glyph.metadata);
            self.ensure_font_loaded(font_id).ok();
            let mut loaded_font = self.loaded_font(font_id);
            if loaded_font.database_id != glyph.font_id {
                match self.font_id_for_cosmic_id(glyph.font_id, glyph.font_weight) {
                    std::result::Result::Ok(resolved_id) => {
                        font_id = resolved_id;
                        loaded_font = self.loaded_font(font_id);
                    }
                    Err(error) => {
                        log::warn!(
                            "failed to resolve cosmic font id {:?}: {error:#}",
                            glyph.font_id
                        );
                        continue;
                    }
                }
            }
            let is_emoji = loaded_font.has_color_glyphs;

            // HACK: Prevent crash caused by variation selectors.
            if glyph.glyph_id == 3 && is_emoji {
                continue;
            }

            let shaped_glyph = ShapedGlyph {
                id: GlyphId(glyph.glyph_id as u32),
                position: point(glyph.x.into(), glyph.y.into()),
                index: glyph.start,
                is_emoji,
            };

            if let Some(last_run) = runs
                .last_mut()
                .filter(|last_run| last_run.font_id == font_id)
            {
                last_run.glyphs.push(shaped_glyph);
            } else {
                runs.push(ShapedRun {
                    font_id,
                    glyphs: vec![shaped_glyph],
                });
            }
        }

        LineLayout {
            font_size,
            width: layout.w.into(),
            ascent: layout.max_ascent.into(),
            descent: layout.max_descent.into(),
            runs,
            len: text.len(),
        }
    }

    fn ensure_font_loaded(&mut self, font_id: FontId) -> Result<Arc<CosmicTextFont>> {
        let loaded = self
            .loaded_fonts
            .get_mut(font_id.0)
            .context("invalid font_id")?;
        loaded.last_accessed = Instant::now();
        if let Some(font) = &loaded.font {
            return Ok(Arc::clone(font));
        }

        let database_id = loaded.database_id;
        let instantiated_weight = loaded.instantiated_weight;
        let is_variable = !loaded.wght_coords.is_empty();

        let font = if is_variable {
            self.font_system
                .get_font(database_id, instantiated_weight)
                .context("could not reload variable font instance")?
        } else {
            self.font_system
                .get_font(database_id, CosmicWeight::NORMAL)
                .or_else(|| self.font_system.get_font(database_id, instantiated_weight))
                .context("could not reload font")?
        };

        let loaded = &mut self.loaded_fonts[font_id.0];
        loaded.font = Some(Arc::clone(&font));
        Ok(font)
    }

    fn active_font_count(&self) -> usize {
        self.loaded_fonts.iter().filter(|f| f.font.is_some()).count()
    }

    fn evict_unused_fonts(&mut self, older_than: Option<Duration>) -> usize {
        let now = Instant::now();
        let ttl = older_than.unwrap_or(self.cache_config.ttl);

        let candidates_to_evict: SmallVec<[FontId; 16]> = self
            .loaded_fonts
            .iter()
            .enumerate()
            .filter(|(_, font)| {
                font.font.is_some()
                    && !font.is_protected
                    && now.saturating_duration_since(font.last_accessed) >= ttl
            })
            .map(|(idx, _)| FontId(idx))
            .collect();

        if candidates_to_evict.is_empty() {
            return 0;
        }

        self.evict_fonts(&candidates_to_evict)
    }

    fn unload_font(&mut self, font_id: FontId) -> bool {
        if let Some(loaded) = self.loaded_fonts.get(font_id.0) {
            if loaded.font.is_some() {
                return self.evict_fonts(&[font_id]) > 0;
            }
        }
        false
    }

    fn clear_font_cache(&mut self) -> usize {
        let all_loaded: SmallVec<[FontId; 16]> = self
            .loaded_fonts
            .iter()
            .enumerate()
            .filter(|(_, font)| font.font.is_some())
            .map(|(idx, _)| FontId(idx))
            .collect();
        self.evict_fonts(&all_loaded)
    }

    fn maybe_evict(&mut self) {
        let now = Instant::now();
        if now.saturating_duration_since(self.last_eviction_check) < Duration::from_millis(500)
            && self.active_font_count() <= self.cache_config.max_loaded_fonts
        {
            return;
        }
        self.last_eviction_check = now;

        let active_count = self.active_font_count();

        // 1. Evict any non-protected fonts older than TTL
        let mut to_evict = SmallVec::<[FontId; 16]>::new();
        for (index, font) in self.loaded_fonts.iter().enumerate() {
            if font.font.is_some() && !font.is_protected {
                if now.saturating_duration_since(font.last_accessed) >= self.cache_config.ttl {
                    to_evict.push(FontId(index));
                }
            }
        }

        // 2. If still exceeding max_loaded_fonts, evict oldest active fonts (LRU)
        let remaining_active = active_count.saturating_sub(to_evict.len());
        if remaining_active > self.cache_config.max_loaded_fonts {
            let mut lru_candidates: Vec<(FontId, Instant)> = self
                .loaded_fonts
                .iter()
                .enumerate()
                .filter(|(idx, f)| {
                    f.font.is_some()
                        && !f.is_protected
                        && !to_evict.contains(&FontId(*idx))
                })
                .map(|(idx, f)| (FontId(idx), f.last_accessed))
                .collect();

            lru_candidates.sort_by_key(|(_, last_accessed)| *last_accessed);

            let excess = remaining_active - self.cache_config.max_loaded_fonts;
            for (id, _) in lru_candidates.into_iter().take(excess) {
                to_evict.push(id);
            }
        }

        if !to_evict.is_empty() {
            self.evict_fonts(&to_evict);
        }
    }

    fn evict_fonts(&mut self, font_ids: &[FontId]) -> usize {
        let mut evicted_count = 0;
        let mut evicted_db_ids = SmallVec::<[cosmic_text::fontdb::ID; 8]>::new();

        for &id in font_ids {
            if let Some(loaded) = self.loaded_fonts.get_mut(id.0) {
                if loaded.font.take().is_some() {
                    evicted_count += 1;
                    evicted_db_ids.push(loaded.database_id);
                }
            }
        }

        if evicted_count == 0 {
            return 0;
        }

        self.reset_font_system_cache(&evicted_db_ids);
        evicted_count
    }

    fn reset_font_system_cache(&mut self, evicted_db_ids: &[cosmic_text::fontdb::ID]) {
        let dummy = FontSystem::new_with_locale_and_db(
            "en-US".to_string(),
            cosmic_text::fontdb::Database::new(),
        );
        let (locale, mut db) = std::mem::replace(&mut self.font_system, dummy).into_locale_and_db();

        let active_db_ids: SmallVec<[cosmic_text::fontdb::ID; 16]> = self
            .loaded_fonts
            .iter()
            .filter_map(|f| if f.font.is_some() { Some(f.database_id) } else { None })
            .collect();

        for &db_id in evicted_db_ids {
            if !active_db_ids.contains(&db_id) {
                db.make_face_data_unshared(db_id);
            }
        }

        self.font_system = FontSystem::new_with_locale_and_db(locale, db);
    }
}

#[inline(always)]
fn is_paragraph_separator(character: char) -> bool {
    unicode_bidi::bidi_class(character) == unicode_bidi::BidiClass::B
}

fn contains_paragraph_separator(text: &str) -> bool {
    if text
        .bytes()
        .any(|byte| matches!(byte, b'\n' | b'\r' | 0x1c | 0x1d | 0x1e))
    {
        return true;
    }

    !text.is_ascii() && text.chars().any(is_paragraph_separator)
}

fn clip_font_runs(font_runs: &[FontRun], range: Range<usize>) -> SmallVec<[FontRun; 4]> {
    let mut clipped = SmallVec::new();
    let mut offs = 0;
    for run in font_runs {
        let run_start = offs;
        offs += run.len;
        if offs <= range.start {
            continue;
        }
        if run_start >= range.end {
            break;
        }
        let start = run_start.max(range.start);
        let end = offs.min(range.end);
        if start < end {
            clipped.push(FontRun {
                len: end - start,
                font_id: run.font_id,
            });
        }
    }
    clipped
}

#[cfg(feature = "font-kit")]
fn find_best_match(
    font: &Font,
    candidates: &[FontId],
    state: &CosmicTextSystemState,
) -> Result<usize> {
    let database_ids = candidates
        .iter()
        .map(|font_id| state.loaded_fonts[font_id.0].database_id)
        .collect::<SmallVec<[_; 4]>>();
    find_best_face(font, &database_ids, state)
}

#[cfg(not(feature = "font-kit"))]
fn find_best_match(
    font: &Font,
    candidates: &[FontId],
    state: &CosmicTextSystemState,
) -> Result<usize> {
    if candidates.is_empty() {
        anyhow::bail!("requested font family contains no font matching the other parameters");
    }
    if candidates.len() == 1 {
        return Ok(0);
    }

    let database_ids = candidates
        .iter()
        .map(|font_id| state.loaded_fonts[font_id.0].database_id)
        .collect::<SmallVec<[_; 4]>>();
    find_best_face(font, &database_ids, state)
}

#[cfg(feature = "font-kit")]
fn find_best_face(
    font: &Font,
    candidates: &[cosmic_text::fontdb::ID],
    state: &CosmicTextSystemState,
) -> Result<usize> {
    let candidate_properties = candidates
        .iter()
        .map(|database_id| {
            let face_info = state
                .font_system
                .db()
                .face(*database_id)
                .context("font face not found in database")?;
            Ok(face_info_into_properties(face_info))
        })
        .collect::<Result<SmallVec<[_; 4]>>>()?;

    font_kit::matching::find_best_match(&candidate_properties, &font_into_properties(font))
        .context("requested font family contains no font matching the other parameters")
}

#[cfg(not(feature = "font-kit"))]
fn find_best_face(
    font: &Font,
    candidates: &[cosmic_text::fontdb::ID],
    state: &CosmicTextSystemState,
) -> Result<usize> {
    if candidates.is_empty() {
        anyhow::bail!("requested font family contains no font matching the other parameters");
    }

    let target_weight = font.weight.0;
    let target_italic = matches!(
        font.style,
        gpui::FontStyle::Italic | gpui::FontStyle::Oblique
    );
    let mut best_index = 0;
    let mut best_score = u32::MAX;

    for (index, database_id) in candidates.iter().enumerate() {
        let face_info = state
            .font_system
            .db()
            .face(*database_id)
            .context("font face not found in database")?;
        let is_italic = matches!(
            face_info.style,
            cosmic_text::Style::Italic | cosmic_text::Style::Oblique
        );
        let style_penalty: u32 = if is_italic == target_italic { 0 } else { 1000 };
        let weight_diff = (face_info.weight.0 as i32 - target_weight as i32).unsigned_abs();
        let score = style_penalty + weight_diff;
        if score < best_score {
            best_score = score;
            best_index = index;
        }
    }

    Ok(best_index)
}

/// one contiguous slice of a `FontRun` that maps to a single slot. `slot` is
/// `None` for the primary font and `Some(ix)` for `fallback_chain[ix]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct RunSpan {
    start: usize,
    end: usize,
    slot: Option<usize>,
    font_id: FontId,
}

/// walks `text[run_offset..run_offset + run_len]` and groups codepoints into
/// spans. inheriting codepoints stay in the current span so shaping clusters
/// like emoji zwj sequences and combining marks are not torn apart.
fn compute_run_spans(
    text: &str,
    run_offset: usize,
    run_len: usize,
    primary: FontId,
    fallback_chain: &[(FontId, SharedString)],
    covers: &impl Fn(FontId, char) -> bool,
) -> SmallVec<[RunSpan; 4]> {
    let mut spans = SmallVec::new();
    let run_end = run_offset + run_len;
    if run_end <= run_offset {
        return spans;
    }
    if fallback_chain.is_empty() {
        spans.push(RunSpan {
            start: run_offset,
            end: run_end,
            slot: None,
            font_id: primary,
        });
        return spans;
    }
    let run_text = &text[run_offset..run_end];
    let mut span_start = run_offset;
    let mut span_slot: Option<usize> = None;
    let mut span_font_id = primary;
    for (grapheme_idx, grapheme) in run_text.grapheme_indices(true) {
        let abs = run_offset + grapheme_idx;
        let ch = grapheme.chars().next().unwrap_or('\0');
        let next_slot = pick_covering_slot(ch, span_slot, primary, fallback_chain, covers);
        if next_slot == span_slot {
            continue;
        }
        if abs > span_start {
            spans.push(RunSpan {
                start: span_start,
                end: abs,
                slot: span_slot,
                font_id: span_font_id,
            });
        }
        span_start = abs;
        span_slot = next_slot;
        span_font_id = slot_font_id(next_slot, primary, fallback_chain);
    }
    if span_start < run_end {
        spans.push(RunSpan {
            start: span_start,
            end: run_end,
            slot: span_slot,
            font_id: span_font_id,
        });
    }
    spans
}

fn slot_font_id(
    slot: Option<usize>,
    primary: FontId,
    fallback_chain: &[(FontId, SharedString)],
) -> FontId {
    match slot {
        None => primary,
        Some(ix) => fallback_chain[ix].0,
    }
}

fn pick_covering_slot(
    ch: char,
    current: Option<usize>,
    primary: FontId,
    fallback_chain: &[(FontId, SharedString)],
    covers: &impl Fn(FontId, char) -> bool,
) -> Option<usize> {
    if (ch as u32) <= 0x7F {
        return None;
    }
    if covers(primary, ch) {
        return None;
    }
    let current_id = slot_font_id(current, primary, fallback_chain);
    if covers(current_id, ch) {
        return current;
    }

    fallback_chain
        .iter()
        .position(|(fb_id, _)| covers(*fb_id, ch))
}

fn charmap_covers(loaded_fonts: &[LoadedFont], id: FontId, ch: char) -> bool {
    loaded_fonts
        .get(id.0)
        .and_then(|loaded| loaded.font.as_ref())
        .is_some_and(|font| font.as_swash().charmap().map(ch) != 0)
}

fn cosmic_font_features(features: &FontFeatures) -> Result<CosmicFontFeatures> {
    let mut result = CosmicFontFeatures::new();
    for feature in features.0.iter() {
        let name_bytes: [u8; 4] = feature
            .0
            .as_bytes()
            .try_into()
            .context("Incorrect feature flag format")?;

        let tag = cosmic_text::FeatureTag::new(&name_bytes);

        result.set(tag, feature.1);
    }
    Ok(result)
}

#[cfg(feature = "font-kit")]
fn font_into_properties(font: &gpui::Font) -> font_kit::properties::Properties {
    font_kit::properties::Properties {
        style: match font.style {
            gpui::FontStyle::Normal => font_kit::properties::Style::Normal,
            gpui::FontStyle::Italic => font_kit::properties::Style::Italic,
            gpui::FontStyle::Oblique => font_kit::properties::Style::Oblique,
        },
        weight: font_kit::properties::Weight(font.weight.0),
        stretch: Default::default(),
    }
}

#[cfg(feature = "font-kit")]
fn face_info_into_properties(
    face_info: &cosmic_text::fontdb::FaceInfo,
) -> font_kit::properties::Properties {
    font_kit::properties::Properties {
        style: match face_info.style {
            cosmic_text::Style::Normal => font_kit::properties::Style::Normal,
            cosmic_text::Style::Italic => font_kit::properties::Style::Italic,
            cosmic_text::Style::Oblique => font_kit::properties::Style::Oblique,
        },
        weight: font_kit::properties::Weight(face_info.weight.0.into()),
        stretch: match face_info.stretch {
            cosmic_text::Stretch::Condensed => font_kit::properties::Stretch::CONDENSED,
            cosmic_text::Stretch::Expanded => font_kit::properties::Stretch::EXPANDED,
            cosmic_text::Stretch::ExtraCondensed => font_kit::properties::Stretch::EXTRA_CONDENSED,
            cosmic_text::Stretch::ExtraExpanded => font_kit::properties::Stretch::EXTRA_EXPANDED,
            cosmic_text::Stretch::Normal => font_kit::properties::Stretch::NORMAL,
            cosmic_text::Stretch::SemiCondensed => font_kit::properties::Stretch::SEMI_CONDENSED,
            cosmic_text::Stretch::SemiExpanded => font_kit::properties::Stretch::SEMI_EXPANDED,
            cosmic_text::Stretch::UltraCondensed => font_kit::properties::Stretch::ULTRA_CONDENSED,
            cosmic_text::Stretch::UltraExpanded => font_kit::properties::Stretch::ULTRA_EXPANDED,
        },
    }
}

fn font_has_color_glyphs(font: &CosmicTextFont) -> bool {
    let swash_font = font.as_swash();
    swash_font.color_palettes().len() > 0 || swash_font.color_strikes().len() > 0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fid(i: usize) -> FontId {
        FontId(i)
    }

    fn chain(ids: &[usize]) -> SmallVec<[(FontId, SharedString); 4]> {
        ids.iter()
            .map(|&i| (fid(i), SharedString::from(format!("fb{i}"))))
            .collect()
    }

    fn span(start: usize, end: usize, slot: Option<usize>, font_id: FontId) -> RunSpan {
        RunSpan {
            start,
            end,
            slot,
            font_id,
        }
    }

    const IBM_PLEX: &[u8] =
        include_bytes!("../../../assets/fonts/ibm-plex-sans/IBMPlexSans-Regular.ttf");

    /// Every code point of `Bidi_Class=B`, each of which starts a new bidi
    /// paragraph and so can split one line into mixed-direction paragraphs.
    const SEPARATORS: &[char] = &[
        '\u{000a}', '\u{000d}', '\u{001c}', '\u{001d}', '\u{001e}', '\u{0085}', '\u{2029}',
    ];

    fn text_system() -> Result<CosmicTextSystem> {
        let text_system = CosmicTextSystem::new_without_system_fonts("IBM Plex Sans");
        text_system.add_fonts(vec![Cow::Borrowed(IBM_PLEX)])?;
        Ok(text_system)
    }

    fn layout_text(text_system: &CosmicTextSystem, text: &str) -> Result<LineLayout> {
        let font_id = text_system.font_id(&gpui::font("IBM Plex Sans"))?;
        let runs = [FontRun {
            len: text.len(),
            font_id,
        }];
        Ok(text_system.layout_line(text, gpui::px(14.0), &runs))
    }

    /// Mirrors the original crash: mixed-direction text reaching the shaper
    /// through `shape_text`, which only splits lines on `\n`.
    #[test]
    fn shape_text_with_mixed_direction_paragraphs() -> Result<()> {
        let platform_text_system = Arc::new(text_system()?);
        let text_system = Arc::new(gpui::TextSystem::new(platform_text_system));
        let window_text_system = gpui::WindowTextSystem::new(text_system);

        let text: SharedString = "first line\n\u{05d0}\u{001c}A".into();
        let runs = [gpui::TextRun {
            len: text.len(),
            font: gpui::font("IBM Plex Sans"),
            ..Default::default()
        }];

        let lines = window_text_system.shape_text(text, gpui::px(14.0), &runs, None, None)?;

        assert_eq!(lines.len(), 2);
        assert_eq!(lines[1].len(), "\u{05d0}\u{001c}A".len());
        assert!(lines[1].width() > Pixels::ZERO);
        Ok(())
    }

    #[test]
    fn layout_line_with_mixed_direction_paragraphs() -> Result<()> {
        let text_system = text_system()?;

        for separator in SEPARATORS {
            for text in [
                format!("\u{05d0}{separator}A"),
                format!("A{separator}\u{05d0}"),
            ] {
                let layout = layout_text(&text_system, &text)?;

                assert_eq!(layout.len, text.len(), "{text:?}");
                assert!(layout.width > Pixels::ZERO, "{text:?}");
                assert!(
                    layout.runs.iter().any(|run| !run.glyphs.is_empty()),
                    "{text:?}"
                );
            }
        }

        Ok(())
    }

    #[test]
    fn layout_line_with_separators_at_line_edges() -> Result<()> {
        let text_system = text_system()?;

        for text in [
            "\u{001c}",
            "\u{001c}\u{001c}",
            "\u{001c}\u{05d0}",
            "\u{05d0}\u{001c}",
            "\u{05d0}\u{001c}\u{001c}A",
            "\u{001c}\u{05d0}\u{001c}A\u{001c}",
        ] {
            let layout = layout_text(&text_system, text)?;
            assert_eq!(layout.len, text.len(), "{text:?}");
        }

        Ok(())
    }

    /// Glyph indices must stay absolute and positions ordered across segment
    /// boundaries, otherwise cursor placement and hit testing desync. Uses
    /// single-direction text so visual order matches logical order.
    #[test]
    fn layout_line_keeps_indices_and_positions_ordered_across_paragraphs() -> Result<()> {
        let text_system = text_system()?;
        let text = "ab\u{001c}cd\u{2029}ef";
        let layout = layout_text(&text_system, text)?;

        let glyphs: Vec<_> = layout.runs.iter().flat_map(|run| &run.glyphs).collect();
        assert!(!glyphs.is_empty());

        for glyph in &glyphs {
            assert!(glyph.index < text.len(), "{:?}", glyph.index);
            assert!(text.is_char_boundary(glyph.index), "{:?}", glyph.index);
        }
        for pair in glyphs.windows(2) {
            assert!(pair[0].index < pair[1].index);
            assert!(pair[0].position.x <= pair[1].position.x);
        }

        // Every segment contributes width, so the whole line is wider than its
        // leading paragraph alone.
        assert!(layout.width > layout_text(&text_system, "ab")?.width);
        Ok(())
    }

    /// A font run boundary that does not line up with a paragraph boundary must
    /// still be clipped to the right segments.
    #[test]
    fn layout_line_with_font_run_straddling_a_separator() -> Result<()> {
        let text_system = text_system()?;
        let font_id = text_system.font_id(&gpui::font("IBM Plex Sans"))?;
        let text = "ab\u{001c}\u{05d0}\u{05d1}";

        // The run boundary falls inside the trailing RTL paragraph.
        let runs = [
            FontRun {
                len: "ab\u{001c}\u{05d0}".len(),
                font_id,
            },
            FontRun {
                len: "\u{05d1}".len(),
                font_id,
            },
        ];
        let layout = text_system.layout_line(text, gpui::px(14.0), &runs);

        assert_eq!(layout.len, text.len());
        assert!(layout.width > Pixels::ZERO);
        Ok(())
    }

    /// Lines with no separator take the fast path and must be shaped exactly as
    /// they were before paragraph splitting existed.
    #[test]
    fn layout_line_without_separators_takes_fast_path() -> Result<()> {
        let text_system = text_system()?;

        for text in [
            "hello world",
            "\u{05d0}\u{05d1}\u{05d2}",
            "mixed \u{05d0}\u{05d1}",
        ] {
            assert!(!contains_paragraph_separator(text), "{text:?}");
            let layout = layout_text(&text_system, text)?;
            assert_eq!(layout.len, text.len(), "{text:?}");
            assert!(layout.width > Pixels::ZERO, "{text:?}");
        }

        Ok(())
    }

    #[test]
    fn paragraph_separator_detection() {
        for separator in SEPARATORS {
            assert!(is_paragraph_separator(*separator), "{separator:?}");
            assert!(contains_paragraph_separator(&format!("a{separator}b")));
        }

        for text in [
            "",
            "plain ascii",
            "\u{05d0}",
            "tab\there",
            "emoji \u{1f600}",
        ] {
            assert!(!contains_paragraph_separator(text), "{text:?}");
        }
    }

    #[test]
    fn font_runs_are_clipped_to_segment() {
        let runs = [
            FontRun {
                len: 3,
                font_id: fid(1),
            },
            FontRun {
                len: 4,
                font_id: fid(2),
            },
        ];

        assert_eq!(clip_font_runs(&runs, 0..7).as_slice(), &runs);
        assert_eq!(
            clip_font_runs(&runs, 2..5).as_slice(),
            &[
                FontRun {
                    len: 1,
                    font_id: fid(1)
                },
                FontRun {
                    len: 2,
                    font_id: fid(2)
                },
            ]
        );
        assert_eq!(
            clip_font_runs(&runs, 3..7).as_slice(),
            &[FontRun {
                len: 4,
                font_id: fid(2)
            }]
        );
        assert!(clip_font_runs(&runs, 5..5).is_empty());
    }

    #[test]
    fn primary_wins_over_current_fallback_when_primary_covers() {
        let primary = fid(0);
        let fb = chain(&[1, 2]);
        let covers = |id: FontId, _: char| id == fid(0) || id == fid(1);
        assert_eq!(
            pick_covering_slot('a', Some(0), primary, &fb, &covers),
            None
        );
    }

    #[test]
    fn primary_preferred_over_fallback_when_both_cover() {
        let primary = fid(0);
        let fb = chain(&[1]);
        let covers = |_: FontId, _: char| true;
        assert_eq!(pick_covering_slot('a', None, primary, &fb, &covers), None);
    }

    #[test]
    fn falls_through_chain_in_order() {
        let primary = fid(0);
        let fb = chain(&[1, 2, 3]);
        // only fallback 2 at index 1 covers.
        let covers = |id: FontId, _: char| id == fid(2);
        assert_eq!(
            pick_covering_slot('字', None, primary, &fb, &covers),
            Some(1)
        );
    }

    #[test]
    fn no_coverage_returns_primary() {
        let primary = fid(0);
        let fb = chain(&[1, 2]);
        let covers = |_: FontId, _: char| false;
        // nothing covers. return `None` so the `cosmic-text` built in script
        // fallback can take over during shaping.
        assert_eq!(
            pick_covering_slot('\u{1F600}', Some(1), primary, &fb, &covers),
            None
        );
    }

    #[test]
    fn empty_chain_always_returns_primary() {
        let primary = fid(0);
        let fb: SmallVec<[(FontId, SharedString); 4]> = SmallVec::new();
        let covers = |_: FontId, _: char| false;
        assert_eq!(pick_covering_slot('a', None, primary, &fb, &covers), None);
    }

    #[test]
    fn slot_font_id_resolution() {
        let primary = fid(7);
        let fb = chain(&[10, 20]);
        assert_eq!(slot_font_id(None, primary, &fb), fid(7));
        assert_eq!(slot_font_id(Some(0), primary, &fb), fid(10));
        assert_eq!(slot_font_id(Some(1), primary, &fb), fid(20));
    }

    #[test]
    fn run_spans_with_no_chain_emit_one_primary_span() {
        let primary = fid(0);
        let fb: SmallVec<[(FontId, SharedString); 4]> = SmallVec::new();
        let covers = |_: FontId, _: char| false;
        let text = "hello";
        let spans = compute_run_spans(text, 0, text.len(), primary, &fb, &covers);
        assert_eq!(spans.as_slice(), &[span(0, text.len(), None, primary)]);
    }

    #[test]
    fn run_spans_use_byte_offsets_for_multibyte_chars() {
        let primary = fid(0);
        let fb = chain(&[1]);
        // primary covers ascii. fallback covers cjk.
        let covers = |id: FontId, ch: char| {
            if id == primary {
                ch.is_ascii()
            } else {
                !ch.is_ascii()
            }
        };
        let text = "a字b";
        let spans = compute_run_spans(text, 0, text.len(), primary, &fb, &covers);
        // '字' is 3 bytes so split is at 1 then 4.
        assert_eq!(
            spans.as_slice(),
            &[
                span(0, 1, None, primary),
                span(1, 4, Some(0), fid(1)),
                span(4, 5, None, primary),
            ]
        );
    }

    #[test]
    fn run_spans_respect_run_offset() {
        let primary = fid(0);
        let fb = chain(&[1]);
        let covers = |id: FontId, ch: char| {
            if id == primary {
                ch.is_ascii()
            } else {
                !ch.is_ascii()
            }
        };
        // outer text has a prefix that is not part of this run.
        let text = "xx字y";
        let run_offset = 2;
        let run_len = text.len() - run_offset;
        let spans = compute_run_spans(text, run_offset, run_len, primary, &fb, &covers);
        assert_eq!(
            spans.as_slice(),
            &[span(2, 5, Some(0), fid(1)), span(5, 6, None, primary)]
        );
    }

    #[test]
    fn run_spans_keep_combining_marks_with_base_in_fallback() {
        let primary = fid(0);
        let fb = chain(&[1]);
        // primary covers ascii only. fallback covers the base char.
        // combining mark must stay in the fallback span even when fallback
        // does not advertise coverage of it.
        let covers = |id: FontId, ch: char| {
            if id == primary {
                ch.is_ascii()
            } else {
                ch == '\u{0905}'
            }
        };
        // \u{0905} devanagari short a + \u{0902} candrabindu mark.
        let text = "\u{0905}\u{0902}";
        let spans = compute_run_spans(text, 0, text.len(), primary, &fb, &covers);
        assert_eq!(spans.as_slice(), &[span(0, text.len(), Some(0), fid(1))]);
    }

    #[test]
    fn run_spans_keep_zwj_inside_emoji_cluster() {
        let primary = fid(0);
        let fb = chain(&[1]);
        // only fallback covers the emoji codepoints. zwj must not split.
        let covers = |id: FontId, ch: char| id == fid(1) && ch != '\u{200D}';
        // family zwj sequence woman zwj girl.
        let text = "\u{1F469}\u{200D}\u{1F467}";
        let spans = compute_run_spans(text, 0, text.len(), primary, &fb, &covers);
        assert_eq!(spans.as_slice(), &[span(0, text.len(), Some(0), fid(1))]);
    }

    #[test]
    fn run_spans_collapse_adjacent_same_slot() {
        let primary = fid(0);
        let fb = chain(&[1]);
        let covers = |id: FontId, ch: char| {
            if id == primary {
                ch.is_ascii()
            } else {
                !ch.is_ascii()
            }
        };
        let text = "字字字";
        let spans = compute_run_spans(text, 0, text.len(), primary, &fb, &covers);
        assert_eq!(spans.as_slice(), &[span(0, text.len(), Some(0), fid(1))]);
    }

    #[test]
    fn run_spans_empty_run_returns_no_spans() {
        let primary = fid(0);
        let fb = chain(&[1]);
        let covers = |_: FontId, _: char| true;
        let spans = compute_run_spans("anything", 3, 0, primary, &fb, &covers);
        assert!(spans.is_empty());
    }
}
