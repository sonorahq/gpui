// --- subpixel sprites --- //

// Subpixel sprites share the monochrome sprite layout and vertex shader.

struct SubpixelSpriteFragmentOutput {
    @location(0) @blend_src(0) foreground: vec4<f32>,
    @location(0) @blend_src(1) alpha: vec4<f32>,
}

@fragment
fn fs_subpixel_sprite(input: MonoSpriteVarying) -> SubpixelSpriteFragmentOutput {
    var sample = textureSample(t_sprite, s_sprite, input.tile_position).rgb;
    if (gamma_params.is_bgr != 0u) {
        sample = sample.bgr;
    }
    let alpha_corrected = apply_contrast_and_gamma_correction3(sample, input.color.rgb, gamma_params.subpixel_enhanced_contrast, gamma_params.gamma_ratios);

    // Alpha clip after using the derivatives.
    if (any(input.clip_distances < vec4<f32>(0.0))) {
        return SubpixelSpriteFragmentOutput(vec4<f32>(0.0), vec4<f32>(0.0));
    }

    var out = SubpixelSpriteFragmentOutput();
    out.foreground = vec4<f32>(input.color.rgb, 1.0);
    out.alpha = vec4<f32>(input.color.a * alpha_corrected, 1.0);
    return out;
}

// `fs_subpixel_sprite`'s dual-source blend writes no alpha at all, which is fine on the frame
// but wrong in an offscreen filter layer: those start transparent and are composited onto
// their parent by that alpha afterward, so a glyph drawn there would add to the parent
// instead of covering it. This collapses ClearType's three channels into one coverage value
// and shades the glyph the way a grayscale one is shaded, trading subpixel fringing for a
// correct blend.
@fragment
fn fs_subpixel_sprite_layered(input: MonoSpriteVarying) -> @location(0) vec4<f32> {
    var sample = textureSample(t_sprite, s_sprite, input.tile_position).rgb;
    if (gamma_params.is_bgr != 0u) {
        sample = sample.bgr;
    }
    let alpha_corrected = apply_contrast_and_gamma_correction3(sample, input.color.rgb, gamma_params.subpixel_enhanced_contrast, gamma_params.gamma_ratios);

    // Alpha clip after using the derivatives.
    if (any(input.clip_distances < vec4<f32>(0.0))) {
        return vec4<f32>(0.0);
    }

    let coverage = (alpha_corrected.r + alpha_corrected.g + alpha_corrected.b) / 3.0;
    return blend_color(input.color, coverage);
}
