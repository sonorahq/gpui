// Aurora: a slow liquid gradient between up to four colours, for large backgrounds and for
// frosted panels to sit on. Noise warped by more noise gives it soft folds, the colours are
// mixed in Oklab, and a dither of one 8 bit step keeps the long gradients from banding.
//
// params[0].x  speed, 1 is a slow drift
// params[0].y  feature size in logical pixels, around a third of the area it covers
// params[0].z  contrast, 0 for the average colour alone and 1 for the full palette
// params[0].w  warp, how far the folds bend, around 1
// params[1].x  how many of the colours to use, 2 to 4
// colors[0..3] the palette, from the bottom of the folds to the top

fn palette(input: ShaderInput, count: i32, s: f32) -> vec3<f32> {
    let span = f32(count - 1);
    let position = saturate(s) * span;
    let index = min(i32(floor(position)), count - 2);
    let blend = smoothstep(0.0, 1.0, position - f32(index));
    return mix_oklab(input.colors[index].rgb, input.colors[index + 1].rgb, blend);
}

fn shade(input: ShaderInput) -> vec4<f32> {
    let t = input.time * input.params[0].x;
    let scale = max(input.params[0].y, 1.0);
    let contrast = saturate(input.params[0].z);
    let warp = input.params[0].w;
    let count = clamp(i32(input.params[1].x + 0.5), 2, 4);

    let q = input.position / scale;
    let fold = vec2<f32>(
        fbm(q + vec2<f32>(0.0, 0.045 * t), 3),
        fbm(q + vec2<f32>(5.2, 1.3) - vec2<f32>(0.035 * t, 0.0), 3),
    );
    let field = fbm(q * 0.8 + warp * 1.7 * fold + vec2<f32>(0.02 * t, -0.015 * t), 3);

    // Lean the palette from its first colour at the bottom to its last at the top, with the
    // folds carrying it back and forth across that.
    let s = 0.5 + 0.75 * field + 0.35 * (0.5 - input.uv.y) + 0.15 * fold.x;
    let full = palette(input, count, s);

    var average = vec3<f32>(0.0);
    for (var i = 0; i < count; i++) {
        average += srgb_to_linear(input.colors[i].rgb);
    }
    average = linear_to_srgb(average / f32(count));

    let color = mix_oklab(average, full, contrast);
    return vec4<f32>(dither(color, input.frag_coord), input.colors[0].a);
}
