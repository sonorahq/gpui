// Aurora: a slow liquid gradient between up to four colours, for large backgrounds and for
// frosted panels to sit on. Noise warped by more noise gives it soft folds, and a dither of one
// 8 bit step keeps the long gradients from banding.
//
// It is built to be cheap over a whole window. The colours are mixed with their components
// squared, which is close to linear light for soft gradients and far cheaper than a per pixel
// trip through Oklab, and the folds are slow waves with only two lookups of noise.
//
// params[0].x  speed, 1 is a slow drift
// params[0].y  feature size in logical pixels, around a third of the area it covers
// params[0].z  contrast, 0 for the average colour alone and 1 for the full palette
// params[0].w  warp, how far the folds bend, around 1
// params[1].x  how many of the colours to use, 2 to 4
// colors[0..3] the palette, from the bottom of the folds to the top

fn light(color: vec4<f32>) -> vec3<f32> {
    return color.rgb * color.rgb;
}

fn palette(input: ShaderInput, count: i32, s: f32) -> vec3<f32> {
    let position = saturate(s) * f32(count - 1);
    let c0 = light(input.colors[0]);
    let c1 = light(input.colors[1]);
    let c2 = light(input.colors[2]);
    let c3 = light(input.colors[3]);
    let low = mix(c0, c1, smoothstep(0.0, 1.0, saturate(position)));
    let middle = mix(low, c2, smoothstep(0.0, 1.0, saturate(position - 1.0)));
    return mix(middle, c3, smoothstep(0.0, 1.0, saturate(position - 2.0)));
}

fn shade(input: ShaderInput) -> vec4<f32> {
    let t = input.time * input.params[0].x;
    let scale = max(input.params[0].y, 1.0);
    let contrast = saturate(input.params[0].z);
    let warp = input.params[0].w;
    let count = clamp(i32(input.params[1].x + 0.5), 2, 4);

    let q = input.position / scale;
    // Slow waves fold the plane, a little noise keeps the folds from repeating, and the field
    // read through them is waves and noise again.
    let drift = gradient_noise(q * 0.9 + vec2<f32>(0.03 * t, -0.02 * t));
    let fold = 0.5 * vec2<f32>(
        sin(q.y * 1.3 + 0.11 * t + drift) + sin(q.x * 0.7 - 0.07 * t),
        cos(q.x * 1.1 - 0.09 * t) + sin(q.y * 0.6 + 0.05 * t - drift),
    );
    let r = q * 0.8 + warp * 1.7 * fold;
    let waves = sin(r.x * 0.9 + sin(r.y * 1.4 + 0.06 * t)) + sin(r.y * 0.8 - 0.04 * t + cos(r.x * 0.5));
    let field = 0.38 * waves + 0.3 * gradient_noise(r * 1.3 - vec2<f32>(0.0, 0.02 * t));

    // Lean the palette from its first colour at the bottom to its last at the top, with the
    // folds carrying it back and forth across that.
    let s = 0.5 + 0.75 * field + 0.35 * (0.5 - input.uv.y) + 0.15 * fold.x;
    let full = palette(input, count, s);

    let average = (light(input.colors[0]) + light(input.colors[1]) + select(vec3<f32>(0.0), light(input.colors[2]), count > 2) + select(vec3<f32>(0.0), light(input.colors[3]), count > 3)) / f32(count);

    let color = sqrt(max(mix(average, full, contrast), vec3<f32>(0.0)));
    return vec4<f32>(dither(color, input.frag_coord), input.colors[0].a);
}
