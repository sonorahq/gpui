// Dot matrix: a field of LEDs lit by ordered dithering. A smooth pattern sets how dense the lit
// dots are, and a Bayer matrix decides which dots light, so a gradient reads as a fade in dot
// density rather than in brightness.
//
// The pitch is rounded to whole device pixels and the grid is laid on the device pixel grid, so
// the dots stay crisp at any scale.
//
// params[0].x  pitch in logical pixels
// params[0].y  dot size as a share of the pitch, 0 to 1
// params[0].z  shape, 0 for square dots and 1 for round ones
// params[0].w  Bayer matrix bits, 1 to 4 for a 2x2 to a 16x16 matrix
// params[1].x  pattern: 0 a wave rising from the bottom, 1 ripples from `params[2].xy`,
//              2 a progress sweep from the left, 3 sparkles, 4 a scanning shimmer
// params[1].y  level, 0 to 1, how much of the field is lit: the wave's height, the ripple's
//              reach, the progress, the sparkle density or the shimmer strength
// params[1].z  speed
// params[1].w  softness, the share of the field over which the density fades out
// params[2].xy ripple origin, in the 0 to 1 coordinates of the bounds
// params[2].z  heat, how far the densest dots shift towards `colors[2]`
// colors[0]    the background behind the dots, transparent to show what is under it
// colors[1]    a lit dot
// colors[2]    a lit dot where the field is densest
// colors[3]    an unlit dot, transparent to leave unlit dots out

const TAU: f32 = 6.2831853;

fn field(pattern: i32, uv: vec2<f32>, aspect: f32, cell: vec2<f32>, level: f32, softness: f32, t: f32, origin: vec2<f32>) -> f32 {
    let soft = max(softness, 1e-3);
    switch pattern {
        case 1: {
            let d = length((uv - origin) * vec2<f32>(aspect, 1.0));
            let reach = level * (1.0 + soft) * max(aspect, 1.0);
            let envelope = saturate((reach - d) / soft + 0.5);
            let rings = 0.5 + 0.5 * cos(d * 17.0 - t * 4.0);
            return envelope * mix(0.18, 1.0, rings) * (1.0 - 0.35 * d / max(reach, 1e-3));
        }
        case 2: {
            let front = saturate((level - uv.x) / soft + 0.5);
            let shimmer = 0.5 + 0.5 * sin((uv.x * aspect - t * 0.6) * 9.0);
            return front * mix(0.82, 1.0, shimmer) + 0.06 * (value_noise(cell * 0.4 + vec2<f32>(t)) - 0.5);
        }
        case 3: {
            let seed = hash2d(vec2<i32>(cell));
            let twinkle = 0.5 + 0.5 * sin(t * (1.3 + 2.1 * seed) + seed * TAU);
            let drift = value_noise(uv * vec2<f32>(aspect, 1.0) * 3.0 + vec2<f32>(0.2 * t, -0.13 * t));
            return pow(twinkle, 5.0) * level * (0.4 + 1.2 * drift);
        }
        case 4: {
            let sweep = fract(t * 0.35) * (1.0 + 2.0 * soft) - soft;
            let band = exp(-pow((uv.x - sweep) / soft, 2.0));
            let grain = 0.7 + 0.3 * value_noise(cell * 0.5 + vec2<f32>(0.0, t));
            return band * level * grain;
        }
        default: {
            let x = uv.x * aspect;
            let surface = level
                + 0.11 * fbm(vec2<f32>(x * 1.3 + 0.12 * t, 0.07 * t), 3)
                + 0.05 * sin(x * 4.2 + t * 0.8)
                + 0.03 * sin(x * 9.7 - t * 1.3);
            let height = 1.0 - uv.y;
            return saturate((surface - height) / soft + 0.5);
        }
    }
}

// Lays `above` over `below`, both with straight alpha.
fn over(below: vec4<f32>, above: vec4<f32>) -> vec4<f32> {
    let alpha = above.a + below.a * (1.0 - above.a);
    if alpha <= 0.0 {
        return vec4<f32>(0.0);
    }
    let rgb = (above.rgb * above.a + below.rgb * below.a * (1.0 - above.a)) / alpha;
    return vec4<f32>(rgb, alpha);
}

fn shade(input: ShaderInput) -> vec4<f32> {
    let pitch = max(1.0, round(input.params[0].x * input.scale_factor));
    let size = saturate(input.params[0].y);
    let round_dots = input.params[0].z >= 0.5;
    let bits = u32(clamp(input.params[0].w, 1.0, 4.0));
    let pattern = i32(input.params[1].x + 0.5);
    let level = input.params[1].y;
    let t = input.time * input.params[1].z;
    let softness = input.params[1].w;
    let origin = input.params[2].xy;
    let heat = input.params[2].z;

    // Centre the grid in the bounds so the margin left over is the same on both sides.
    let count = ceil(input.device_size / pitch);
    let offset = floor((count * pitch - input.device_size) * 0.5);
    let local = input.device_position + offset;
    let cell = floor(local / pitch);
    let within = local - cell * pitch;
    let cell_uv = ((cell + 0.5) * pitch - offset) / max(input.device_size, vec2<f32>(1.0));
    let aspect = input.size.x / max(input.size.y, 1e-3);

    let density = field(pattern, cell_uv, aspect, cell, level, softness, t, origin);
    let lit = density > bayer(vec2<u32>(cell), bits);

    let dot_size = max(1.0, round(pitch * size));
    let lead = floor((pitch - dot_size) * 0.5);
    var covered = 0.0;
    if round_dots {
        let centre = vec2<f32>(lead + dot_size * 0.5);
        covered = saturate(dot_size * 0.5 + 0.5 - length(within - centre));
    } else {
        let inside = within >= vec2<f32>(lead) & within < vec2<f32>(lead + dot_size);
        covered = select(0.0, 1.0, all(inside));
    }

    var dot_color = input.colors[3];
    if lit {
        let hot = input.colors[2];
        let warmth = smoothstep(0.55, 1.0, density) * heat * hot.a;
        dot_color = vec4<f32>(mix(input.colors[1].rgb, hot.rgb, warmth), input.colors[1].a);
    }
    dot_color.a *= covered;
    return over(input.colors[0], dot_color);
}
