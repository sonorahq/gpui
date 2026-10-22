// Liquid metal: polished chrome whose reflections flow slowly, with bright glints travelling
// around the edge and an iridescent fringe where the reflected horizon breaks.
//
// The surface is a bevel along the rounded edge. With a rim width it is a thin tube, a chrome
// border over whatever the interior shows. With a rim width of zero it is a dome over the whole
// shape.
//
// params[0].x  rim width in logical pixels, 0 fills the whole shape
// params[0].y  flow speed, 1 is a calm drift
// params[0].z  light angle in radians, 0 lights from the top
// params[0].w  highlight intensity, around 1
// params[1].x  chromatic fringe amount, 0 to 1
// params[1].y  pressure, 0 at rest and 1 fully pressed or hovered, which brightens and speeds up
// colors[0]    the darkest tone the metal reflects
// colors[1]    the brightest tone the metal reflects
// colors[2]    what the interior inside the rim shows, transparent to show nothing

const TAU: f32 = 6.2831853;

// The direction away from the shape at `position`, from the gradient of its distance field.
fn outward(input: ShaderInput, position: vec2<f32>) -> vec2<f32> {
    let half_size = input.size * 0.5;
    let p = position - half_size;
    let e = 0.5;
    let dx = sd_rounded_rect(p + vec2<f32>(e, 0.0), half_size, input.corner_radii)
        - sd_rounded_rect(p - vec2<f32>(e, 0.0), half_size, input.corner_radii);
    let dy = sd_rounded_rect(p + vec2<f32>(0.0, e), half_size, input.corner_radii)
        - sd_rounded_rect(p - vec2<f32>(0.0, e), half_size, input.corner_radii);
    let gradient = vec2<f32>(dx, dy);
    let length_squared = dot(gradient, gradient);
    if length_squared < 1e-8 {
        return normalize(p + vec2<f32>(1e-3, 0.0));
    }
    return gradient * inverseSqrt(length_squared);
}

// A studio as the metal sees it: a bright sky above a sharp horizon, a dark floor below, and a
// pair of thin strip lights. `e` runs from -1, straight down, to 1, straight up.
fn studio(e: f32) -> f32 {
    let sky = smoothstep(-0.03, 0.03, e) * (0.42 + 0.45 * smoothstep(0.0, 1.0, e));
    let floor_glow = 0.05 * smoothstep(-1.0, -0.3, e);
    let strip_high = exp(-pow((e - 0.58) / 0.05, 2.0)) * 0.8;
    let strip_low = exp(-pow((e + 0.42) / 0.035, 2.0)) * 0.45;
    return sky + floor_glow + strip_high + strip_low;
}

fn shade(input: ShaderInput) -> vec4<f32> {
    let rim_width = input.params[0].x;
    let speed = input.params[0].y;
    let light_angle = input.params[0].z;
    let pressure = saturate(input.params[1].y);
    let intensity = input.params[0].w * (1.0 + 0.6 * pressure);
    let chroma = input.params[1].x * (1.0 + 0.8 * pressure);
    let t = input.time * speed * (1.0 + 0.75 * pressure);

    let half_size = input.size * 0.5;
    let p = input.position - half_size;
    let filled = rim_width <= 0.0;
    let width = select(rim_width, min(half_size.x, half_size.y), filled);
    let depth = -input.distance;
    let across = saturate(depth / max(width, 1e-3));

    // Where around the outline the pixel sits, from 0 to 1, with each side given an equal share.
    let around = atan2(p.y / max(half_size.y, 1e-3), p.x / max(half_size.x, 1e-3)) / TAU + 0.5;

    // Up in the studio is the light direction, in a y-down frame.
    let light = vec2<f32>(sin(light_angle), -cos(light_angle));
    let away = outward(input, input.position);
    let flow_domain = input.position / max(min(input.size.x, input.size.y), 1.0);
    let flow_scale = select(1.3, 0.75, filled);
    let flow = fbm(flow_domain * flow_scale + vec2<f32>(0.11 * t, -0.07 * t), 3);
    let swirl = fbm(flow_domain * 2.7 - vec2<f32>(0.05 * t, 0.09 * t) + vec2<f32>(flow), 2);

    // Where the pixel looks in the studio. A thin rim is a tube seen edge on, so what it reflects
    // turns with the outline: the top of the shape sees the sky and the bottom the floor, with
    // the flow pushing the horizon around it. A dome reflects through its curved normal, which
    // faces the viewer in the middle, so there the flow alone decides.
    var e: f32;
    if filled {
        let bend = 1.5707963 * across;
        let normal = normalize(vec3<f32>(away * cos(bend), sin(bend) + 0.15));
        let reflected = reflect(vec3<f32>(0.0, 0.0, -1.0), normal);
        e = dot(reflected.xy, light) * 0.9 + 0.45 * flow + 0.08 * swirl;
    } else {
        let facing = dot(away, light);
        e = facing * (0.85 - 0.5 * across) + 0.25 * (0.5 - across) + 0.4 * flow + 0.12 * swirl;
    }

    // Each channel sees the studio a little displaced, which fringes the sharp edges.
    let spread = 0.045 * chroma;
    var env = vec3<f32>(studio(e + spread), studio(e), studio(e - spread));

    // Thin film colour where the horizon is crossed, strongest on the steepest part of it.
    let crossing = exp(-pow(e / 0.08, 2.0));
    let film = 0.5 + 0.5 * cos(TAU * (vec3<f32>(0.0, 0.33, 0.67) + e * 3.0 + around * 2.0 + 0.05 * t));
    env += film * crossing * chroma * 0.55;

    // Fine streaks along the outline, the grain of brushed metal.
    let grain = value_noise(vec2<f32>(around * 140.0, across * 9.0 + depth * 0.6));
    env *= 0.93 + 0.14 * grain;

    // Two glints running around the rim in opposite directions, each a white core in a halo
    // that splits into colour.
    let phase_a = saturate(cos(TAU * (around - 0.07 * t)));
    let phase_b = saturate(cos(TAU * (around + 0.045 * t + 0.37)));
    let crest = exp(-pow((across - select(0.5, 0.25, filled)) / 0.3, 2.0));
    let cores = (pow(phase_a, 120.0) * 1.8 + pow(phase_b, 180.0) * 1.2) * crest;
    let halos = (pow(phase_a, 14.0) + 0.7 * pow(phase_b, 18.0)) * crest * 0.45;
    let halo_tint = 0.5 + 0.5 * cos(TAU * (vec3<f32>(0.0, 0.33, 0.67) + around * 6.0 - 0.1 * t));

    let dark = input.colors[0].rgb;
    let bright = input.colors[1].rgb;
    var metal = mix(dark, bright, saturate(env * intensity));
    metal += (vec3<f32>(cores) + halos * mix(vec3<f32>(1.0), halo_tint, chroma)) * intensity;
    // A highlight past full white rolls off instead of clipping flat.
    metal = vec3<f32>(1.0) - exp(-max(metal, vec3<f32>(0.0)) * 1.15);
    metal = metal / (vec3<f32>(1.0) - exp(vec3<f32>(-1.15)));

    if filled {
        return vec4<f32>(dither(metal, input.frag_coord), 1.0);
    }

    // Inside the rim the interior shows, with a soft sheen at the top and a shadow cast by the
    // rim onto it.
    let ring = saturate(0.5 + (rim_width - depth) * input.scale_factor);
    let interior = input.colors[2];
    let inner_depth = max(depth - rim_width, 0.0);
    let shadow = 1.0 - 0.35 * exp(-inner_depth / max(rim_width * 2.5, 1.0));
    let sheen = 0.05 * smoothstep(0.55, 0.0, input.uv.y) * (1.0 + pressure);
    let inside = interior.rgb * shadow + vec3<f32>(sheen);
    let color = mix(inside, metal, ring);
    let alpha = mix(interior.a, 1.0, ring);
    return vec4<f32>(dither(color, input.frag_coord), alpha);
}
