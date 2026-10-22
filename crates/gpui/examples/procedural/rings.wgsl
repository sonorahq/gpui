// Chromatic rings: thin lines of light expanding from a centre, each split into a cool, a white
// and a warm copy a little apart, so a ring has a white core and coloured edges. An optional set of
// diagonal slats cuts across them, every other slat dimmed and its rings shifted half a step,
// like light seen through a louvre.
//
// params[0].xy centre, in the 0 to 1 coordinates of the bounds, and it may lie outside them
// params[0].z  speed, in rings per second
// params[0].w  spacing between rings, in logical pixels
// params[1].x  line width, in logical pixels
// params[1].y  colour separation between the channels, in logical pixels
// params[1].z  slat angle, in radians
// params[1].w  slat width in logical pixels, 0 for no slats
// params[2].x  how much the alternate slats are dimmed, 0 to 1
// params[2].y  intensity
// params[2].z  the distance in logical pixels over which the rings fade, 0 for no fade
// colors[0]    the background
// colors[1]    the tint of rings near the centre
// colors[2]    the tint of rings far from the centre

// The light from the rings at `distance` from the centre. Each ring is drawn three times a
// little apart, cool on the inside, white in the middle and warm on the outside, so where the
// three overlap the core burns white and the edges carry the colour.
fn rings(distance: f32, spacing: f32, width: f32, separation: f32, shift: f32) -> vec3<f32> {
    let tints = array<vec3<f32>, 3>(
        vec3<f32>(0.2, 0.5, 1.0),
        vec3<f32>(1.0, 1.0, 1.0),
        vec3<f32>(1.0, 0.5, 0.15),
    );
    var light = vec3<f32>(0.0);
    for (var layer = 0; layer < 3; layer++) {
        let d = distance + f32(layer - 1) * separation;
        let phase = d / spacing - shift;
        let gap = abs(fract(phase) - 0.5) * spacing;
        let core = exp(-pow(gap / width, 2.0));
        let halo = 0.12 * exp(-pow(gap / (width * 6.0), 2.0));
        light += tints[layer] * (core + halo);
    }
    return light * 0.6;
}

// What one slat shows: its shading and the rings through it.
fn slat(index: f32, across: f32, distance: f32, input: ShaderInput, t: f32) -> vec3<f32> {
    let spacing = max(input.params[0].w, 1.0);
    let width = max(input.params[1].x, 0.25);
    let separation = input.params[1].y;
    let dim = saturate(input.params[2].x);
    let alternate = fract(index * 0.5) > 0.25;
    let shift = t + select(0.0, 0.3, alternate);

    let light = rings(distance, spacing, width, separation, shift);
    let fade_distance = input.params[2].z;
    let fade = select(1.0, exp(-distance / max(fade_distance, 1.0)), fade_distance > 0.0);
    let tint = mix(input.colors[1].rgb, input.colors[2].rgb, saturate(distance / max(fade_distance, 1.0) * 0.8));

    // Each slat is lit from one edge, which gives it depth.
    let shading = mix(0.5, 1.0, smoothstep(0.0, 1.0, across));
    let strength = select(1.0, 1.0 - dim, alternate);
    let background = input.colors[0].rgb * shading * mix(1.0, 0.55, select(0.0, dim, alternate));
    return background + light * tint * fade * strength * shading * input.params[2].y;
}

fn shade(input: ShaderInput) -> vec4<f32> {
    let t = input.time * input.params[0].z;
    let centre = input.params[0].xy * input.size;
    let distance = length(input.position - centre);
    let slat_width = input.params[1].w;

    var color: vec3<f32>;
    if slat_width <= 0.0 {
        color = slat(0.0, 1.0, distance, input, t);
    } else {
        let angle = input.params[1].z;
        let coordinate = dot(input.position, vec2<f32>(cos(angle), sin(angle))) / slat_width;
        let index = floor(coordinate);
        let across = fract(coordinate);
        color = slat(index, across, distance, input, t);

        // Blend with the neighbouring slat over the last device pixel so the edge is smooth.
        let to_edge = min(across, 1.0 - across) * slat_width * input.scale_factor;
        let blend = saturate(0.5 - to_edge);
        if blend > 0.0 {
            let neighbour = select(index + 1.0, index - 1.0, across < 0.5);
            let neighbour_across = select(0.0, 1.0, across < 0.5);
            color = mix(color, slat(neighbour, neighbour_across, distance, input, t), blend * 0.5);
        }
    }

    // A soft roll off keeps the brightest crossings white instead of clipping.
    color = vec3<f32>(1.0) - exp(-max(color, vec3<f32>(0.0)) * 1.3);
    return vec4<f32>(dither(color, input.frag_coord), input.colors[0].a);
}
