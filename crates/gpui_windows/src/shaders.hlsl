#include "alpha_correction.hlsl"

cbuffer GlobalParams: register(b0) {
    float4 gamma_ratios;
    float2 global_viewport_size;
    float grayscale_enhanced_contrast;
    float subpixel_enhanced_contrast;
    uint is_bgr;
    float depth_scale;
    uint2 global_pad;
};

cbuffer BatchParams: register(b1) {
    uint batch_start_index;
    uint3 batch_pad;
};

Texture2D<float4> t_sprite: register(t0);
SamplerState s_sprite: register(s0);

struct SubpixelSpriteFragmentOutput {
    float4 foreground : SV_Target0;
    float4 alpha : SV_Target1;
};

struct Bounds {
    float2 origin;
    float2 size;
};

struct Corners {
    float top_left;
    float top_right;
    float bottom_right;
    float bottom_left;
};

struct Edges {
    float top;
    float right;
    float bottom;
    float left;
};

struct Hsla {
    float h;
    float s;
    float l;
    float a;
};

struct LinearColorStop {
    Hsla color;
    float percentage;
};

struct Background {
    // 0u is Solid
    // 1u is LinearGradient
    // 2u is PatternSlash
    uint tag;
    // 0u is sRGB color
    // 1u is Oklab color
    uint color_space;
    Hsla solid;
    float gradient_angle_or_pattern_height;
    LinearColorStop colors[2];
    uint pad;
};

struct PreparedBackground {
    nointerpolation float4 color0: COLOR1;
    nointerpolation float4 color1: COLOR2;
    nointerpolation uint tag: TEXCOORD2;
};

struct AtlasTextureId {
    uint index;
    uint kind;
};

struct AtlasBounds {
    int2 origin;
    int2 size;
};

struct AtlasTile {
    AtlasTextureId texture_id;
    uint tile_id;
    uint padding;
    AtlasBounds bounds;
};

struct TransformationMatrix {
    float2x2 rotation_scale;
    float2 translation;
};

static const float M_PI_F = 3.141592653f;
// How much further along each edge a fully smoothed corner starts than a round one, as a share
// of its radius. The same as `SQUIRCLE_REACH` in gpui.
static const float SQUIRCLE_REACH = 0.528665f;
static const float3 GRAYSCALE_FACTORS = float3(0.2126f, 0.7152f, 0.0722f);

static const uint SIMPLE_PRIMITIVE_FLAG = 1u << 31;

float4 to_device_position_impl(float2 position) {
    float2 device_position = position / global_viewport_size * float2(2.0, -2.0) + float2(-1.0, 1.0);
    return float4(device_position, 0., 1.);
}

float4 to_device_position(float2 unit_vertex, Bounds bounds) {
    float2 position = unit_vertex * bounds.size + bounds.origin;
    return to_device_position_impl(position);
}

float quad_depth(uint quad_id) {
    return saturate(float(quad_id + 1u) * depth_scale);
}

static const float4 CULLED_POSITION = float4(0.0, 0.0, 0.0, 1.0);

float4 distance_from_clip_rect_impl(float2 position, Bounds clip_bounds) {
    float2 tl = position - clip_bounds.origin;
    float2 br = clip_bounds.origin + clip_bounds.size - position;
    return float4(tl.x, br.x, tl.y, br.y);
}

// The device pixels `bounds` reach into: the bounds grown out to whole pixels, so an edge that
// falls between pixels still draws the pixel it partly covers. Bounds on the pixel grid stay as
// they are.
Bounds pixel_cover(Bounds bounds) {
    Bounds cover;
    cover.origin = floor(bounds.origin);
    cover.size = ceil(bounds.origin + bounds.size) - cover.origin;
    return cover;
}

// How much of the pixel centred on `position` the bounds cover: 1 inside them, and the share
// left where an edge falls through the pixel.
float pixel_coverage(float2 position, Bounds bounds) {
    float2 near_edge = max(position - 0.5, bounds.origin);
    float2 far_edge = min(position + 0.5, bounds.origin + bounds.size);
    float2 overlap = saturate(far_edge - near_edge);
    return overlap.x * overlap.y;
}

float4 distance_from_clip_rect_transformed(float2 unit_vertex, Bounds bounds, Bounds clip_bounds, TransformationMatrix transformation) {
    float2 position = unit_vertex * bounds.size + bounds.origin;
    float2 transformed = mul(position, transformation.rotation_scale) + transformation.translation;
    return distance_from_clip_rect_impl(transformed, clip_bounds);
}

struct ClippedVertex {
    float2 position;
    // The corner as a fraction of the original bounds.
    float2 unit_vertex;
};

ClippedVertex clip_to_mask(float2 unit_vertex, Bounds bounds, Bounds mask) {
    float2 origin = max(bounds.origin, mask.origin);
    float2 corner = min(bounds.origin + bounds.size, mask.origin + mask.size);
    float2 size = max(corner - origin, float2(0.0, 0.0));

    ClippedVertex result;
    result.position = origin + unit_vertex * size;
    result.unit_vertex = (result.position - bounds.origin) / bounds.size;
    return result;
}

// Convert linear RGB to sRGB
float3 linear_to_srgb(float3 color) {
    return pow(color, float3(1.0 / 2.2, 1.0 / 2.2, 1.0 / 2.2));
}

// Convert sRGB to linear RGB
float3 srgb_to_linear(float3 color) {
    return pow(color, float3(2.2, 2.2, 2.2));
}

/// Hsla to sRGB conversion.
float4 hsla_to_rgba(Hsla hsla) {
    float h = hsla.h * 6.0; // Now, it's an angle but scaled in [0, 6) range
    float s = hsla.s;
    float l = hsla.l;
    float a = hsla.a;

    float c = (1.0 - abs(2.0 * l - 1.0)) * s;
    float x = c * (1.0 - abs(fmod(h, 2.0) - 1.0));
    float m = l - c / 2.0;

    float r = 0.0;
    float g = 0.0;
    float b = 0.0;

    if (h >= 0.0 && h < 1.0) {
        r = c;
        g = x;
        b = 0.0;
    } else if (h >= 1.0 && h < 2.0) {
        r = x;
        g = c;
        b = 0.0;
    } else if (h >= 2.0 && h < 3.0) {
        r = 0.0;
        g = c;
        b = x;
    } else if (h >= 3.0 && h < 4.0) {
        r = 0.0;
        g = x;
        b = c;
    } else if (h >= 4.0 && h < 5.0) {
        r = x;
        g = 0.0;
        b = c;
    } else {
        r = c;
        g = 0.0;
        b = x;
    }

    float4 rgba;
    rgba.x = (r + m);
    rgba.y = (g + m);
    rgba.z = (b + m);
    rgba.w = a;
    return rgba;
}

// Converts a sRGB color to the Oklab color space.
// Reference: https://bottosson.github.io/posts/oklab/#converting-from-linear-srgb-to-oklab
float4 srgb_to_oklab(float4 color) {
    // Convert non-linear sRGB to linear sRGB
    color = float4(srgb_to_linear(color.rgb), color.a);

    float l = 0.4122214708 * color.r + 0.5363325363 * color.g + 0.0514459929 * color.b;
    float m = 0.2119034982 * color.r + 0.6806995451 * color.g + 0.1073969566 * color.b;
    float s = 0.0883024619 * color.r + 0.2817188376 * color.g + 0.6299787005 * color.b;

    float l_ = pow(l, 1.0/3.0);
    float m_ = pow(m, 1.0/3.0);
    float s_ = pow(s, 1.0/3.0);

    return float4(
        0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
        1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
        0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_,
        color.a
    );
}

// Converts an Oklab color to the sRGB color space.
float4 oklab_to_srgb(float4 color) {
    float l_ = color.r + 0.3963377774 * color.g + 0.2158037573 * color.b;
    float m_ = color.r - 0.1055613458 * color.g - 0.0638541728 * color.b;
    float s_ = color.r - 0.0894841775 * color.g - 1.2914855480 * color.b;

    float l = l_ * l_ * l_;
    float m = m_ * m_ * m_;
    float s = s_ * s_ * s_;

    float3 linear_rgb = float3(
        4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
        -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
        -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s
    );

    // Convert linear sRGB to non-linear sRGB
    return float4(linear_to_srgb(linear_rgb), color.a);
}

float4 over(float4 below, float4 above) {
    float alpha = above.a + below.a * (1.0 - above.a);
    float3 premultiplied = above.rgb * above.a + below.rgb * below.a * (1.0 - above.a);
    return float4(premultiplied / (alpha == 0.0 ? 1.0 : alpha), alpha);
}

float2 to_tile_position(float2 unit_vertex, AtlasTile tile) {
    float2 atlas_size;
    t_sprite.GetDimensions(atlas_size.x, atlas_size.y);
    return (float2(tile.bounds.origin) + unit_vertex * float2(tile.bounds.size)) / atlas_size;
}

// Selects corner radius based on quadrant.
float pick_corner_radius(float2 center_to_point, Corners corner_radii) {
    if (center_to_point.x < 0.) {
        if (center_to_point.y < 0.) {
            return corner_radii.top_left;
        } else {
            return corner_radii.bottom_left;
        }
    } else {
        if (center_to_point.y < 0.) {
            return corner_radii.top_right;
        } else {
            return corner_radii.bottom_right;
        }
    }
}

float4 to_device_position_transformed(float2 unit_vertex, Bounds bounds,
                                      TransformationMatrix transformation) {
    float2 position = unit_vertex * bounds.size + bounds.origin;
    float2 transformed = mul(position, transformation.rotation_scale) + transformation.translation;
    float2 device_position = transformed / global_viewport_size * float2(2.0, -2.0) + float2(-1.0, 1.0);
    return float4(device_position, 0.0, 1.0);
}

// The shape of a corner of `radius` smoothed by `smoothing`: how far along each edge it starts,
// and the exponent of the superellipse it follows, 2 for a circular arc. The exponent is picked
// so that the corner crosses its diagonal where a round corner of `radius` would, which keeps
// it the same size to the eye. A corner with no room to reach further stays round.
float2 corner_shape(float radius, float smoothing, float2 half_size) {
    if (smoothing <= 0.0 || radius <= 0.0) {
        return float2(radius, 2.0);
    }
    float room = min(half_size.x, half_size.y);
    float extent = max(radius, min(radius * (1.0 + SQUIRCLE_REACH * smoothing), room));
    float inset = (1.0 - sqrt(0.5)) * radius / extent;
    return float2(extent, -1.0 / log2(1.0 - inset));
}

// Signed distance from a point, measured from a corner's center, to a superellipse corner of
// `radius` and `exponent`. The superellipse's norm divided by the length of its gradient is a
// true distance near the curve, which is where antialiasing reads it.
float superellipse_sdf(float2 pt, float radius, float exponent) {
    float2 component = abs(pt);
    float norm = pow(pow(component.x, exponent) + pow(component.y, exponent), 1.0 / exponent);
    if (norm <= 0.0) {
        return -radius;
    }
    float slope = length(pow(component / norm, float2(exponent - 1.0, exponent - 1.0)));
    return (norm - radius) / slope;
}

// Implementation of quad signed distance field
float quad_sdf_impl(float2 corner_center_to_point, float corner_radius, float corner_exponent) {
    if (corner_radius == 0.0) {
        // Fast path for unrounded corners
        return max(corner_center_to_point.x, corner_center_to_point.y);
    } else {
        // 0 inside the quad inset by corner_radius, and positive outside
        float2 outside = max(float2(0.0, 0.0), corner_center_to_point);
        // 0 outside the inset quad, and negative inside
        float inside = min(0.0, max(corner_center_to_point.x, corner_center_to_point.y));
        if (corner_exponent == 2.0) {
            return length(outside) + inside - corner_radius;
        }
        return superellipse_sdf(outside, corner_radius, corner_exponent) + inside;
    }
}

float quad_sdf(float2 pt, Bounds bounds, Corners corner_radii, float corner_smoothing) {
    float2 half_size = bounds.size / 2.;
    float2 center = bounds.origin + half_size;
    float2 center_to_point = pt - center;
    float2 corner = corner_shape(pick_corner_radius(center_to_point, corner_radii),
                                 corner_smoothing, half_size);
    float2 corner_to_point = abs(center_to_point) - half_size;
    float2 corner_center_to_point = corner_to_point + corner.x;
    return quad_sdf_impl(corner_center_to_point, corner.x, corner.y);
}

PreparedBackground prepare_background(Background background) {
    PreparedBackground output;
    output.tag = background.tag;
    output.color1 = float4(0.0, 0.0, 0.0, 0.0);

    if (background.tag == 1) {
        output.color0 = hsla_to_rgba(background.colors[0].color);
        output.color1 = hsla_to_rgba(background.colors[1].color);
        // Interpolating in Oklab means converting the endpoints once rather
        // than at every stop evaluation.
        if (background.color_space == 1) {
            output.color0 = srgb_to_oklab(output.color0);
            output.color1 = srgb_to_oklab(output.color1);
        }
    } else {
        output.color0 = hsla_to_rgba(background.solid);
    }

    return output;
}

// Ordered (Bayer 4x4) dither threshold, remapped to [-1, +1].
float dither_offset(float2 position) {
    uint2 cell = uint2(position) & 3u;
    uint z = cell.x ^ cell.y;
    uint threshold = ((z & 1u) << 3)
        | ((cell.y & 1u) << 2)
        | (z & 2u)
        | ((cell.y & 2u) >> 1);
    return (float(threshold) + 0.5) * (1.0 / 8.0) - 1.0;
}

float4 background_color(PreparedBackground prepared, Background background, Bounds bounds,
                        float2 position) {
    float4 color = prepared.color0;
    if (prepared.tag == 0u) {
        return color;
    }

    switch (prepared.tag) {
        case 1: {
            // -90 degrees to match the CSS gradient angle.
            float radians = (fmod(background.gradient_angle_or_pattern_height, 360.0) - 90.0)
                * (M_PI_F / 180.0);
            float2 direction = float2(cos(radians), sin(radians));

            // Expand the short side to be the same as the long side
            if (bounds.size.x > bounds.size.y) {
                direction.y *= bounds.size.y / bounds.size.x;
            } else {
                direction.x *= bounds.size.x / bounds.size.y;
            }

            float extent = abs(direction.x) > abs(direction.y)
                ? bounds.size.x
                : bounds.size.y;
            float direction_length = length(direction);
            float span = background.colors[1].percentage - background.colors[0].percentage;
            // The half-extent term is exactly half of `extent`, so the
            // position-independent part reduces to this.
            float offset = 0.5 - background.colors[0].percentage;

            // t = dot(position - center, gradient.xy) + gradient.z
            float3 gradient = float3(0.0, 0.0, 0.0);
            if (direction_length != 0.0 && extent != 0.0) {
                if (span == 0.0) {
                    const float STEP = 1e30;
                    gradient = float3(direction / (direction_length * extent) * STEP, offset * STEP);
                } else {
                    gradient = float3(direction / (direction_length * extent * span), offset / span);
                }
            }
            float2 center = bounds.origin + bounds.size * 0.5;
            float t = saturate(dot(position - center, gradient.xy) + gradient.z);
            color = lerp(prepared.color0, prepared.color1, t);
            if (background.color_space == 1) {
                color = oklab_to_srgb(color);
            }

            // Dither to reduce banding in gradients (especially dark/alpha).
            float dither = dither_offset(position);
            color.rgb += dither * 1.0 / 255.0;
            color.a   += dither * 1.5 / 255.0;
            break;
        }
        case 2: {
            float pattern_height_encoded = background.gradient_angle_or_pattern_height;
            float pattern_width = (pattern_height_encoded / 65535.0) / 255.0;
            float pattern_interval = fmod(pattern_height_encoded, 65535.0) / 255.0;
            float pattern_height = pattern_width + pattern_interval;
            // Only the rotated x coordinate is used, and a row-vector multiply
            // by the rotation reduces it to dot(v, float2(cos a, sin a)).
            float stripe_angle = M_PI_F / 4.0;
            float2 stripe_direction = float2(cos(stripe_angle), sin(stripe_angle));
            float pattern_period = pattern_height * sin(stripe_angle);
            float threshold = pattern_period * (pattern_width / pattern_height) / 2.0;
            float pattern = fmod(dot(position - bounds.origin, stripe_direction), pattern_period);
            float distance = min(pattern, pattern_period - pattern) - threshold;
            color.a *= saturate(0.5 - distance);
            break;
        }
        case 3: {
            // checkerboard
            float cell_size = background.gradient_angle_or_pattern_height;
            float2 relative_position = position - bounds.origin;
            float x_index = floor(relative_position.x / cell_size);
            float y_index = floor(relative_position.y / cell_size);
            color.a *= saturate((x_index + y_index) % 2.0);
            break;
        }
    }

    return color;
}

// Returns the dash velocity of a corner given the dash velocity of the two
// sides, by returning the slower velocity (larger dashes).
//
// Since 0 is used for dash velocity when the border width is 0 (instead of
// +inf), this returns the other dash velocity in that case.
//
// An alternative to this might be to appropriately interpolate the dash
// velocity around the corner, but that seems overcomplicated.
float corner_dash_velocity(float dv1, float dv2) {
    if (dv1 == 0.0) {
        return dv2;
    } else if (dv2 == 0.0) {
        return dv1;
    } else {
        return min(dv1, dv2);
    }
}

// Returns alpha used to render antialiased dashes.
// `t` is within the dash when `fmod(t, period) < length`.
float dash_alpha(
    float t, float period, float length, float dash_velocity,
    float antialias_threshold
) {
    float half_period = period / 2.0;
    float half_length = length / 2.0;
    // Value in [-half_period, half_period]
    // The dash is in [-half_length, half_length]
    float centered = fmod(t + half_period - half_length, period) - half_period;
    // Signed distance for the dash, negative values are inside the dash
    float signed_distance = abs(centered) - half_length;
    // Antialiased alpha based on the signed distance
    return saturate(antialias_threshold - signed_distance / dash_velocity);
}

// This approximates distance to the nearest point to a quarter ellipse in a way
// that is sufficient for anti-aliasing when the ellipse is not very eccentric.
// The components of `point` are expected to be positive.
//
// Negative on the outside and positive on the inside.
float quarter_ellipse_sdf(float2 pt, float2 radii) {
    // Scale the space to treat the ellipse like a unit circle
    float2 circle_vec = pt / radii;
    float unit_circle_sdf = length(circle_vec) - 1.0;
    // Approximate up-scaling of the length by using the average of the radii.
    //
    // TODO: A better solution would be to use the gradient of the implicit
    // function for an ellipse to approximate a scaling factor.
    return unit_circle_sdf * (radii.x + radii.y) * -0.5;
}

/*
**
**              Quads
**
*/

struct Quad {
    uint order;
    uint border_style;
    Bounds bounds;
    Bounds content_mask;
    Background background;
    Hsla border_color;
    Corners corner_radii;
    Edges border_widths;
    float corner_smoothing;
    uint pad;
};

bool has_rounded_corners(Corners corner_radii) {
    return corner_radii.top_left != 0.0
        || corner_radii.top_right != 0.0
        || corner_radii.bottom_right != 0.0
        || corner_radii.bottom_left != 0.0;
}

bool has_border(Edges border_widths) {
    return border_widths.top != 0.0
        || border_widths.right != 0.0
        || border_widths.bottom != 0.0
        || border_widths.left != 0.0;
}

bool is_shaped(Quad quad) {
    return has_rounded_corners(quad.corner_radii) || has_border(quad.border_widths);
}

struct QuadFragmentInput {
    nointerpolation uint quad_id: TEXCOORD0;
    float4 position: SV_Position;
    nointerpolation float4 border_color: COLOR0;
    PreparedBackground background;
};

// Opaque rectangles are drawn by the opaque pass instead of the quad pass. The fork draws no
// opaque pass (`DEPTH_PREPASS` in scene.rs), so the quad pass keeps every quad. Upstream's test is
// a solid opaque background with neither rounded corners nor a border.
bool is_opaque_rectangle(Quad quad) {
    return false;
}

StructuredBuffer<Quad> quads: register(t1);

QuadFragmentInput quad_vertex(uint vertex_id: SV_VertexID, uint instance_id: SV_InstanceID) {
    uint quad_id = batch_start_index + instance_id;
    Quad quad = quads[quad_id];

    if (is_opaque_rectangle(quad)) {
        QuadFragmentInput culled = (QuadFragmentInput)0;
        culled.position = CULLED_POSITION;
        return culled;
    }

    float2 unit_vertex = float2(float(vertex_id & 1u), 0.5 * float(vertex_id & 2u));
    Bounds cover = pixel_cover(quad.bounds);
    ClippedVertex vertex = clip_to_mask(unit_vertex, cover, quad.content_mask);
    float4 device_position = to_device_position_impl(vertex.position);

    float4 border_color = hsla_to_rgba(quad.border_color);

    QuadFragmentInput output;
    output.position = device_position;
    output.position.z = quad_depth(quad_id);
    output.border_color = border_color;
    // A quad off the pixel grid partly covers the pixels its edges fall through, so only one on
    // the grid fills every pixel it draws.
    bool on_grid = all(cover.origin == quad.bounds.origin) && all(cover.size == quad.bounds.size);
    bool simple = !is_shaped(quad) && quad.background.tag == 0u && on_grid;
    output.quad_id = quad_id | (simple ? SIMPLE_PRIMITIVE_FLAG : 0u);
    output.background = prepare_background(quad.background);
    return output;
}

float4 quad_fragment(QuadFragmentInput input): SV_Target {
    if ((input.quad_id & SIMPLE_PRIMITIVE_FLAG) != 0u) {
        return input.background.color0;
    }

    Quad quad = quads[input.quad_id];
    float4 background =
        background_color(input.background, quad.background, quad.bounds, input.position.xy);
    bool unrounded = !has_rounded_corners(quad.corner_radii);

    // A quad off the pixel grid partly covers the pixels its edges fall through. On the grid
    // this is 1 for every pixel drawn.
    float coverage = pixel_coverage(input.position.xy, quad.bounds);

    // Fast path when the quad is not rounded and doesn't have any border
    if (!is_shaped(quad)) {
        return background * float4(1.0, 1.0, 1.0, coverage);
    }

    float2 size = quad.bounds.size;
    float2 half_size = size / 2.;
    float2 the_point = input.position.xy - quad.bounds.origin;
    float2 center_to_point = the_point - half_size;

    // Signed distance field threshold for inclusion of pixels. 0.5 is the
    // minimum distance between the center of the pixel and the edge.
    const float antialias_threshold = 0.5;

    // How far along the edges the nearest corner reaches, and the exponent of its curve
    float2 corner = corner_shape(pick_corner_radius(center_to_point, quad.corner_radii),
                                 quad.corner_smoothing, half_size);
    float corner_radius = corner.x;

    float2 border = float2(
        center_to_point.x < 0.0 ? quad.border_widths.left : quad.border_widths.right,
        center_to_point.y < 0.0 ? quad.border_widths.top : quad.border_widths.bottom
    );

    // 0-width borders are reduced so that `inner_sdf >= antialias_threshold`.
    // The purpose of this is to not draw antialiasing pixels in this case.
    float2 reduced_border = float2(
        border.x == 0.0 ? -antialias_threshold : border.x,
        border.y == 0.0 ? -antialias_threshold : border.y
    );

    // Vector from the corner of the quad bounds to the point, after mirroring
    // the point into the bottom right quadrant. Both components are <= 0.
    float2 corner_to_point = abs(center_to_point) - half_size;

    // Vector from the point to the center of the rounded corner's circle, also
    // mirrored into bottom right quadrant.
    float2 corner_center_to_point = corner_to_point + corner_radius;

    // Whether the nearest point on the border is rounded
    bool is_near_rounded_corner =
        corner_center_to_point.x >= 0.0 &&
        corner_center_to_point.y >= 0.0;

    // Vector from straight border inner corner to point.
    //
    // 0-width borders are turned into width -1 so that inner_sdf is > 1.0 near
    // the border. Without this, antialiasing pixels would be drawn.
    float2 straight_border_inner_corner_to_point = corner_to_point + reduced_border;

    // Whether the point is beyond the inner edge of the straight border
    bool is_beyond_inner_straight_border =
        straight_border_inner_corner_to_point.x > 0.0 ||
        straight_border_inner_corner_to_point.y > 0.0;

    // Whether the point is far enough inside the quad, such that the pixels are
    // not affected by the straight border.
    bool is_within_inner_straight_border =
        straight_border_inner_corner_to_point.x < -antialias_threshold &&
        straight_border_inner_corner_to_point.y < -antialias_threshold;

    // Fast path for points that must be part of the background
    if (is_within_inner_straight_border && !is_near_rounded_corner) {
        return background * float4(1.0, 1.0, 1.0, coverage);
    }

    // Signed distance of the point to the outside edge of the quad's border
    float outer_sdf = quad_sdf_impl(corner_center_to_point, corner_radius, corner.y);

    // Approximate signed distance of the point to the inside edge of the quad's
    // border. It is negative outside this edge (within the border), and
    // positive inside.
    //
    // This is not always an accurate signed distance:
    // * The rounded portions with varying border width use an approximation of
    //   nearest-point-on-ellipse.
    // * When it is quickly known to be outside the edge, -1.0 is used.
    float inner_sdf = 0.0;
    if (corner_center_to_point.x <= 0.0 || corner_center_to_point.y <= 0.0) {
        // Fast paths for straight borders
        inner_sdf = -max(straight_border_inner_corner_to_point.x,
                        straight_border_inner_corner_to_point.y);
    } else if (is_beyond_inner_straight_border) {
        // Fast path for points that must be outside the inner edge
        inner_sdf = -1.0;
    } else if (reduced_border.x == reduced_border.y || corner.y != 2.0) {
        // Fast path for circular inner edge. A smoothed corner offsets its outer edge by the
        // wider border, since its inner edge has no exact form either way.
        inner_sdf = -(outer_sdf + max(reduced_border.x, reduced_border.y));
    } else {
        float2 ellipse_radii = max(float2(0.0, 0.0), float2(corner_radius, corner_radius) - reduced_border);
        inner_sdf = quarter_ellipse_sdf(corner_center_to_point, ellipse_radii);
    }

    // Negative when inside the border
    float border_sdf = max(inner_sdf, outer_sdf);

    float4 color = background;
    if (border_sdf < antialias_threshold) {
        float4 border_color = input.border_color;
        // Dashed border logic when border_style == 1
        if (quad.border_style == 1) {
            // Position along the perimeter in "dash space", where each dash
            // period has length 1
            float t = 0.0;

            // Total number of dash periods, so that the dash spacing can be
            // adjusted to evenly divide it
            float max_t = 0.0;

            // Border width is proportional to dash size. This is the behavior
            // used by browsers, but also avoids dashes from different segments
            // overlapping when dash size is smaller than the border width.
            //
            // Dash pattern: (2 * border width) dash, (1 * border width) gap
            const float dash_length_per_width = 2.0;
            const float dash_gap_per_width = 1.0;
            const float dash_period_per_width = dash_length_per_width + dash_gap_per_width;

            // Since the dash size is determined by border width, the density of
            // dashes varies. Multiplying a pixel distance by this returns a
            // position in dash space - it has units (dash period / pixels). So
            // a dash velocity of (1 / 10) is 1 dash every 10 pixels.
            float dash_velocity = 0.0;

            // Dividing this by the border width gives the dash velocity
            const float dv_numerator = 1.0 / dash_period_per_width;

            if (unrounded) {
                // When corners aren't rounded, the dashes are separately laid
                // out on each straight line, rather than around the whole
                // perimeter. This way each line starts and ends with a dash.
                bool is_horizontal = corner_center_to_point.x < corner_center_to_point.y;
                // Choosing the right border width for dashed borders.
                // TODO: A better solution exists taking a look at the whole file.
                // this does not fix single dashed borders at the corners
                float2 dashed_border = float2(
                    max(quad.border_widths.bottom, quad.border_widths.top),
                    max(quad.border_widths.right, quad.border_widths.left)
                );
                float border_width = is_horizontal ? dashed_border.x : dashed_border.y;
                dash_velocity = dv_numerator / border_width;
                t = is_horizontal ? the_point.x : the_point.y;
                t *= dash_velocity;
                max_t = is_horizontal ? size.x : size.y;
                max_t *= dash_velocity;
            } else {
                // When corners are rounded, the dashes are laid out clockwise
                // around the whole perimeter.

                float r_tr = quad.corner_radii.top_right;
                float r_br = quad.corner_radii.bottom_right;
                float r_bl = quad.corner_radii.bottom_left;
                float r_tl = quad.corner_radii.top_left;

                float w_t = quad.border_widths.top;
                float w_r = quad.border_widths.right;
                float w_b = quad.border_widths.bottom;
                float w_l = quad.border_widths.left;

                // Straight side dash velocities
                float dv_t = w_t <= 0.0 ? 0.0 : dv_numerator / w_t;
                float dv_r = w_r <= 0.0 ? 0.0 : dv_numerator / w_r;
                float dv_b = w_b <= 0.0 ? 0.0 : dv_numerator / w_b;
                float dv_l = w_l <= 0.0 ? 0.0 : dv_numerator / w_l;

                // Straight side lengths in dash space
                float s_t = (size.x - r_tl - r_tr) * dv_t;
                float s_r = (size.y - r_tr - r_br) * dv_r;
                float s_b = (size.x - r_br - r_bl) * dv_b;
                float s_l = (size.y - r_bl - r_tl) * dv_l;

                float corner_dash_velocity_tr = corner_dash_velocity(dv_t, dv_r);
                float corner_dash_velocity_br = corner_dash_velocity(dv_b, dv_r);
                float corner_dash_velocity_bl = corner_dash_velocity(dv_b, dv_l);
                float corner_dash_velocity_tl = corner_dash_velocity(dv_t, dv_l);

                // Corner lengths in dash space
                float c_tr = r_tr * (M_PI_F / 2.0) * corner_dash_velocity_tr;
                float c_br = r_br * (M_PI_F / 2.0) * corner_dash_velocity_br;
                float c_bl = r_bl * (M_PI_F / 2.0) * corner_dash_velocity_bl;
                float c_tl = r_tl * (M_PI_F / 2.0) * corner_dash_velocity_tl;

                // Cumulative dash space upto each segment
                float upto_tr = s_t;
                float upto_r = upto_tr + c_tr;
                float upto_br = upto_r + s_r;
                float upto_b = upto_br + c_br;
                float upto_bl = upto_b + s_b;
                float upto_l = upto_bl + c_bl;
                float upto_tl = upto_l + s_l;
                max_t = upto_tl + c_tl;

                if (is_near_rounded_corner) {
                    float radians = atan2(corner_center_to_point.y, corner_center_to_point.x);
                    float corner_t = radians * corner_radius;

                    if (center_to_point.x >= 0.0) {
                        if (center_to_point.y < 0.0) {
                            dash_velocity = corner_dash_velocity_tr;
                            // Subtracted because radians is pi/2 to 0 when
                            // going clockwise around the top right corner,
                            // since the y axis has been flipped
                            t = upto_r - corner_t * dash_velocity;
                        } else {
                            dash_velocity = corner_dash_velocity_br;
                            // Added because radians is 0 to pi/2 when going
                            // clockwise around the bottom-right corner
                            t = upto_br + corner_t * dash_velocity;
                        }
                    } else {
                        if (center_to_point.y >= 0.0) {
                            dash_velocity = corner_dash_velocity_bl;
                            // Subtracted because radians is pi/1 to 0 when
                            // going clockwise around the bottom-left corner,
                            // since the x axis has been flipped
                            t = upto_l - corner_t * dash_velocity;
                        } else {
                            dash_velocity = corner_dash_velocity_tl;
                            // Added because radians is 0 to pi/2 when going
                            // clockwise around the top-left corner, since both
                            // axis were flipped
                            t = upto_tl + corner_t * dash_velocity;
                        }
                    }
                } else {
                    // Straight borders
                    bool is_horizontal = corner_center_to_point.x < corner_center_to_point.y;
                    if (is_horizontal) {
                        if (center_to_point.y < 0.0) {
                            dash_velocity = dv_t;
                            t = (the_point.x - r_tl) * dash_velocity;
                        } else {
                            dash_velocity = dv_b;
                            t = upto_bl - (the_point.x - r_bl) * dash_velocity;
                        }
                    } else {
                        if (center_to_point.x < 0.0) {
                            dash_velocity = dv_l;
                            t = upto_tl - (the_point.y - r_tl) * dash_velocity;
                        } else {
                            dash_velocity = dv_r;
                            t = upto_r + (the_point.y - r_tr) * dash_velocity;
                        }
                    }
                }
            }
            float dash_length = dash_length_per_width / dash_period_per_width;
            float desired_dash_gap = dash_gap_per_width / dash_period_per_width;

            // Straight borders should start and end with a dash, so max_t is
            // reduced to cause this.
            max_t -= unrounded ? dash_length : 0.0;
            if (max_t >= 1.0) {
                // Adjust dash gap to evenly divide max_t
                float dash_count = floor(max_t);
                float dash_period = max_t / dash_count;
                border_color.a *= dash_alpha(t, dash_period, dash_length, dash_velocity, antialias_threshold);
            } else if (unrounded) {
                // When there isn't enough space for the full gap between the
                // two start / end dashes of a straight border, reduce gap to
                // make them fit.
                float dash_gap = max_t - dash_length;
                if (dash_gap > 0.0) {
                    float dash_period = dash_length + dash_gap;
                    border_color.a *= dash_alpha(t, dash_period, dash_length, dash_velocity, antialias_threshold);
                }
            }
        }

        border_color.a *= saturate(antialias_threshold - inner_sdf);
        color = over(background, border_color);
    }

    // The distance field knows nothing of how much of the pixel the quad covers, so a quad
    // thinner than a pixel would come out stronger than the share of the pixel it fills.
    return color * float4(1.0, 1.0, 1.0, min(coverage, saturate(antialias_threshold - outer_sdf)));
}

/*
**
**              Opaque quads
**
*/

struct OpaqueQuadFragmentInput {
    float4 position: SV_Position;
    nointerpolation float4 color: COLOR0;
};

OpaqueQuadFragmentInput opaque_quad_vertex(uint vertex_id: SV_VertexID, uint instance_id: SV_InstanceID) {
    uint quad_id = batch_start_index - instance_id;
    Quad quad = quads[quad_id];

    OpaqueQuadFragmentInput output;
    if (!is_opaque_rectangle(quad)) {
        output.position = CULLED_POSITION;
        output.color = float4(0.0, 0.0, 0.0, 0.0);
        return output;
    }

    float2 unit_vertex = float2(float(vertex_id & 1u), 0.5 * float(vertex_id & 2u));
    ClippedVertex vertex = clip_to_mask(unit_vertex, quad.bounds, quad.content_mask);

    output.position = to_device_position_impl(vertex.position);
    output.position.z = quad_depth(quad_id);
    output.color = hsla_to_rgba(quad.background.solid);
    return output;
}

float4 opaque_quad_fragment(OpaqueQuadFragmentInput input): SV_Target {
    return input.color;
}

/*
**
**              Shadows
**
*/

struct Shadow {
    uint order;
    float blur_radius;
    // The area the shadow paints: the blurred extent of a drop shadow, or the element
    // that an inset shadow is clipped to.
    Bounds bounds;
    Corners corner_radii;
    Bounds content_mask;
    Hsla color;
    // The shape that is blurred: the silhouette of a drop shadow, or the hole of an
    // inset shadow.
    Bounds shape_bounds;
    Corners shape_corner_radii;
    // 0 = drop shadow, 1 = inset shadow.
    uint inset;
    float element_corner_smoothing;
    // The element that clips the shadow: a drop shadow is drawn only outside it, an inset one
    // only inside it.
    Bounds element_bounds;
    Corners element_corner_radii;
};

struct PreparedBlurredRoundedRect {
    nointerpolation float4 shape: TEXCOORD1;
    nointerpolation float4 r1: TEXCOORD2;
    nointerpolation float4 exponent: TEXCOORD3;
    nointerpolation float4 scale: TEXCOORD4;
};

struct ShadowFragmentInput {
    nointerpolation uint shadow_id: TEXCOORD0;
    float4 position: SV_Position;
    nointerpolation float4 color: COLOR;
    PreparedBlurredRoundedRect blur;
};

StructuredBuffer<Shadow> shadows: register(t1);

float erf7(float x) {
    x = clamp(x, -10., 10.) * (2. / sqrt(M_PI_F));
    float xx = x * x;
    x = x + (0.24295 + (0.03395 + 0.0104 * xx) * xx) * (x * xx);
    return x / sqrt(1. + x * x);
}

// Raph Levien's blurred rounded rectangle approximation, whose blur s is sigma * sqrt(2):
// https://raphlinus.github.io/graphics/2020/04/21/blurred-rounded-rects.html
PreparedBlurredRoundedRect prepare_blurred_rounded_rect(float2 size, Corners corner_radii, float sigma) {
    float s = max(sigma * sqrt(2.), 1e-6);
    float s_inv = 1. / s;
    float min_edge = min(size.x, size.y);
    float r_max = 0.5 * min_edge;
    float4 radii = float4(corner_radii.top_left, corner_radii.top_right,
                          corner_radii.bottom_right, corner_radii.bottom_left);
    float4 r0 = min(sqrt(radii * radii + 1.15 * 1.15 * s * s), r_max);
    float4 r1 = min(sqrt(radii * radii + 4. * s * s), r_max);

    float2 half_extent = 0.5 * s_inv * size;
    float delta = 1.25 * s * (exp(-half_extent.x * half_extent.x) - exp(-half_extent.y * half_extent.y));
    float2 adjusted_size = float2(size.x + min(delta, 0.), size.y - max(delta, 0.));
    float4 peak = s_inv * 0.5 * (max(adjusted_size.x, adjusted_size.y) - 0.5 * radii);

    PreparedBlurredRoundedRect output;
    output.shape = float4(0.5 * adjusted_size, s_inv, min_edge);
    output.r1 = r1;
    output.exponent = 2. * r1 / r0;
    output.scale = 0.5 * float4(erf7(peak.x), erf7(peak.y), erf7(peak.z), erf7(peak.w));
    return output;
}

float pick_corner(float2 center_to_point, float4 values) {
    Corners corners = { values.x, values.y, values.z, values.w };
    return pick_corner_radius(center_to_point, corners);
}

float blurred_rounded_rect(float2 center_to_point, PreparedBlurredRoundedRect prepared) {
    float r1 = pick_corner(center_to_point, prepared.r1);
    float exponent = pick_corner(center_to_point, prepared.exponent);
    float scale = pick_corner(center_to_point, prepared.scale);
    float s_inv = prepared.shape.z;
    float min_edge = prepared.shape.w;

    float2 p0 = abs(center_to_point) - (prepared.shape.xy - r1);
    float2 p1 = max(p0, 0.);
    float d_pos;
    [branch] if (p1.x > 0. && p1.y > 0.) {
        d_pos = pow(pow(p1.x, exponent) + pow(p1.y, exponent), 1. / exponent);
    } else {
        d_pos = p1.x + p1.y;
    }

    float d_neg = min(max(p0.x, p0.y), 0.);
    float d = d_pos + d_neg - r1;
    return scale * (erf7(s_inv * (min_edge + d)) - erf7(s_inv * d));
}

ShadowFragmentInput shadow_vertex(uint vertex_id: SV_VertexID, uint instance_id: SV_InstanceID) {
    float2 unit_vertex = float2(float(vertex_id & 1u), 0.5 * float(vertex_id & 2u));
    uint shadow_id = batch_start_index + instance_id;
    Shadow shadow = shadows[shadow_id];

    ClippedVertex vertex = clip_to_mask(unit_vertex, shadow.bounds, shadow.content_mask);
    float4 device_position = to_device_position_impl(vertex.position);
    float4 color = hsla_to_rgba(shadow.color);

    ShadowFragmentInput output;
    output.position = device_position;
    output.color = color;
    output.shadow_id = shadow_id;
    output.blur = prepare_blurred_rounded_rect(shadow.shape_bounds.size, shadow.shape_corner_radii, shadow.blur_radius);

    return output;
}

float4 shadow_fragment(ShadowFragmentInput input): SV_TARGET {
    Shadow shadow = shadows[input.shadow_id];

    float alpha;
    if (shadow.blur_radius == 0.) {
        float distance = quad_sdf(input.position.xy, shadow.shape_bounds, shadow.shape_corner_radii, 0.0);
        alpha = saturate(0.5 - distance);
    } else {
        float2 center = shadow.shape_bounds.origin + shadow.shape_bounds.size / 2.;
        alpha = blurred_rounded_rect(input.position.xy - center, input.blur);
    }

    if (shadow.inset != 0u) {
        // The inset shadow is the complement of the (blurred) hole rect, clipped to the element.
        // `saturate(0.5 - d)` gives a 1-pixel antialiased edge: d <= -0.5 -> 1, d >= 0.5 -> 0.
        alpha = 1.0 - alpha;
        float element_distance = quad_sdf(input.position.xy, shadow.element_bounds,
                                          shadow.element_corner_radii, shadow.element_corner_smoothing);
        alpha *= saturate(0.5 - element_distance);
    } else {
        // A drop shadow is clipped to outside the element, the way CSS draws one, so an
        // element with a translucent fill never shows its own shadow through it.
        float element_distance = quad_sdf(input.position.xy, shadow.element_bounds,
                                          shadow.element_corner_radii, shadow.element_corner_smoothing);
        alpha *= saturate(0.5 + element_distance);
    }

    return input.color * float4(1., 1., 1., alpha);
}

/*
**
**              Path Tiles
**
*/

struct PathCurve {
    float2 p0;
    float2 p1;
    float2 a;
    float2 b;
    float sign;
    uint pad;
};

struct PathTile {
    uint paint;
    uint curve_start;
    uint curve_count;
    int backdrop;
    float2 top_left;
    float2 bottom_right;
    float2 reference;
    float2 curves_top_left;
    float2 curves_bottom_right;
};

struct PathPaint {
    Bounds bounds;
    Bounds content_mask;
    Background color;
    uint even_odd;
    uint pad;
};

StructuredBuffer<PathTile> path_tiles: register(t1);
StructuredBuffer<PathCurve> path_curves: register(t2);
StructuredBuffer<PathPaint> path_paints: register(t3);

struct PathTileFragmentInput {
    float4 position: SV_Position;
    nointerpolation uint tile_id: TEXCOORD0;
    nointerpolation float4 curve_box: TEXCOORD1;
    PreparedBackground background;
};

PathTileFragmentInput path_tile_vertex(uint vertex_id: SV_VertexID, uint tile_id: SV_InstanceID) {
    uint tile_index = batch_start_index + tile_id;
    PathTile tile = path_tiles[tile_index];
    PathPaint paint = path_paints[tile.paint];

    // Neighboring tiles store their shared edge as the same number, so clipping each corner on its
    // own gives them bit-identical edges, which the rasterizer neither overlaps nor leaves gaps
    // between. Interpolating from an origin and a size would not.
    float2 mask_top_left = paint.content_mask.origin;
    float2 mask_bottom_right = paint.content_mask.origin + paint.content_mask.size;
    float2 top_left = clamp(tile.top_left, mask_top_left, mask_bottom_right);
    float2 bottom_right = clamp(tile.bottom_right, top_left, mask_bottom_right);
    float2 position = float2(
        (vertex_id & 1u) != 0u ? bottom_right.x : top_left.x,
        (vertex_id & 2u) != 0u ? bottom_right.y : top_left.y);

    PathTileFragmentInput output;
    output.position = to_device_position_impl(position);
    output.tile_id = tile_index;
    output.curve_box = float4(tile.curves_top_left, tile.curves_bottom_right);
    output.background = prepare_background(paint.color);
    return output;
}

// Solves a*t^2 + b*t + c = 0 for the root in [0, 1] on a monotone piece.
float monotone_quadratic_root(float a, float b, float c, float direction) {
    float discriminant = max(b * b - 4.0 * a * c, 0.0);
    float denominator = b + direction * sqrt(discriminant);
    if (denominator == 0.0) {
        return 0.0;
    }
    return saturate(-2.0 * c / denominator);
}

// Coverage estimate adapted from https://github.com/EricLengyel/Slug/blob/be3c13eb7d63f9e8aa5c583e42d92c374cb91d98/SlugPixelShader.hlsl
// Copyright (c) 2017 Eric Lengyel.
// Licensed under the MIT license.

// Accumulates the curve's winding along the pixel's leftward and upward rays. Like Slug, a crossing
// within half a pixel of the center counts fractionally.
void add_curve_winding(PathCurve curve, float2 center, inout float2 winding, inout float2 weight) {
    float min_x = min(curve.p0.x, curve.p1.x);
    float max_x = max(curve.p0.x, curve.p1.x);

    [branch]
    if (curve.p0.y <= center.y && center.y < curve.p1.y) {
        [branch]
        if (max_x <= center.x - 0.5) {
            winding.x += curve.sign;
        } else {
            float t = monotone_quadratic_root(curve.a.y, curve.b.y, curve.p0.y - center.y, 1.0);
            float x = (curve.a.x * t + curve.b.x) * t + curve.p0.x;
            float distance = center.x - x;
            winding.x += curve.sign * saturate(distance + 0.5);
            weight.x = max(weight.x, saturate(1.0 - 2.0 * abs(distance)));
        }
    }

    [branch]
    if (min_x <= center.x && center.x < max_x) {
        float direction = curve.p1.x > curve.p0.x ? 1.0 : -1.0;
        [branch]
        if (curve.p1.y <= center.y - 0.5) {
            winding.y -= curve.sign * direction;
        } else if (curve.p0.y < center.y + 0.5) {
            float t = monotone_quadratic_root(curve.a.x, curve.b.x, curve.p0.x - center.x, direction);
            float y = (curve.a.y * t + curve.b.y) * t + curve.p0.y;
            float distance = center.y - y;
            winding.y -= curve.sign * direction * saturate(distance + 0.5);
            weight.y = max(weight.y, saturate(1.0 - 2.0 * abs(distance)));
        }
    }
}

// The backdrop gives the unlisted curves' winding along a leftward ray from the tile's reference
// corner. We need it at the current pixel, along both leftward and upward rays. Because paths are
// closed, chains of unlisted curves can only begin or end where a listed curve does. So, we can
// compute the winding at the current pixel by correcting the backdrop for each listed endpoint
// between its ray and the pixel.
void correct_for_endpoint(float2 endpoint, float step, float2 reference, float2 center,
                          inout float2 winding) {
    if (endpoint.x <= reference.x) {
        winding.x -= step * (float(endpoint.y <= center.y) - float(endpoint.y <= reference.y));
    }
    if (endpoint.y <= reference.y && endpoint.x <= center.x) {
        winding.y += step;
    }
}

float4 path_tile_fragment(PathTileFragmentInput input): SV_Target {
    float2 center = floor(input.position.xy) + 0.5;
    if (input.background.tag == 0u
        && (any(center < input.curve_box.xy) || any(center > input.curve_box.zw))) {
        return input.background.color0;
    }
    PathTile tile = path_tiles[input.tile_id];
    PathPaint paint = path_paints[tile.paint];

    float2 winding = float(tile.backdrop);
    float2 weight = float2(0.0, 0.0);
    [loop]
    for (uint i = 0u; i < tile.curve_count; i++) {
        PathCurve curve = path_curves[tile.curve_start + i];
        if (min(curve.p0.x, curve.p1.x) >= center.x + 0.5) {
            break;
        }
        if (curve.p0.y < center.y + 0.5) {
            add_curve_winding(curve, center, winding, weight);
            correct_for_endpoint(curve.p0, curve.sign, tile.reference, center, winding);
            correct_for_endpoint(curve.p1, -curve.sign, tile.reference, center, winding);
        }
    }

    // Weight the winding computed by each ray by its nearest crossing's distance from the pixel center,
    // using Slug's technique.
    float coverage = max(
        abs(dot(winding, weight)) / max(weight.x + weight.y, 1.0 / 65536.0),
        min(abs(winding.x), abs(winding.y)));
    coverage = paint.even_odd != 0u
        ? 1.0 - abs(1.0 - frac(coverage * 0.5) * 2.0)
        : saturate(coverage);

    if (coverage == 0.0) {
        return float4(0.0, 0.0, 0.0, 0.0);
    }

    float4 color =
        background_color(input.background, paint.color, paint.bounds, input.position.xy);
    return float4(color.rgb, color.a * coverage);
}

/*
**
**              Underlines
**
*/

struct Underline {
    uint order;
    uint pad;
    Bounds bounds;
    Bounds content_mask;
    Hsla color;
    float thickness;
    uint wavy;
};

struct UnderlineFragmentInput {
  nointerpolation uint underline_id: TEXCOORD0;
  float4 position: SV_Position;
  nointerpolation float4 color: COLOR;
};

StructuredBuffer<Underline> underlines: register(t1);

UnderlineFragmentInput underline_vertex(uint vertex_id: SV_VertexID, uint instance_id: SV_InstanceID) {
    float2 unit_vertex = float2(float(vertex_id & 1u), 0.5 * float(vertex_id & 2u));
    uint underline_id = batch_start_index + instance_id;
    Underline underline = underlines[underline_id];
    ClippedVertex vertex = clip_to_mask(unit_vertex, underline.bounds, underline.content_mask);
    float4 device_position = to_device_position_impl(vertex.position);
    float4 color = hsla_to_rgba(underline.color);

    UnderlineFragmentInput output;
    output.position = device_position;
    output.color = color;
    output.underline_id = underline_id | (underline.wavy == 0u ? SIMPLE_PRIMITIVE_FLAG : 0u);
    return output;
}

float4 underline_fragment(UnderlineFragmentInput input): SV_Target {
    const float WAVE_FREQUENCY = 2.0;
    const float WAVE_HEIGHT_RATIO = 0.8;

    if ((input.underline_id & SIMPLE_PRIMITIVE_FLAG) != 0u) {
        return input.color;
    }

    Underline underline = underlines[input.underline_id];
    float half_thickness = underline.thickness * 0.5;
    float2 origin = underline.bounds.origin;

    float2 st = ((input.position.xy - origin) / underline.bounds.size.y) - float2(0., 0.5);
    float frequency = (M_PI_F * WAVE_FREQUENCY * underline.thickness) / underline.bounds.size.y;
    float amplitude = (underline.thickness * WAVE_HEIGHT_RATIO) / underline.bounds.size.y;

    float sine = sin(st.x * frequency) * amplitude;
    float dSine = cos(st.x * frequency) * amplitude * frequency;
    float distance = (st.y - sine) / sqrt(1. + dSine * dSine);
    float distance_in_pixels = distance * underline.bounds.size.y;
    float distance_from_top_border = distance_in_pixels - half_thickness;
    float distance_from_bottom_border = distance_in_pixels + half_thickness;
    float alpha = saturate(
        0.5 - max(-distance_from_bottom_border, distance_from_top_border));
    return input.color * float4(1., 1., 1., alpha);
}

/*
**
**              Monochrome sprites
**
*/

struct MonochromeSprite {
    uint order;
    uint pad;
    AtlasTile tile;
    Bounds bounds;
    Bounds content_mask;
    Hsla color;
    TransformationMatrix transformation;
};

struct MonochromeSpriteFragmentInput {
    float4 position: SV_Position;
    float2 tile_position: POSITION;
    nointerpolation float4 color: COLOR;
    float4 clip_distance: SV_ClipDistance;
};

StructuredBuffer<MonochromeSprite> mono_sprites: register(t1);

MonochromeSpriteFragmentInput monochrome_sprite_vertex(uint vertex_id: SV_VertexID, uint instance_id: SV_InstanceID) {
    float2 unit_vertex = float2(float(vertex_id & 1u), 0.5 * float(vertex_id & 2u));
    uint sprite_id = batch_start_index + instance_id;
    MonochromeSprite sprite = mono_sprites[sprite_id];
    float4 device_position =
        to_device_position_transformed(unit_vertex, sprite.bounds, sprite.transformation);
    float4 clip_distance = distance_from_clip_rect_transformed(unit_vertex, sprite.bounds, sprite.content_mask, sprite.transformation);
    float2 tile_position = to_tile_position(unit_vertex, sprite.tile);
    float4 color = hsla_to_rgba(sprite.color);

    MonochromeSpriteFragmentInput output;
    output.position = device_position;
    output.tile_position = tile_position;
    output.color = color;
    output.clip_distance = clip_distance;
    return output;
}

float4 monochrome_sprite_fragment(MonochromeSpriteFragmentInput input): SV_Target {
    float sample = t_sprite.Sample(s_sprite, input.tile_position).r;
    float alpha_corrected = apply_contrast_and_gamma_correction(sample, input.color.rgb, grayscale_enhanced_contrast, gamma_ratios);
    return float4(input.color.rgb, input.color.a * alpha_corrected);
}

MonochromeSpriteFragmentInput subpixel_sprite_vertex(uint vertex_id: SV_VertexID, uint instance_id: SV_InstanceID) {
    return monochrome_sprite_vertex(vertex_id, instance_id);
}

SubpixelSpriteFragmentOutput subpixel_sprite_fragment(MonochromeSpriteFragmentInput input) {
    float3 sample = t_sprite.Sample(s_sprite, input.tile_position).rgb;
    if (is_bgr) {
        sample = sample.bgr;
    }
    float3 alpha_corrected = apply_contrast_and_gamma_correction3(sample, input.color.rgb, subpixel_enhanced_contrast, gamma_ratios);

    SubpixelSpriteFragmentOutput output;
    output.foreground = float4(input.color.rgb, 1.0f);
    output.alpha = float4(input.color.a * alpha_corrected, 1.0f);
    return output;
}

// Drawn from `subpixel_sprite_vertex`, so this half has no vertex entry point of its own.
//
// `subpixel_sprite_fragment`'s dual-source blend always writes alpha = 1, which is
// fine on the opaque frame but wrong in an offscreen filter layer: those start
// transparent, and get composited onto their parent by that alpha afterward, so an
// antialiased glyph pixel would overwrite the parent instead of blending into it.
// This collapses ClearType's three channels into one coverage value and writes a
// normal premultiplied color instead, trading subpixel fringing for a correct blend.
float4 subpixel_sprite_layered_fragment(MonochromeSpriteFragmentInput input): SV_Target {
    float3 sample = t_sprite.Sample(s_sprite, input.tile_position).rgb;
    if (is_bgr) {
        sample = sample.bgr;
    }
    float3 alpha_corrected = apply_contrast_and_gamma_correction3(sample, input.color.rgb, subpixel_enhanced_contrast, gamma_ratios);
    float average_coverage = (alpha_corrected.r + alpha_corrected.g + alpha_corrected.b) / 3.0f;
    float coverage = input.color.a * average_coverage;
    return float4(input.color.rgb * coverage, coverage);
}

/*
**
**              Polychrome sprites
**
*/

struct PolychromeSprite {
    uint order;
    uint pad;
    AtlasTile tile;
    uint grayscale;
    float opacity;
    Bounds bounds;
    Bounds content_mask;
    Corners corner_radii;
};

struct PolychromeSpriteFragmentInput {
    nointerpolation uint sprite_id: TEXCOORD0;
    float4 position: SV_Position;
    float2 tile_position: POSITION;
    nointerpolation float opacity: TEXCOORD1;
};

StructuredBuffer<PolychromeSprite> poly_sprites: register(t1);

PolychromeSpriteFragmentInput polychrome_sprite_vertex(uint vertex_id: SV_VertexID, uint instance_id: SV_InstanceID) {
    float2 unit_vertex = float2(float(vertex_id & 1u), 0.5 * float(vertex_id & 2u));
    uint sprite_id = batch_start_index + instance_id;
    PolychromeSprite sprite = poly_sprites[sprite_id];
    ClippedVertex vertex = clip_to_mask(unit_vertex, sprite.bounds, sprite.content_mask);
    float4 device_position = to_device_position_impl(vertex.position);
    float2 tile_position = to_tile_position(vertex.unit_vertex, sprite.tile);

    PolychromeSpriteFragmentInput output;
    output.position = device_position;
    output.tile_position = tile_position;
    bool simple = !has_rounded_corners(sprite.corner_radii) && sprite.grayscale == 0u;
    output.sprite_id = sprite_id | (simple ? SIMPLE_PRIMITIVE_FLAG : 0u);
    output.opacity = sprite.opacity;
    return output;
}

float4 polychrome_sprite_fragment(PolychromeSpriteFragmentInput input): SV_Target {
    float4 color = t_sprite.Sample(s_sprite, input.tile_position);
    if ((input.sprite_id & SIMPLE_PRIMITIVE_FLAG) != 0u) {
        return float4(color.rgb, color.a * input.opacity);
    }

    PolychromeSprite sprite = poly_sprites[input.sprite_id];
    if (sprite.grayscale != 0u) {
        color.rgb = dot(color.rgb, GRAYSCALE_FACTORS);
    }
    float distance = quad_sdf(input.position.xy, sprite.bounds, sprite.corner_radii, 0.0);
    color.a *= input.opacity * saturate(0.5 - distance);
    return color;
}

/*
**
**              Backdrops
**
*/

struct Backdrop {
    uint order;
    float corner_smoothing;
    float blur;
    float opacity;
    Bounds bounds;
    Bounds content_mask;
    Corners corner_radii;
};

struct BackdropFragmentInput {
    float4 position: SV_Position;
    nointerpolation uint backdrop_id: TEXCOORD0;
};

StructuredBuffer<Backdrop> backdrops: register(t1);

BackdropFragmentInput backdrop_vertex(uint vertex_id: SV_VertexID, uint instance_id: SV_InstanceID) {
    float2 unit_vertex = float2(float(vertex_id & 1u), 0.5 * float(vertex_id & 2u));
    uint backdrop_id = batch_start_index + instance_id;
    Backdrop backdrop = backdrops[backdrop_id];
    ClippedVertex vertex = clip_to_mask(unit_vertex, backdrop.bounds, backdrop.content_mask);

    BackdropFragmentInput output;
    output.position = to_device_position_impl(vertex.position);
    output.backdrop_id = backdrop_id;
    return output;
}

// A backdrop replaces what it covers with the blurred copy of it rather than laying that copy
// over it: `dst = mix(dst, blurred, k)`, k being the coverage times the instance's opacity.
// Over an opaque frame the two are one picture, since a copy of alpha one hides what it
// covers; over a see-through window laying it over left `a + a(1 - a)` where the window asked
// for `a`. The mix is two draws over one instance buffer: `backdrop_punch_fragment` for
// `dst *= (1 - k)`, then this one, added on top.
float4 backdrop_fragment(BackdropFragmentInput input): SV_Target {
    Backdrop backdrop = backdrops[input.backdrop_id];
    float2 position = input.position.xy;
    float4 sampled = t_sprite.Sample(s_sprite, position / global_viewport_size);
    float distance = quad_sdf(position, backdrop.bounds, backdrop.corner_radii, backdrop.corner_smoothing);
    float coverage = saturate(0.5 - distance);

    return sampled * coverage * backdrop.opacity;
}

// The hole the blurred copy lands in: alpha alone, blended so what it covers keeps `1 - k` of
// itself, colour and alpha together.
float4 backdrop_punch_fragment(BackdropFragmentInput input): SV_Target {
    Backdrop backdrop = backdrops[input.backdrop_id];
    float2 position = input.position.xy;
    float distance = quad_sdf(position, backdrop.bounds, backdrop.corner_radii, backdrop.corner_smoothing);
    float coverage = saturate(0.5 - distance);

    return float4(0.0, 0.0, 0.0, coverage * backdrop.opacity);
}

/*
**
**              Layer filters
**
*/

struct FilterParams {
    Bounds source_bounds;
    Bounds fade_bounds;
    float2 direction;
    float2 transform_origin;
    float radius;
    float scale;
    float fade_top;
    float fade_bottom;
    float fade_left;
    float fade_right;
    float2 translation;
    // How many texels of the texture a blur pass reads hold the frame.
    float2 extent;
};

struct FilterVertexOutput {
    float4 position: SV_Position;
    float2 uv: TEXCOORD0;
};

struct FilterFragmentInput {
    float4 position: SV_Position;
    float2 uv: TEXCOORD0;
};

StructuredBuffer<FilterParams> filter_params: register(t1);

FilterVertexOutput filter_vertex_impl(uint vertex_id) {
    float2 unit_vertex = float2(float(vertex_id & 1u), 0.5 * float(vertex_id & 2u));

    FilterVertexOutput output;
    output.position = float4(unit_vertex.x * 2.0 - 1.0, 1.0 - unit_vertex.y * 2.0, 0.0, 1.0);
    output.uv = unit_vertex;
    return output;
}

FilterVertexOutput gaussian_vertex(uint vertex_id: SV_VertexID) {
    return filter_vertex_impl(vertex_id);
}

FilterVertexOutput pyramid_down_vertex(uint vertex_id: SV_VertexID) {
    return filter_vertex_impl(vertex_id);
}

FilterVertexOutput blit_vertex(uint vertex_id: SV_VertexID) {
    return filter_vertex_impl(vertex_id);
}

FilterVertexOutput mask_vertex(uint vertex_id: SV_VertexID) {
    return filter_vertex_impl(vertex_id);
}

// How much of a texel wide footprint centred at `at`, in texels, lies within `0..extent` on both
// axes. Nothing lies past the frame's edges, so every blur pass weighs its taps by this and
// divides by the weight it kept, with `extent` as how many texels of the texture it reads hold
// the frame. A blur against an edge then averages what is there instead of stretching the border
// row across the kernel, which made every row that scrolled out under the edge flash through the
// blur on its way.
float blur_inside(float2 at, float2 extent) {
    float2 span = saturate(min(at + 0.5, extent) - max(at - 0.5, float2(0.0, 0.0)));
    return span.x * span.y;
}

// How much of a texel wide span centred at `at`, in texels, lies within `0..limit` along one
// axis. A gaussian pass reads along one axis only, so a texel's coverage across it is the same
// for every tap and must not weigh the taps against the centre.
float blur_along(float at, float limit) {
    return saturate(min(at + 0.5, limit) - max(at - 0.5, 0.0));
}

// A narrow blur runs `gaussian_fragment` across and then down at full resolution, with `radius`
// as its standard deviation, and a wide one runs it across and down the deepest level of a
// pyramid. It pairs neighbouring taps into one bilinear fetch placed between them by their
// weights, which halves the fetches, and relies on sampling a texture of the same size as the
// target, so every pair lands between two texel centres along the axis. Every tap, the centre
// too, counts by how much of it lies inside the frame along the axis, so a tap past the frame's
// edge drops out of the pair it belongs to.
float4 gaussian_fragment(FilterFragmentInput input): SV_Target {
    FilterParams params = filter_params[batch_start_index];
    float width;
    float height;
    t_sprite.GetDimensions(width, height);
    float2 size = float2(width, height);
    float2 texel = params.direction / size;
    float axis = dot(input.uv * size, params.direction);
    float limit = dot(params.extent, params.direction);
    float sigma = max(params.radius, 0.0001);
    float reach = min(ceil(sigma * 3.0), 24.0);
    float spread = 2.0 * sigma * sigma;

    float own = blur_along(axis, limit);
    float4 total = own * t_sprite.Sample(s_sprite, input.uv);
    float weight = own;
    for (float inner = 1.0; inner <= reach; inner += 2.0) {
        float outer = inner + 1.0;
        float inner_weight = exp(-inner * inner / spread);
        float outer_weight = outer <= reach ? exp(-outer * outer / spread) : 0.0;
        [unroll]
        for (int i = 0; i < 2; i++) {
            float side = i == 0 ? -1.0 : 1.0;
            float inner_tap = inner_weight * blur_along(axis + inner * side, limit);
            float outer_tap = outer_weight * blur_along(axis + outer * side, limit);
            float pair = inner_tap + outer_tap;
            float2 at = texel * (side * (inner * inner_tap + outer * outer_tap) / max(pair, 0.0001));
            total += pair * t_sprite.Sample(s_sprite, input.uv + at);
            weight += pair;
        }
    }

    return total / max(weight, 0.0001);
}

// Four bilinear taps at `centre` plus `offset` mirrored into each quadrant, in texels of the
// texture read, each weighed by how much of it falls within `extent`.
float4 pyramid_taps(float2 centre, float2 offset, float2 extent) {
    float width;
    float height;
    t_sprite.GetDimensions(width, height);
    float2 size = float2(width, height);
    float4 total = float4(0.0, 0.0, 0.0, 0.0);
    float weight = 0.0;
    [unroll]
    for (int i = 0; i < 4; i++) {
        float2 corner = float2((i & 1) ? 1.0 : -1.0, (i & 2) ? 1.0 : -1.0);
        float2 at = centre + offset * corner;
        float tap = blur_inside(at, extent);
        total += tap * t_sprite.Sample(s_sprite, at / size);
        weight += tap;
    }
    return total / max(weight, 0.0001);
}

// Halves the level it reads, the first passes of a pyramid. A target texel sits over the corner
// of four source texels, and a bilinear tap on each of the four corners around that one averages
// the four by four texels under it.
float4 pyramid_down_fragment(FilterFragmentInput input): SV_Target {
    FilterParams params = filter_params[batch_start_index];
    return pyramid_taps(input.position.xy * 2.0, float2(1.0, 1.0), params.extent);
}

// Doubles the level it reads, back towards full size. Four bilinear taps half a texel out on
// each diagonal lay a box of one texel over the bilinear tent, which rounds off the corners a
// tent alone leaves at every texel of the level below and keeps its grid from showing.
float4 pyramid_up_fragment(FilterFragmentInput input): SV_Target {
    FilterParams params = filter_params[batch_start_index];
    return pyramid_taps(input.position.xy * 0.5, float2(0.5, 0.5), params.extent);
}


float4 blit_fragment(FilterFragmentInput input): SV_Target {
    return t_sprite.Sample(s_sprite, input.uv);
}

// Where a point of a layer's source lands in the frame, applying the layer's transform.
float2 mask_destination(FilterParams params, float2 source_point) {
    return params.transform_origin + (source_point - params.transform_origin) * params.scale
        + params.translation;
}

float4 mask_fragment(FilterFragmentInput input): SV_Target {
    FilterParams params = filter_params[batch_start_index];
    float2 destination_point = input.position.xy;
    float2 source_point = params.transform_origin
        + (destination_point - params.translation - params.transform_origin) / params.scale;
    // Only the part of the source inside the frame was drawn, so only that part covers pixels.
    float2 source_start = max(params.source_bounds.origin, float2(0.0, 0.0));
    float2 source_end = min(params.source_bounds.origin + params.source_bounds.size,
        global_viewport_size);

    // An untransformed layer keeps the frame's pixel grid, so a pixel is in or out. A
    // transformed one covers each pixel as far as its transformed bounds overlap it, which
    // smooths the edge the clamped sample below leaves hard.
    float coverage = 1.0;
    if (params.scale == 1.0 && all(params.translation == float2(0.0, 0.0))) {
        if (any(source_point < source_start) || any(source_point >= source_end)) {
            return float4(0.0, 0.0, 0.0, 0.0);
        }
    } else {
        float2 start = mask_destination(params, source_start);
        float2 end = mask_destination(params, source_end);
        float2 overlap = saturate(min(destination_point + 0.5, end) - max(destination_point - 0.5, start));
        coverage = overlap.x * overlap.y;
        if (coverage <= 0.0) {
            return float4(0.0, 0.0, 0.0, 0.0);
        }
    }

    float top = source_point.y - params.fade_bounds.origin.y;
    float bottom = params.fade_bounds.origin.y + params.fade_bounds.size.y - source_point.y;

    float fade = 1.0;
    if (params.fade_top > 0.0) {
        fade = min(fade, smoothstep(0.0, params.fade_top, top));
    }
    if (params.fade_bottom > 0.0) {
        fade = min(fade, smoothstep(0.0, params.fade_bottom, bottom));
    }
    if (params.fade_left > 0.0) {
        float left = source_point.x - params.fade_bounds.origin.x;
        fade = min(fade, smoothstep(0.0, params.fade_left, left));
    }
    if (params.fade_right > 0.0) {
        float right = params.fade_bounds.origin.x + params.fade_bounds.size.x - source_point.x;
        fade = min(fade, smoothstep(0.0, params.fade_right, right));
    }

    // The sample stays half a texel inside the texels the layer covers and inside the frame, so
    // a stretched edge repeats the layer's last texels rather than blending them with whatever
    // lies past it. A pixel centre of an untransformed layer is already inside and does not move.
    float2 low = max(floor(source_start), float2(0.0, 0.0)) + 0.5;
    float2 high = min(ceil(source_end), global_viewport_size) - 0.5;
    float2 texel = clamp(source_point, low, max(low, high));
    return t_sprite.Sample(s_sprite, texel / global_viewport_size) * coverage * fade;
}
