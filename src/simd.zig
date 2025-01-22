const std = @import("std");
const ln = @import("math_simd.zig");
const lns = @import("math_scalar.zig");
const Image = @import("Image.zig");

const image_width = 7680;
const image_height = 4320;

pub const config = struct {
    pub const ray_batch_size = 8;
    pub const batch_width = 4;
    pub const batch_height = 2;
};

const MaskInt = switch (config.ray_batch_size) {
    4 => u4,
    8 => u8,
    16 => u16,
    32 => u32,
    else => @compileError("Invalid ray batch size"),
};

const BoolVec = @Vector(config.ray_batch_size, bool);

comptime {
    std.debug.assert(image_width % config.batch_width == 0);
    std.debug.assert(image_height % config.batch_height == 0);
}

const Material = struct {
    albedo: lns.Vec3,
    roughness: f32,
    metallic: f32,
};

const HitRecord = struct {
    normal: [3]ln.Vec,
    t: ln.Vec,
    material: [config.ray_batch_size]u32,

    pub fn setNormal(self: *HitRecord, dir: []const ln.Vec, outward_normal: []const ln.Vec) void {
        const front_face = ln.dot(dir, outward_normal) < ln.splat(0);
        self.normal = .{
            @select(f32, front_face, outward_normal[0], ln.splat(-1.0) * outward_normal[0]),
            @select(f32, front_face, outward_normal[1], ln.splat(-1.0) * outward_normal[1]),
            @select(f32, front_face, outward_normal[2], ln.splat(-1.0) * outward_normal[2]),
        };
    }
};

const Sphere = struct {
    center: lns.Vec3,
    radius: f32,
    material: u32,

    pub fn hit(self: Sphere, origin: []const ln.Vec, dir: []const ln.Vec, t_min: ln.Vec, t_max: ln.Vec, rec: *HitRecord) BoolVec {
        const oc: [3]ln.Vec = .{
            ln.splat(self.center[0]) - origin[0],
            ln.splat(self.center[1]) - origin[1],
            ln.splat(self.center[2]) - origin[2],
        };
        const a = ln.dot(dir, dir);
        const h = ln.dot(dir, &oc);
        const c = ln.dot(&oc, &oc) - ln.splat(self.radius * self.radius);
        const discriminant = h * h - a * c;

        const did_hit = discriminant > ln.splat(0);
        if (!@reduce(.Or, did_hit)) {
            return @splat(false);
        }

        const sqrtd = @sqrt(discriminant);
        const x1 = (h - sqrtd) / a;
        const x2 = (h + sqrtd) / a;

        var mask1: MaskInt = @bitCast(x1 >= t_min);
        mask1 &= @bitCast(x1 <= t_max);

        var mask2: MaskInt = @bitCast(x2 >= t_min);
        mask2 &= @bitCast(x2 <= t_max);

        mask2 &= ~(mask1);

        var t = @select(f32, @as(BoolVec, @bitCast(mask1)), x1, ln.splat(1e30));
        t = @select(f32, @as(BoolVec, @bitCast(mask2)), x2, t);

        const radius = ln.splat(1.0 / self.radius);
        const outward_normal: [3]ln.Vec = .{
            ((origin[0] + dir[0] * x1) - ln.splat(self.center[0])) * radius,
            ((origin[1] + dir[1] * x1) - ln.splat(self.center[1])) * radius,
            ((origin[2] + dir[2] * x1) - ln.splat(self.center[2])) * radius,
        };
        rec.setNormal(dir, &outward_normal);
        rec.t = t;
        rec.material = .{self.material} ** config.ray_batch_size;

        return @bitCast(mask1 | mask2);
    }
};

const HittableList = struct {
    materials: std.ArrayListUnmanaged(Material) = .{},
    objects: std.ArrayListUnmanaged(Sphere) = .{},

    pub fn hit(self: *const HittableList, origin: []const ln.Vec, dir: []const ln.Vec, min_t: ln.Vec, max_t: ln.Vec, rec: *HitRecord) BoolVec {
        var temp_rec: HitRecord = undefined;
        var hit_anything: MaskInt = 0;
        var closest_hit_so_far = max_t;

        for (self.objects.items) |*obj| {
            const h = obj.hit(origin, dir, min_t, closest_hit_so_far, &temp_rec);
            rec.normal = .{
                @select(f32, h, temp_rec.normal[0], rec.normal[0]),
                @select(f32, h, temp_rec.normal[1], rec.normal[1]),
                @select(f32, h, temp_rec.normal[2], rec.normal[2]),
            };

            rec.t = @select(f32, h, temp_rec.t, rec.t);
            rec.material = @select(u32, h, temp_rec.material, rec.material);
            closest_hit_so_far = @select(f32, h, temp_rec.t, closest_hit_so_far);
            hit_anything |= @bitCast(h);
        }
        return @bitCast(hit_anything);
    }
};

fn pow(a: ln.Vec, b: u32) ln.Vec {
    var ret: ln.Vec = ln.splat(1);
    for (0..b) |_| {
        ret *= a;
    }

    return ret;
}

fn fSchlick(cos_theta: ln.Vec, F0: []const ln.Vec) [3]ln.Vec {
    const a = pow(
        std.math.clamp(ln.splat(1.0) - cos_theta, ln.splat(0.0), ln.splat(1.0)),
        5,
    );
    return .{
        F0[0] + (ln.splat(1.0) - F0[0]) * a,
        F0[1] + (ln.splat(1.0) - F0[1]) * a,
        F0[2] + (ln.splat(1.0) - F0[2]) * a,
    };
}

fn dGGX(n: []const ln.Vec, h: []const ln.Vec, roughness: ln.Vec) ln.Vec {
    const a = roughness * roughness;
    const a2 = a * a;
    const NdotH = @max(ln.dot(n, h), ln.splat(0.0));
    const NdotH2 = NdotH * NdotH;

    const num = a2;
    var denom = (NdotH2 * (a2 - ln.splat(1.0)) + ln.splat(1.0));
    denom = ln.splat(std.math.pi) * denom * denom;

    return num / denom;
}

fn gSchlickGGX(n_dot_v: ln.Vec, roughness: ln.Vec) ln.Vec {
    const r = (roughness + ln.splat(1.0));
    const k = (r * r) / ln.splat(8.0);
    const num = n_dot_v;
    const denom = n_dot_v * (ln.splat(1.0) - k) + k;
    return num / denom;
}

fn gSmith(n_dot_l: ln.Vec, n_dot_v: ln.Vec, roughness: ln.Vec) ln.Vec {
    const ggx2 = gSchlickGGX(n_dot_v, roughness);
    const ggx1 = gSchlickGGX(n_dot_l, roughness);
    return ggx2 * ggx1;
}

const light_dir = lns.normalize(lns.Vec3{ 1, -1, -1 });

fn rayColor(origin: []const ln.Vec, dir: []const ln.Vec, world: *const HittableList) [3]ln.Vec {
    var rec: HitRecord = undefined;
    rec.material = .{0} ** config.ray_batch_size;
    const hit = world.hit(origin, dir, ln.splat(0), ln.splat(std.math.inf(f32)), &rec);

    if (!@reduce(.Or, hit)) {
        const unit_dir = ln.normalize(dir);
        const a = ln.splat(0.5) * (unit_dir[1] + ln.splat(1.0));
        const one_minus_a = ln.splat(1.0) - a;

        return .{
            one_minus_a + a * ln.splat(0.5),
            one_minus_a + a * ln.splat(0.7),
            one_minus_a + a * ln.splat(1.0),
        };
    }

    var albedo: [3]ln.Vec = undefined;
    var roughness: ln.Vec = undefined;
    var metallic: ln.Vec = undefined;
    for (rec.material, 0..) |mat_idx, i| {
        const mat = &world.materials.items[mat_idx];
        albedo[0][i] = mat.albedo[0];
        albedo[1][i] = mat.albedo[1];
        albedo[2][i] = mat.albedo[2];

        roughness[i] = mat.roughness;
        metallic[i] = mat.metallic;
    }

    const shadow_origin: [3]ln.Vec = .{
        origin[0] + dir[0] * rec.t,
        origin[1] + dir[1] * rec.t,
        origin[2] + dir[2] * rec.t,
    };
    const L = ln.splatVector(lns.mul(light_dir, -1.0));
    const V: [3]ln.Vec = .{
        ln.splat(-1.0) * dir[0],
        ln.splat(-1.0) * dir[1],
        ln.splat(-1.0) * dir[2],
    };

    var temp_rec: HitRecord = undefined;
    const in_shadow = world.hit(&shadow_origin, &L, ln.splat(0.001), ln.splat(std.math.inf(f32)), &temp_rec);

    var out_color: [3]ln.Vec = .{
        ln.splat(0.03) * albedo[0],
        ln.splat(0.03) * albedo[1],
        ln.splat(0.03) * albedo[2],
    };

    // Light calc.
    {
        const H = ln.normalize(&.{ L[0] + V[0], L[1] + V[1], L[2] + V[2] });
        const n_dot_l = @max(ln.dot(&rec.normal, &L), ln.splat(0.0001));
        const n_dot_v = @max(ln.dot(&rec.normal, &V), ln.splat(0.0001));

        const F0 = ln.lerp(&.{ ln.splat(0.04), ln.splat(0.04), ln.splat(0.04) }, &albedo, metallic);
        const f = fSchlick(@max(ln.dot(&H, &V), ln.splat(0.0)), &F0);
        const d = dGGX(&rec.normal, &H, roughness);
        const g = gSmith(n_dot_l, n_dot_v, roughness);

        const num: [3]ln.Vec = .{
            f[0] * d * g,
            f[1] * d * g,
            f[2] * d * g,
        };

        const denom = ln.splat(4.0) * n_dot_v * n_dot_l + ln.splat(0.0001);
        const specular: [3]ln.Vec = .{
            num[0] / denom,
            num[1] / denom,
            num[2] / denom,
        };

        const one_over_pi = ln.splat(1.0 / std.math.pi);
        const diffuse: [3]ln.Vec = .{
            (ln.splat(1.0) - f[0]) * (ln.splat(1.0) - metallic) * albedo[0] * one_over_pi,
            (ln.splat(1.0) - f[1]) * (ln.splat(1.0) - metallic) * albedo[1] * one_over_pi,
            (ln.splat(1.0) - f[2]) * (ln.splat(1.0) - metallic) * albedo[2] * one_over_pi,
        };

        out_color = .{
            @select(f32, in_shadow, out_color[0], out_color[0] + (specular[0] + diffuse[0]) * n_dot_l),
            @select(f32, in_shadow, out_color[1], out_color[1] + (specular[1] + diffuse[1]) * n_dot_l),
            @select(f32, in_shadow, out_color[2], out_color[2] + (specular[2] + diffuse[2]) * n_dot_l),
        };
    }

    if (@reduce(.And, hit)) {
        return out_color;
    }

    const unit_dir = ln.normalize(dir);
    const a = ln.splat(0.5) * (unit_dir[1] + ln.splat(1.0));
    const one_minus_a = ln.splat(1.0) - a;

    const background: [3]ln.Vec = .{
        one_minus_a + a * ln.splat(0.5),
        one_minus_a + a * ln.splat(0.7),
        one_minus_a + a * ln.splat(1.0),
    };

    return .{
        @select(f32, hit, out_color[0], background[0]),
        @select(f32, hit, out_color[1], background[1]),
        @select(f32, hit, out_color[2], background[2]),
    };
}

pub fn main() !void {
    var gpa_state = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa_state.deinit();
    const gpa = gpa_state.allocator();

    const image: Image = .{
        .data = try gpa.alignedAlloc(u8, std.mem.page_size, image_width * image_height * 4),
        .width = image_width,
        .height = image_height,
    };
    defer gpa.free(image.data);
    const image_data = @as([*]u32, @ptrCast(@alignCast(image.data.ptr)))[0 .. image_width * image_height];

    var world: HittableList = .{};
    defer world.objects.deinit(gpa);
    defer world.materials.deinit(gpa);

    try world.materials.append(gpa, .{ .albedo = .{ 1, 1, 0 }, .metallic = 0.6, .roughness = 0.4 });
    try world.materials.append(gpa, .{ .albedo = .{ 0, 0.3, 1 }, .metallic = 0.3, .roughness = 0.7 });

    try world.objects.append(gpa, .{ .center = .{ 0, 0, -1 }, .radius = 0.5, .material = 0 });
    try world.objects.append(gpa, .{ .center = .{ 0, -100.5, -1 }, .radius = 100, .material = 1 });

    const aspect_ratio = @as(comptime_float, image_width) / @as(comptime_float, image_height);

    const focal_length = 1.0;
    const viewport_height = 2.0;
    const viewport_width = viewport_height * aspect_ratio;

    const camera_center_scalar: lns.Vec3 = .{ 0, 0, 0 };

    const viewport_u: lns.Vec3 = .{ viewport_width, 0, 0 };
    const viewport_v: lns.Vec3 = .{ 0, -viewport_height, 0 };

    const pixel_delta_u_scalar = lns.mul(viewport_u, 1.0 / @as(comptime_float, image_width));
    const pixel_delta_v_scalar = lns.mul(viewport_v, 1.0 / @as(comptime_float, image_height));

    const viewport_upper_left = blk: {
        var r = camera_center_scalar;
        r = lns.sub(r, lns.Vec3{ 0, 0, focal_length });
        r = lns.sub(r, lns.mul(viewport_u, 0.5));
        r = lns.sub(r, lns.mul(viewport_v, 0.5));
        break :blk r;
    };

    const pixels_offsets: [2]ln.Vec = blk: {
        var offset_x: ln.Vec = @splat(0);
        var offset_y: ln.Vec = @splat(0);
        for (0..config.batch_height) |y| {
            for (0..config.batch_width) |x| {
                offset_y[y * config.batch_width + x] = @floatFromInt(y);
                offset_x[y * config.batch_width + x] = @floatFromInt(x);
            }
        }

        break :blk .{ offset_x, offset_y };
    };

    const pixel00_loc_scalar = lns.add(viewport_upper_left, lns.mul(0.5, lns.add(pixel_delta_u_scalar, pixel_delta_v_scalar)));
    const pixel00_loc = ln.splatVector(pixel00_loc_scalar);
    const pixel_delta_u = ln.splatVector(pixel_delta_u_scalar);
    const pixel_delta_v = ln.splatVector(pixel_delta_v_scalar);
    const camera_center = ln.splatVector(camera_center_scalar);

    var iy: u32 = 0;
    while (iy < image_height) : (iy += config.batch_height) {
        var ix: u32 = 0;
        while (ix < image_width) : (ix += config.batch_width) {
            const fx: ln.Vec = @splat(@as(f32, @floatFromInt(ix)));
            const fy: ln.Vec = @splat(@as(f32, @floatFromInt(iy)));

            const x = pixels_offsets[0] + fx;
            const y = pixels_offsets[1] + fy;

            const center_x = ln.mulS(x, &pixel_delta_u);
            const center_y = ln.mulS(y, &pixel_delta_v);
            const pixel_center = ln.add(&pixel00_loc, &ln.add(&center_x, &center_y));

            const ray_dir = ln.sub(&pixel_center, &camera_center);

            var col = rayColor(&camera_center, &ray_dir, &world);
            const tone_mapped = toneMapKHRPBRNeutral(&col);
            const corrected = gammaCorrect(&tone_mapped);

            // Doing this is a bit faster.
            const U32: [config.ray_batch_size]u32 = @bitCast(RGBtoU32SIMD(&corrected));
            for (0..config.batch_height) |v| {
                const idx = (iy + v) * image_width + ix;
                image_data[idx..][0..config.batch_width].* = U32[v * config.batch_width ..][0..config.batch_width].*;
            }
        }
    }

    try image.save("out_simd.bmp");
}

// https://www.khronos.org/news/press/khronos-pbr-neutral-tone-mapper-released-for-true-to-life-color-rendering-of-3d-products
// https://github.com/KhronosGroup/glTF-Sample-Renderer/blob/63b7c128266cfd86bbd3f25caf8b3db3fe854015/source/Renderer/shaders/tonemapping.glsl
fn toneMapKHRPBRNeutral(col: []const ln.Vec) [3]ln.Vec {
    const start_compression = 0.8 - 0.04;
    const desaturation = 0.15;

    const x = @min(col[0], @min(col[1], col[2]));
    const offset = @select(f32, x < ln.splat(0.08), x - ln.splat(6.25) * x * x, ln.splat(0.04));
    const out = ln.subS(col, offset);

    const peak = @max(col[0], @max(col[1], col[2]));
    const less_than_comp = peak < ln.splat(start_compression);
    if (@reduce(.And, less_than_comp)) {
        return out;
    }

    const d = 1.0 - start_compression;
    const new_peak = ln.splat(1.0) - ln.splat(d * d) / (peak + ln.splat(d - start_compression));
    var new_out: [3]ln.Vec = .{
        out[0] * new_peak / peak,
        out[1] * new_peak / peak,
        out[2] * new_peak / peak,
    };

    const g = ln.splat(1.0) - ln.splat(1.0) / (ln.splat(desaturation) * (peak - new_peak) + ln.splat(1.0));
    new_out = ln.lerp(&new_out, &.{ new_peak, new_peak, new_peak }, g);
    return .{
        @select(f32, less_than_comp, out[0], new_out[0]),
        @select(f32, less_than_comp, out[1], new_out[1]),
        @select(f32, less_than_comp, out[2], new_out[2]),
    };
}

// ACES tone map (faster approximation)
// see: https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/
fn toneMapACES_Narkowicz(color: []const ln.Vec) [3]ln.Vec {
    const A = 2.51;
    const B = 0.03;
    const C = 2.43;
    const D = 0.59;
    const E = 0.14;
    return .{
        std.math.clamp(color[0] * (ln.splat(A) * color[0] + ln.splat(B)) / (color[0] * (ln.splat(C) * color[0] + ln.splat(D)) + ln.splat(E)), ln.splat(0.0), ln.splat(1.0)),
        std.math.clamp(color[1] * (ln.splat(A) * color[1] + ln.splat(B)) / (color[1] * (ln.splat(C) * color[1] + ln.splat(D)) + ln.splat(E)), ln.splat(0.0), ln.splat(1.0)),
        std.math.clamp(color[2] * (ln.splat(A) * color[2] + ln.splat(B)) / (color[2] * (ln.splat(C) * color[2] + ln.splat(D)) + ln.splat(E)), ln.splat(0.0), ln.splat(1.0)),
    };
}

fn gammaCorrect(rgb: []const ln.Vec) [3]ln.Vec {
    // Use gamma = 2.0 since it is much much faster to compute for our purposes.
    return .{
        @sqrt(rgb[0]),
        @sqrt(rgb[1]),
        @sqrt(rgb[2]),
    };
}

const UVec = @Vector(config.ray_batch_size, u32);
fn U32Splat(v: u32) UVec {
    return @splat(v);
}

fn RGBtoU32SIMD(rgb: []const ln.Vec) UVec {
    const b: UVec = @intFromFloat(ln.splat(255.999) * rgb[2]);
    const g: UVec = @intFromFloat(ln.splat(255.999) * rgb[1]);
    const r: UVec = @intFromFloat(ln.splat(255.999) * rgb[0]);
    return b | (g << @splat(8)) | (r << @splat(16)) | (U32Splat(0xff) << @splat(24));
}

fn RGBtoU32(rgb: lns.Vec3) u32 {
    const b: u32 = @intFromFloat(255.999 * rgb[2]);
    const g: u32 = @intFromFloat(255.999 * rgb[1]);
    const r: u32 = @intFromFloat(255.999 * rgb[0]);
    return b | (g << 8) | (r << 16) | (0xff << 24);
}
