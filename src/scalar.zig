const std = @import("std");
const ln = @import("math_scalar.zig");
const Image = @import("Image.zig");
const Allocator = std.mem.Allocator;

const image_width = 7680;
const image_height = 4320;

const Interval = struct {
    min: f32 = std.math.inf(f32),
    max: f32 = -std.math.inf(f32),

    pub fn size(self: Interval) f32 {
        return self.max - self.min;
    }

    pub fn contains(self: Interval, x: f32) bool {
        return self.min <= x and x <= self.max;
    }

    pub fn surrounds(self: Interval, x: f32) bool {
        return self.min < x and x < self.max;
    }
};

const Material = struct {
    albedo: ln.Vec3,
    roughness: f32,
    metallic: f32,
};

const HitRecord = struct {
    p: ln.Vec3,
    normal: ln.Vec3,
    t: f32,
    material: u32,

    pub fn setNormal(self: *HitRecord, r: *const Ray, outward_normal: ln.Vec3) void {
        const front_face = ln.dot(r.dir, outward_normal) < 0;
        self.normal = if (front_face) outward_normal else ln.mul(-1.0, outward_normal);
    }
};

const Sphere = struct {
    center: ln.Vec3,
    radius: f32,
    material: u32,

    pub fn hit(self: *const Sphere, r: *const Ray, ray_t: Interval, rec: *HitRecord) bool {
        const oc = ln.sub(self.center, r.origin);
        const a = ln.lengthSquared(r.dir);
        const h = ln.dot(r.dir, oc);
        const c = ln.lengthSquared(oc) - self.radius * self.radius;
        const discriminant = h * h - a * c;

        if (discriminant < 0) {
            return false;
        }

        const sqrtd = @sqrt(discriminant);
        var root = (h - sqrtd) / a;
        if (!ray_t.surrounds(root)) {
            root = (h + sqrtd) / a;
            if (!ray_t.surrounds(root)) {
                return false;
            }
        }

        rec.t = root;
        rec.p = r.at(rec.t);
        const outward_normal = ln.mul(ln.sub(rec.p, self.center), 1.0 / self.radius);
        rec.setNormal(r, outward_normal);
        rec.material = self.material;

        return true;
    }
};

const HittableList = struct {
    materials: std.ArrayListUnmanaged(Material) = .{},
    objects: std.ArrayListUnmanaged(Sphere) = .{},

    pub fn hit(self: *const HittableList, r: *const Ray, ray_t: Interval, rec: *HitRecord) bool {
        var temp_rec: HitRecord = undefined;
        var hit_anything = false;
        var closest_hit_so_far = ray_t.max;

        for (self.objects.items) |*obj| {
            if (obj.hit(r, .{ .min = ray_t.min, .max = closest_hit_so_far }, &temp_rec)) {
                rec.* = temp_rec;
                closest_hit_so_far = temp_rec.t;
                hit_anything = true;
            }
        }

        return hit_anything;
    }
};

const Ray = struct {
    origin: ln.Vec3,
    dir: ln.Vec3,

    pub fn at(r: Ray, t: f32) ln.Vec3 {
        return ln.add(r.origin, ln.mul(t, r.dir));
    }
};

const light_dir = ln.normalize(ln.Vec3{ 1, -1, -1 });

// We replace std.math.pow with our own function in the SIMD version
// because it did not support vectors so it is only fair that we replace it here too.
// This version is slighly faster than std.math.pow because we know the exponent is a integer.
fn pow(v: f32, exp: u32) f32 {
    var ret: f32 = 1;
    for (0..exp) |_| ret *= v;
    return ret;
}

fn fSchlick(cos_theta: f32, F0: ln.Vec3) ln.Vec3 {
    const a = pow(std.math.clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
    return ln.add(F0, ln.mul(ln.sub(1.0, F0), a));
}

fn dGGX(n: ln.Vec3, h: ln.Vec3, roughness: f32) f32 {
    const a = roughness * roughness;
    const a2 = a * a;
    const NdotH = @max(ln.dot(n, h), 0.0);
    const NdotH2 = NdotH * NdotH;

    const num = a2;
    var denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = std.math.pi * denom * denom;

    return num / denom;
}

fn gSchlickGGX(n_dot_v: f32, roughness: f32) f32 {
    const r = (roughness + 1.0);
    const k = (r * r) / 8.0;

    const num = n_dot_v;
    const denom = n_dot_v * (1.0 - k) + k;
    return num / denom;
}

fn gSmith(n: ln.Vec3, v: ln.Vec3, l: ln.Vec3, roughness: f32) f32 {
    const n_dot_v = @max(ln.dot(n, v), 0.0001);
    const n_dot_l = @max(ln.dot(n, l), 0.001);
    const ggx2 = gSchlickGGX(n_dot_v, roughness);
    const ggx1 = gSchlickGGX(n_dot_l, roughness);
    return ggx2 * ggx1;
}

fn rayColor(r: Ray, world: *const HittableList) ln.Vec3 {
    var rec: HitRecord = undefined;
    if (world.hit(&r, .{ .min = 0, .max = std.math.inf(f32) }, &rec)) {
        const L = ln.mul(-1.0, light_dir);
        const V = ln.mul(-1.0, r.dir);

        const shadow_ray: Ray = .{ .origin = rec.p, .dir = ln.mul(-1.0, light_dir) };

        var temp_rec: HitRecord = undefined;
        const in_shadow = world.hit(&shadow_ray, .{ .min = 1e-4, .max = std.math.inf(f32) }, &temp_rec);
        const mat = world.materials.items[rec.material];

        var out_color = ln.mul(0.03, mat.albedo);
        if (!in_shadow) {
            const H = ln.normalize(ln.add(L, V));
            const n_dot_l = @max(ln.dot(rec.normal, L), 0.0);

            const F0 = ln.lerp(.{ 0.04, 0.04, 0.04 }, mat.albedo, mat.metallic);
            const f = fSchlick(@max(ln.dot(H, V), 0.0), F0);
            const d = dGGX(rec.normal, H, mat.roughness);
            const g = gSmith(rec.normal, V, L, mat.roughness);

            const num = ln.mul(f, d * g);
            const denom = 4.0 * @max(ln.dot(rec.normal, V), 0.0) * n_dot_l + 0.0001;
            const specular = ln.mul(num, 1.0 / denom);

            const ks = f;
            var kd = ln.sub(1.0, ks);
            kd = ln.mul(kd, 1.0 - mat.metallic);
            const diffuse = ln.mul(kd, ln.mul(mat.albedo, 1.0 / std.math.pi));

            const brdf = ln.mul(ln.add(diffuse, specular), n_dot_l);
            out_color = ln.add(out_color, brdf);
        }

        return out_color;
    }

    const unit_dir = ln.normalize(r.dir);
    const a = 0.5 * (unit_dir[1] + 1.0);
    return ln.add(1.0 - a, ln.mul(a, ln.Vec3{ 0.5, 0.7, 1.0 }));
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

    const camera_center: ln.Vec3 = .{ 0, 0, 0 };

    const viewport_u: ln.Vec3 = .{ viewport_width, 0, 0 };
    const viewport_v: ln.Vec3 = .{ 0, -viewport_height, 0 };

    const pixel_delta_u = ln.mul(viewport_u, 1.0 / @as(comptime_float, image_width));
    const pixel_delta_v = ln.mul(viewport_v, 1.0 / @as(comptime_float, image_height));

    const viewport_upper_left = blk: {
        var r = camera_center;
        r = ln.sub(r, ln.Vec3{ 0, 0, focal_length });
        r = ln.sub(r, ln.mul(viewport_u, 0.5));
        r = ln.sub(r, ln.mul(viewport_v, 0.5));
        break :blk r;
    };

    const pixel00_loc = ln.add(viewport_upper_left, ln.mul(0.5, ln.add(pixel_delta_u, pixel_delta_v)));

    for (0..image_height) |y| {
        for (0..image_width) |x| {
            const fy: f32 = @floatFromInt(y);
            const fx: f32 = @floatFromInt(x);

            const center_x = ln.mul(fx, pixel_delta_u);
            const center_y = ln.mul(fy, pixel_delta_v);
            const pixel_center = ln.add(pixel00_loc, ln.add(center_x, center_y));
            const ray_dir = ln.sub(pixel_center, camera_center);

            const r: Ray = .{ .dir = ray_dir, .origin = camera_center };
            const col = rayColor(r, &world);
            const tone_mapped = toneMapKHRPBRNeutral(col);
            image_data[y * image_width + x] = RGBtoU32(gammaCorrect(2.0, tone_mapped));
        }
    }

    try image.save("out.bmp");
}

// https://www.khronos.org/news/press/khronos-pbr-neutral-tone-mapper-released-for-true-to-life-color-rendering-of-3d-products
// https://github.com/KhronosGroup/glTF-Sample-Renderer/blob/63b7c128266cfd86bbd3f25caf8b3db3fe854015/source/Renderer/shaders/tonemapping.glsl
fn toneMapKHRPBRNeutral(col: ln.Vec3) ln.Vec3 {
    const start_compression = 0.8 - 0.04;
    const desaturation = 0.15;

    const x = @min(col[0], @min(col[1], col[2]));
    const offset = if (x < 0.08) x - 6.25 * x * x else 0.04;
    var out = ln.sub(col, offset);

    const peak = @max(col[0], @max(col[1], col[2]));
    if (peak < start_compression) return out;

    const d = 1.0 - start_compression;
    const new_peak = 1.0 - d * d / (peak + d - start_compression);
    out = ln.mul(out, new_peak / peak);

    const g = 1.0 - 1.0 / (desaturation * (peak - new_peak) + 1.0);
    return ln.lerp(out, ln.Vec3{ new_peak, new_peak, new_peak }, g);
}

fn toneMapACES_Narkowicz(color: ln.Vec3) ln.Vec3 {
    const A = 2.51;
    const B = 0.03;
    const C = 2.43;
    const D = 0.59;
    const E = 0.14;
    const num = ln.mul(color, ln.add(ln.mul(A, color), B));
    const denom = ln.add(ln.mul(color, ln.add(ln.mul(C, color), D)), E);
    const result = ln.div(num, denom);
    return .{
        std.math.clamp(result[0], 0.0, 1.0),
        std.math.clamp(result[1], 0.0, 1.0),
        std.math.clamp(result[2], 0.0, 1.0),
    };
}

fn gammaCorrect(gamma: f32, rgb: ln.Vec3) ln.Vec3 {
    return .{
        std.math.pow(f32, rgb[0], 1.0 / gamma),
        std.math.pow(f32, rgb[1], 1.0 / gamma),
        std.math.pow(f32, rgb[2], 1.0 / gamma),
    };
}

fn RGBtoU32(rgb: ln.Vec3) u32 {
    const b: u32 = @intFromFloat(255.999 * rgb[2]);
    const g: u32 = @intFromFloat(255.999 * rgb[1]);
    const r: u32 = @intFromFloat(255.999 * rgb[0]);
    return b | (g << 8) | (r << 16) | (0xff << 24);
}
