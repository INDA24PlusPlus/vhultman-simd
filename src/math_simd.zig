const std = @import("std");
const lns = @import("math_scalar.zig");
const config = @import("simd.zig").config;

pub const Vec = @Vector(config.ray_batch_size, f32);

pub inline fn splat(v: f32) Vec {
    return @splat(v);
}

pub inline fn splatVector(s: lns.Vec3) [3]Vec {
    return .{
        @splat(s[0]),
        @splat(s[1]),
        @splat(s[2]),
    };
}

pub fn add(a: []const Vec, b: []const Vec) [3]Vec {
    return .{
        a[0] + b[0],
        a[1] + b[1],
        a[2] + b[2],
    };
}

pub fn sub(a: []const Vec, b: []const Vec) [3]Vec {
    return .{
        a[0] - b[0],
        a[1] - b[1],
        a[2] - b[2],
    };
}

pub inline fn subS(a: anytype, b: anytype) [3]Vec {
    if (@TypeOf(a) == Vec) {
        return .{
            a - b[0],
            a - b[1],
            a - b[2],
        };
    } else if (@TypeOf(b) == Vec) {
        return .{
            a[0] - b,
            a[1] - b,
            a[2] - b,
        };
    } else {
        @compileError("Invalid types! One argument has to be a scalar and the other a vector!");
    }
}

pub fn mul(a: []const Vec, b: []const Vec) [3]Vec {
    return .{
        a[0] * b[0],
        a[1] * b[1],
        a[2] * b[2],
    };
}

pub fn mulS(s: Vec, v: []const Vec) [3]Vec {
    return .{
        s * v[0],
        s * v[1],
        s * v[2],
    };
}

pub fn normalize(a: []const Vec) [3]Vec {
    const len = splat(1.0) / (@sqrt(a[0] * a[0] +
        a[1] * a[1] +
        a[2] * a[2]));

    return .{
        a[0] * len,
        a[1] * len,
        a[2] * len,
    };
}

fn lenSq(a: []const Vec) Vec {
    return a[0] * a[0] +
        a[1] * a[1] +
        a[2] * a[2];
}

pub fn dot(a: []const Vec, b: []const Vec) Vec {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

pub fn lerp(a: []const Vec, b: []const Vec, t: Vec) [3]Vec {
    return .{
        @mulAdd(Vec, b[0] - a[0], t, a[0]),
        @mulAdd(Vec, b[1] - a[1], t, a[1]),
        @mulAdd(Vec, b[2] - a[2], t, a[2]),
    };
}
