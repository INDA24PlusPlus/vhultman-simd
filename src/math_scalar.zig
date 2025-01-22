const std = @import("std");

pub const Vec3 = [3]f32;

pub fn add(a: anytype, b: anytype) Vec3 {
    if (@TypeOf(a) == Vec3 and @TypeOf(b) == Vec3) {
        return .{ a[0] + b[0], a[1] + b[1], a[2] + b[2] };
    } else if (@TypeOf(a) == Vec3 and (@TypeOf(b) == f32 or @TypeOf(b) == comptime_float)) {
        return .{ a[0] + b, a[1] + b, a[2] + b };
    } else if (@TypeOf(a) == f32 and @TypeOf(b) == Vec3) {
        return .{ b[0] + a, b[1] + a, b[2] + a };
    } else {
        @compileError("Addition not supported for type " ++ @typeName(@TypeOf(a)) ++ " and " ++ @typeName(@TypeOf(b)));
    }
}

pub fn sub(a: anytype, b: anytype) Vec3 {
    if (@TypeOf(a) == Vec3 and @TypeOf(b) == Vec3) {
        return .{ a[0] - b[0], a[1] - b[1], a[2] - b[2] };
    } else if (@TypeOf(a) == Vec3 and @TypeOf(b) == f32) {
        return .{ a[0] - b, a[1] - b, a[2] - b };
    } else if (@TypeOf(a) == comptime_float and @TypeOf(b) == Vec3) {
        return .{ a - b[0], a - b[1], a - b[2] };
    } else {
        @compileError("Subtraction not supported for type " ++ @typeName(@TypeOf(a)) ++ " and " ++ @typeName(@TypeOf(b)));
    }
}

pub fn mul(a: anytype, b: anytype) Vec3 {
    if (@TypeOf(a) == Vec3 and @TypeOf(b) == Vec3) {
        return .{ a[0] * b[0], a[1] * b[1], a[2] * b[2] };
    } else if (@TypeOf(a) == Vec3 and (@TypeOf(b) == f32 or @TypeOf(b) == comptime_float)) {
        return .{ a[0] * b, a[1] * b, a[2] * b };
    } else if ((@TypeOf(a) == f32 or @TypeOf(a) == comptime_float) and @TypeOf(b) == Vec3) {
        return .{ b[0] * a, b[1] * a, b[2] * a };
    } else {
        @compileError("Multiplication not supported for type " ++ @typeName(@TypeOf(a)) ++ " and " ++ @typeName(@TypeOf(b)));
    }
}

pub fn div(a: anytype, b: anytype) Vec3 {
    if (@TypeOf(a) == Vec3 and @TypeOf(b) == Vec3) {
        return .{ a[0] / b[0], a[1] / b[1], a[2] / b[2] };
    } else {
        @compileError("Div not supported for type " ++ @typeName(@TypeOf(a)) ++ " and " ++ @typeName(@TypeOf(b)));
    }
}

pub fn length(v: Vec3) f32 {
    return @sqrt(lengthSquared(v));
}

pub fn lengthSquared(v: Vec3) f32 {
    return dot(v, v);
}

pub fn dot(a: Vec3, b: Vec3) f32 {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

pub fn cross(a: Vec3, b: Vec3) Vec3 {
    return .{
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    };
}

pub fn normalize(v: Vec3) Vec3 {
    const len = 1.0 / length(v);
    return .{ v[0] * len, v[1] * len, v[2] * len };
}

pub fn lerp(a: Vec3, b: Vec3, t: f32) Vec3 {
    return .{
        @mulAdd(f32, b[0] - a[0], t, a[0]),
        @mulAdd(f32, b[1] - a[1], t, a[1]),
        @mulAdd(f32, b[2] - a[2], t, a[2]),
    };
}
