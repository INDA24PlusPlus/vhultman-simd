const std = @import("std");

pub fn build(b: *std.Build) void {
    var target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const scalar = b.option(bool, "scalar", "Build the scalar version of the program") orelse true;
    const disable_sse = b.option(bool, "disable-sse", "Disables SSE instructions") orelse false;
    const path = if (scalar) b.path("src/scalar.zig") else b.path("src/simd.zig");

    if (disable_sse) {
        target.query.cpu_features_sub.addFeature(@intFromEnum(std.Target.x86.Feature.sse));
    }

    const exe_mod = b.createModule(.{
        .root_source_file = path,
        .target = target,
        .optimize = optimize,
    });

    const exe = b.addExecutable(.{
        .name = "ray_tracer",
        .root_module = exe_mod,
    });

    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const run_step = b.step("run", "Run the app");
    run_step.dependOn(&run_cmd.step);
}
