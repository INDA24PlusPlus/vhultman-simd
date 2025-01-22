const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const scalar = b.option(bool, "scalar", "Build the scalar version of the program") orelse true;

    const path = if (scalar) b.path("src/scalar.zig") else b.path("src/simd.zig");

    const exe = b.addExecutable(.{
        .name = "ray_tracer",
        .root_source_file = path,
        .target = target,
        .optimize = optimize,
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
