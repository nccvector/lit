const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Get dependencies
    const lmao_dep = b.dependency("lmao", .{
        .target = target,
        .optimize = optimize,
    });
    const chad_dep = b.dependency("chad", .{
        .target = target,
        .optimize = optimize,
    });

    const lmao_mod = lmao_dep.module("lmao");
    const chad_mod = chad_dep.module("chad");

    // Create the lit library module
    const lit_mod = b.addModule("lit", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "lmao", .module = lmao_mod },
            .{ .name = "chad", .module = chad_mod },
        },
    });

    // Create the executable
    const exe = b.addExecutable(.{
        .name = "lit",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "lit", .module = lit_mod },
                .{ .name = "lmao", .module = lmao_mod },
                .{ .name = "chad", .module = chad_mod },
            },
        }),
    });

    b.installArtifact(exe);

    // Run step
    const run_step = b.step("run", "Run the raytracer demo");
    const run_cmd = b.addRunArtifact(exe);
    run_step.dependOn(&run_cmd.step);
    run_cmd.step.dependOn(b.getInstallStep());

    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    // Test filter option
    const test_filter = b.option([]const u8, "test-filter", "Filter tests by name");

    // Test step
    const mod_tests = b.addTest(.{
        .root_module = lit_mod,
        .filters = if (test_filter) |f| &.{f} else &.{},
    });
    const run_mod_tests = b.addRunArtifact(mod_tests);

    const exe_tests = b.addTest(.{
        .root_module = exe.root_module,
        .filters = if (test_filter) |f| &.{f} else &.{},
    });
    const run_exe_tests = b.addRunArtifact(exe_tests);

    const test_step = b.step("test", "Run tests");
    test_step.dependOn(&run_mod_tests.step);
    test_step.dependOn(&run_exe_tests.step);
}
