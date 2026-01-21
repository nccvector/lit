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

    // ========================================================================
    // Benchmark targets
    // ========================================================================

    // Shared benchmark common module
    const bench_common_mod = b.addModule("bench_common", .{
        .root_source_file = b.path("src/bench_common.zig"),
        .target = target,
        .optimize = .ReleaseFast, // Benchmarks always ReleaseFast
    });

    // API-level benchmarks
    const bench_api_exe = b.addExecutable(.{
        .name = "bench-api",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/bench_api.zig"),
            .target = target,
            .optimize = .ReleaseFast,
            .imports = &.{
                .{ .name = "lit", .module = lit_mod },
                .{ .name = "lmao", .module = lmao_mod },
                .{ .name = "chad", .module = chad_mod },
                .{ .name = "bench_common", .module = bench_common_mod },
            },
        }),
    });
    b.installArtifact(bench_api_exe);

    const bench_api_step = b.step("bench-api", "Run API-level benchmarks (Camera, Scene)");
    const bench_api_run = b.addRunArtifact(bench_api_exe);
    bench_api_step.dependOn(&bench_api_run.step);
    bench_api_run.step.dependOn(b.getInstallStep());
    if (b.args) |bench_args| {
        bench_api_run.addArgs(bench_args);
    }

    // Application-level benchmarks
    const bench_app_exe = b.addExecutable(.{
        .name = "bench-app",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/bench_app.zig"),
            .target = target,
            .optimize = .ReleaseFast,
            .imports = &.{
                .{ .name = "lit", .module = lit_mod },
                .{ .name = "lmao", .module = lmao_mod },
                .{ .name = "chad", .module = chad_mod },
                .{ .name = "bench_common", .module = bench_common_mod },
            },
        }),
    });
    b.installArtifact(bench_app_exe);

    const bench_app_step = b.step("bench-app", "Run application-level benchmarks (loading, rendering, PPM)");
    const bench_app_run = b.addRunArtifact(bench_app_exe);
    bench_app_step.dependOn(&bench_app_run.step);
    bench_app_run.step.dependOn(b.getInstallStep());
    if (b.args) |bench_args| {
        bench_app_run.addArgs(bench_args);
    }

    // Combined benchmark target
    const bench_step = b.step("bench", "Run all benchmarks");
    bench_step.dependOn(&bench_api_run.step);
    bench_step.dependOn(&bench_app_run.step);
}
