const std = @import("std");
const lit = @import("lit");
const lmao = @import("lmao");
const chad = @import("chad");
const common = @import("bench_common.zig");

const Camera = lit.Camera32;
const Scene = lit.Scene(void);
const ImageBuffer = lit.image.ImageBuffer;
const writePPM = lit.image.writePPM;
const Vec3 = lmao.Vec3f;
const Mat4 = lmao.Mat4f;
const Ray = chad.geometry.Ray;
const Mesh = chad.geometry.Mesh;
const Model = chad.geometry.Model;

const print = common.print;
const printHeader = common.printHeader;
const printRow = common.printRow;
const printRowCustom = common.printRowCustom;
const printFooter = common.printFooter;
const doNotOptimizeAway = common.doNotOptimizeAway;
const formatThroughputBuf = common.formatThroughputBuf;

// ============================================================================
// Helpers
// ============================================================================

fn createGridModel(allocator: std.mem.Allocator, grid_size: usize) !Model {
    var model = Model.init(allocator);

    const num_vertices = grid_size * grid_size;
    const num_tris = (grid_size - 1) * (grid_size - 1) * 2;

    var vertices = try allocator.alloc([3]f32, num_vertices);
    defer allocator.free(vertices);
    var indices = try allocator.alloc(u32, num_tris * 3);
    defer allocator.free(indices);

    // Generate vertices
    for (0..grid_size) |y| {
        for (0..grid_size) |x| {
            const idx = y * grid_size + x;
            vertices[idx] = .{
                @as(f32, @floatFromInt(x)) / @as(f32, @floatFromInt(grid_size - 1)) - 0.5,
                0.0,
                @as(f32, @floatFromInt(y)) / @as(f32, @floatFromInt(grid_size - 1)) - 0.5,
            };
        }
    }

    // Generate triangle indices
    var tri_idx: usize = 0;
    for (0..grid_size - 1) |y| {
        for (0..grid_size - 1) |x| {
            const v0: u32 = @intCast(y * grid_size + x);
            const v1: u32 = @intCast(y * grid_size + x + 1);
            const v2: u32 = @intCast((y + 1) * grid_size + x);
            const v3: u32 = @intCast((y + 1) * grid_size + x + 1);

            indices[tri_idx * 3 + 0] = v0;
            indices[tri_idx * 3 + 1] = v1;
            indices[tri_idx * 3 + 2] = v2;
            tri_idx += 1;

            indices[tri_idx * 3 + 0] = v1;
            indices[tri_idx * 3 + 1] = v3;
            indices[tri_idx * 3 + 2] = v2;
            tri_idx += 1;
        }
    }

    const mesh = try Mesh.initCopy(allocator, vertices, indices);
    try model.addMesh(mesh);

    return model;
}

fn findObjFile() ?[]const u8 {
    const possible_paths = [_][]const u8{
        "bunny.obj",
        "assets/bunny.obj",
        "../bunny.obj",
        "../assets/bunny.obj",
        "models/bunny.obj",
    };

    for (possible_paths) |path| {
        if (std.fs.cwd().access(path, .{})) |_| {
            return path;
        } else |_| {}
    }
    return null;
}

// ============================================================================
// Scene Loading Benchmarks
// ============================================================================

fn benchmarkObjLoad(iterations: usize, allocator: std.mem.Allocator, obj_path: []const u8) !f64 {
    var total_time: i128 = 0;

    for (0..iterations) |_| {
        var scene = Scene.init(allocator);

        const start = std.time.nanoTimestamp();
        _ = try scene.loadOBJ(obj_path, 1.0);
        const end = std.time.nanoTimestamp();

        total_time += end - start;
        scene.deinit();
    }

    return @as(f64, @floatFromInt(total_time)) / @as(f64, @floatFromInt(iterations));
}

fn benchmarkModelAddAndBuild(iterations: usize, allocator: std.mem.Allocator, grid_size: usize) !f64 {
    var total_time: i128 = 0;

    for (0..iterations) |_| {
        var scene = Scene.init(allocator);

        const start = std.time.nanoTimestamp();
        const model = try createGridModel(allocator, grid_size);
        const model_id = try scene.addModel(model);
        _ = try scene.instantiate(model_id, Mat4.identity(), {});
        try scene.buildAccelerationStructure();
        const end = std.time.nanoTimestamp();

        total_time += end - start;
        scene.deinit();
    }

    return @as(f64, @floatFromInt(total_time)) / @as(f64, @floatFromInt(iterations));
}

// ============================================================================
// Rendering Benchmarks
// ============================================================================

fn benchmarkRenderScene(allocator: std.mem.Allocator, width: u32, height: u32, grid_size: usize) !struct { ns_per_frame: f64, ns_per_pixel: f64, rays_per_sec: f64 } {
    var scene = Scene.init(allocator);
    defer scene.deinit();

    // Create and add model
    const model = try createGridModel(allocator, grid_size);
    const model_id = try scene.addModel(model);
    _ = try scene.instantiate(model_id, Mat4.identity(), {});

    // Setup camera
    var cam = Camera.fromFOV(60.0, width, height);
    const eye = Vec3.fromArray(&.{ 0, 1.5, 1.5 });
    const target = Vec3.fromArray(&.{ 0, 0, 0 });
    const up = Vec3.fromArray(&.{ 0, 1, 0 });
    cam.lookAt(eye, target, up);

    _ = try scene.addCamera(cam);
    try scene.buildAccelerationStructure();

    // Warm-up render
    try scene.render();

    // Benchmark render (multiple iterations)
    const num_iterations: usize = 5;
    var total_time: i128 = 0;

    for (0..num_iterations) |_| {
        const start = std.time.nanoTimestamp();
        try scene.render();
        const end = std.time.nanoTimestamp();
        total_time += end - start;
    }

    const ns_per_frame = @as(f64, @floatFromInt(total_time)) / @as(f64, @floatFromInt(num_iterations));
    const total_pixels = @as(f64, @floatFromInt(width)) * @as(f64, @floatFromInt(height));
    const ns_per_pixel = ns_per_frame / total_pixels;
    const rays_per_sec = total_pixels * 1_000_000_000.0 / ns_per_frame;

    return .{
        .ns_per_frame = ns_per_frame,
        .ns_per_pixel = ns_per_pixel,
        .rays_per_sec = rays_per_sec,
    };
}

fn benchmarkRenderCustom(allocator: std.mem.Allocator, width: u32, height: u32, grid_size: usize) !struct { ns_per_frame: f64, ns_per_pixel: f64, rays_per_sec: f64 } {
    var scene = Scene.init(allocator);
    defer scene.deinit();

    const model = try createGridModel(allocator, grid_size);
    const model_id = try scene.addModel(model);
    _ = try scene.instantiate(model_id, Mat4.identity(), {});

    var cam = Camera.fromFOV(60.0, width, height);
    const eye = Vec3.fromArray(&.{ 0, 1.5, 1.5 });
    const target = Vec3.fromArray(&.{ 0, 0, 0 });
    const up = Vec3.fromArray(&.{ 0, 1, 0 });
    cam.lookAt(eye, target, up);

    try cam.allocateImageBuffer(allocator);
    defer cam.freeImageBuffer();

    try scene.buildAccelerationStructure();

    var hits = std.ArrayListUnmanaged(Scene.Hit){};
    defer hits.deinit(allocator);

    // Light direction for shading
    const light_dir = Vec3.fromArray(&.{ 0.5, 1.0, 0.3 }).normalized();

    // Warm-up
    for (0..height) |y| {
        for (0..width) |x| {
            const ray = cam.getRay(@as(f32, @floatFromInt(x)) + 0.5, @as(f32, @floatFromInt(y)) + 0.5);
            hits.clearRetainingCapacity();
            try scene.castRay(ray, .forward, .closest, &hits);
        }
    }

    // Benchmark
    const num_iterations: usize = 5;
    var total_time: i128 = 0;

    for (0..num_iterations) |_| {
        const start = std.time.nanoTimestamp();

        for (0..height) |y| {
            for (0..width) |x| {
                const pixel_x: f32 = @floatFromInt(x);
                const pixel_y: f32 = @floatFromInt(y);
                const ray = cam.getRay(pixel_x + 0.5, pixel_y + 0.5);

                hits.clearRetainingCapacity();
                try scene.castRay(ray, .forward, .closest, &hits);

                var r: u8 = 30;
                var g: u8 = 30;
                var b: u8 = 50;

                if (hits.items.len > 0) {
                    const hit = hits.items[0];
                    const normal = scene.getHitNormal(hit);
                    const ndotl = @max(0.0, normal.dotProduct(light_dir));
                    const shade = 0.2 + 0.8 * ndotl;
                    r = @intFromFloat(shade * 200);
                    g = @intFromFloat(shade * 180);
                    b = @intFromFloat(shade * 160);
                }

                cam.setPixel(@intCast(x), @intCast(y), r, g, b);
            }
        }

        const end = std.time.nanoTimestamp();
        total_time += end - start;
    }

    const ns_per_frame = @as(f64, @floatFromInt(total_time)) / @as(f64, @floatFromInt(num_iterations));
    const total_pixels = @as(f64, @floatFromInt(width)) * @as(f64, @floatFromInt(height));
    const ns_per_pixel = ns_per_frame / total_pixels;
    const rays_per_sec = total_pixels * 1_000_000_000.0 / ns_per_frame;

    return .{
        .ns_per_frame = ns_per_frame,
        .ns_per_pixel = ns_per_pixel,
        .rays_per_sec = rays_per_sec,
    };
}

// ============================================================================
// Image Output Benchmarks
// ============================================================================

fn benchmarkImageBufferAlloc(iterations: usize, allocator: std.mem.Allocator, width: u32, height: u32) !f64 {
    var total_time: i128 = 0;

    for (0..iterations) |_| {
        const start = std.time.nanoTimestamp();
        var img = try ImageBuffer.init(allocator, width, height);
        const end = std.time.nanoTimestamp();
        total_time += end - start;
        img.deinit();
    }

    return @as(f64, @floatFromInt(total_time)) / @as(f64, @floatFromInt(iterations));
}

fn benchmarkImageBufferFill(iterations: usize, allocator: std.mem.Allocator, width: u32, height: u32) !f64 {
    var img = try ImageBuffer.init(allocator, width, height);
    defer img.deinit();

    const start = std.time.nanoTimestamp();
    for (0..iterations) |i| {
        const color: u8 = @truncate(i);
        img.fill(color, color, color);
    }
    const end = std.time.nanoTimestamp();
    doNotOptimizeAway(img.data[0]);

    return @as(f64, @floatFromInt(end - start)) / @as(f64, @floatFromInt(iterations));
}

fn benchmarkImageBufferSetPixel(iterations: usize, allocator: std.mem.Allocator, width: u32, height: u32) !f64 {
    var img = try ImageBuffer.init(allocator, width, height);
    defer img.deinit();

    const total_pixels = @as(usize, width) * @as(usize, height);

    const start = std.time.nanoTimestamp();
    for (0..iterations) |i| {
        const pixel_idx = i % total_pixels;
        const x: u32 = @intCast(pixel_idx % width);
        const y: u32 = @intCast(pixel_idx / width);
        const color: u8 = @truncate(i);
        img.setPixel(x, y, color, color, color);
    }
    const end = std.time.nanoTimestamp();
    doNotOptimizeAway(img.data[0]);

    return @as(f64, @floatFromInt(end - start)) / @as(f64, @floatFromInt(iterations));
}

fn benchmarkPPMWrite(iterations: usize, allocator: std.mem.Allocator, width: u32, height: u32) !f64 {
    var img = try ImageBuffer.init(allocator, width, height);
    defer img.deinit();

    // Fill with test pattern
    for (0..height) |y| {
        for (0..width) |x| {
            img.setPixel(@intCast(x), @intCast(y), @truncate(x), @truncate(y), @truncate(x + y));
        }
    }

    const tmp_path = "/tmp/bench_test.ppm";
    var total_time: i128 = 0;

    for (0..iterations) |_| {
        const start = std.time.nanoTimestamp();
        try img.savePPM(tmp_path);
        const end = std.time.nanoTimestamp();
        total_time += end - start;
    }

    // Cleanup
    std.fs.cwd().deleteFile(tmp_path) catch {};

    return @as(f64, @floatFromInt(total_time)) / @as(f64, @floatFromInt(iterations));
}

// ============================================================================
// Main
// ============================================================================

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    // ========================================================================
    // Scene Loading Benchmarks
    // ========================================================================
    print("\n{s}\n", .{"═" ** 70});
    print(" APPLICATION-LEVEL BENCHMARKS (ReleaseFast)\n", .{});
    print("{s}\n", .{"═" ** 70});

    // Check if OBJ file exists
    if (findObjFile()) |obj_path| {
        print("\n Scene Loading: {s}\n", .{obj_path});
        print("{s}\n", .{"─" ** 70});
        print(" {s: <35} │ {s: >12} │ {s: >15}\n", .{ "Operation", "Time/op", "Throughput" });
        print("{s}\n", .{"─" ** 70});

        const load_ns = try benchmarkObjLoad(5, allocator, obj_path);
        printRow("Load OBJ file", load_ns);
        printFooter();
    } else {
        print("\n [NOTE] No OBJ file found, skipping OBJ load benchmark\n", .{});
    }

    // Model creation benchmarks
    print("\n Scene Setup (Create + Build AS)\n", .{});
    print("{s}\n", .{"─" ** 70});
    print(" {s: <35} │ {s: >12} │ {s: >15}\n", .{ "Scene Configuration", "Time", "Throughput" });
    print("{s}\n", .{"─" ** 70});

    printRow("32x32 grid (1.8K tris)", try benchmarkModelAddAndBuild(20, allocator, 32));
    printRow("64x64 grid (7.9K tris)", try benchmarkModelAddAndBuild(10, allocator, 64));
    printRow("128x128 grid (32K tris)", try benchmarkModelAddAndBuild(5, allocator, 128));
    printRow("256x256 grid (130K tris)", try benchmarkModelAddAndBuild(2, allocator, 256));

    printFooter();

    // ========================================================================
    // Rendering Benchmarks
    // ========================================================================
    print("\n Rendering (Scene.render with normal shading)\n", .{});
    print("{s}\n", .{"─" ** 70});
    print(" {s: <35} │ {s: >12} │ {s: >15}\n", .{ "Resolution / Triangles", "Frame Time", "Rays/sec" });
    print("{s}\n", .{"─" ** 70});

    // Various resolutions with fixed scene
    const resolutions = [_]struct { w: u32, h: u32, name: []const u8 }{
        .{ .w = 128, .h = 128, .name = "128x128" },
        .{ .w = 256, .h = 256, .name = "256x256" },
        .{ .w = 512, .h = 512, .name = "512x512" },
        .{ .w = 640, .h = 480, .name = "640x480 (VGA)" },
        .{ .w = 1280, .h = 720, .name = "1280x720 (720p)" },
    };

    for (resolutions) |res| {
        const result = try benchmarkRenderScene(allocator, res.w, res.h, 64);
        var throughput_buf: [32]u8 = undefined;
        const throughput_str = formatThroughputBuf(result.rays_per_sec, &throughput_buf);
        var name_buf: [64]u8 = undefined;
        const name = std.fmt.bufPrint(&name_buf, "{s} / 7.9K tris", .{res.name}) catch res.name;
        printRowCustom(name, result.ns_per_frame, throughput_str);
    }

    printFooter();

    // Custom render loop benchmark (showing overhead)
    print("\n Rendering (Custom loop with diffuse shading)\n", .{});
    print("{s}\n", .{"─" ** 70});
    print(" {s: <35} │ {s: >12} │ {s: >15}\n", .{ "Resolution", "Frame Time", "Rays/sec" });
    print("{s}\n", .{"─" ** 70});

    {
        const result = try benchmarkRenderCustom(allocator, 256, 256, 64);
        var throughput_buf: [32]u8 = undefined;
        printRowCustom("256x256", result.ns_per_frame, formatThroughputBuf(result.rays_per_sec, &throughput_buf));
    }
    {
        const result = try benchmarkRenderCustom(allocator, 512, 512, 64);
        var throughput_buf: [32]u8 = undefined;
        printRowCustom("512x512", result.ns_per_frame, formatThroughputBuf(result.rays_per_sec, &throughput_buf));
    }

    printFooter();

    // ========================================================================
    // Image Output Benchmarks
    // ========================================================================
    print("\n Image Buffer Operations (1920x1080)\n", .{});
    print("{s}\n", .{"─" ** 70});
    print(" {s: <35} │ {s: >12} │ {s: >15}\n", .{ "Operation", "Time/op", "Throughput" });
    print("{s}\n", .{"─" ** 70});

    printRow("ImageBuffer.init", try benchmarkImageBufferAlloc(100, allocator, 1920, 1080));
    printRow("ImageBuffer.fill", try benchmarkImageBufferFill(100, allocator, 1920, 1080));
    printRow("ImageBuffer.setPixel", try benchmarkImageBufferSetPixel(1_000_000, allocator, 1920, 1080));

    printFooter();

    print("\n PPM Export\n", .{});
    print("{s}\n", .{"─" ** 70});
    print(" {s: <35} │ {s: >12} │ {s: >15}\n", .{ "Resolution", "Time", "MB/sec" });
    print("{s}\n", .{"─" ** 70});

    const ppm_configs = [_]struct { w: u32, h: u32, name: []const u8 }{
        .{ .w = 640, .h = 480, .name = "640x480" },
        .{ .w = 1280, .h = 720, .name = "1280x720" },
        .{ .w = 1920, .h = 1080, .name = "1920x1080" },
    };

    for (ppm_configs) |cfg| {
        const ns = try benchmarkPPMWrite(10, allocator, cfg.w, cfg.h);
        const bytes = @as(f64, @floatFromInt(cfg.w)) * @as(f64, @floatFromInt(cfg.h)) * 3.0;
        const mb_per_sec = bytes / (ns / 1_000_000_000.0) / (1024.0 * 1024.0);
        var mb_buf: [32]u8 = undefined;
        const mb_str = std.fmt.bufPrint(&mb_buf, "{d:.1}MB/s", .{mb_per_sec}) catch "?";
        printRowCustom(cfg.name, ns, mb_str);
    }

    printFooter();
}
