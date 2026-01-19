const std = @import("std");
const lit = @import("lit");
const lmao = @import("lmao");
const chad = @import("chad");
const common = @import("bench_common.zig");

const Camera = lit.Camera32;
const Scene = lit.Scene(void);
const Vec3 = lmao.Vec3f;
const Mat4 = lmao.Mat4f;
const Ray = chad.geometry.Ray;
const Triangle = chad.geometry.Triangle;
const Mesh = chad.geometry.Mesh;
const Model = chad.geometry.Model;

const print = common.print;
const printHeader = common.printHeader;
const printRow = common.printRow;
const printFooter = common.printFooter;
const doNotOptimizeAway = common.doNotOptimizeAway;
const randomFloatRange = common.randomFloatRange;

// ============================================================================
// Camera Benchmarks
// ============================================================================

fn benchmarkCameraFromFOV(iterations: usize) f64 {
    const start = std.time.nanoTimestamp();
    for (0..iterations) |i| {
        const fov: f32 = 45.0 + @as(f32, @floatFromInt(i % 90));
        const cam = Camera.fromFOV(fov, 1920, 1080);
        doNotOptimizeAway(cam.K.data[0]);
    }
    const end = std.time.nanoTimestamp();
    return @as(f64, @floatFromInt(end - start)) / @as(f64, @floatFromInt(iterations));
}

fn benchmarkCameraFromIntrinsics(iterations: usize) f64 {
    const start = std.time.nanoTimestamp();
    for (0..iterations) |i| {
        const f: f32 = @floatFromInt(i % 1000 + 500);
        const cam = Camera.fromIntrinsics(f, f, 960.0, 540.0, 1920, 1080);
        doNotOptimizeAway(cam.K.data[0]);
    }
    const end = std.time.nanoTimestamp();
    return @as(f64, @floatFromInt(end - start)) / @as(f64, @floatFromInt(iterations));
}

fn benchmarkCameraGetRay(iterations: usize, rng: std.Random) f64 {
    const cam = Camera.fromFOV(60.0, 1920, 1080);

    // Pre-generate random pixel coordinates
    var pixels_x: [1024]f32 = undefined;
    var pixels_y: [1024]f32 = undefined;
    for (0..1024) |j| {
        pixels_x[j] = randomFloatRange(f32, rng, 0.0, 1920.0);
        pixels_y[j] = randomFloatRange(f32, rng, 0.0, 1080.0);
    }

    var acc: f32 = 0;
    const start = std.time.nanoTimestamp();
    for (0..iterations) |i| {
        const idx = i % 1024;
        const ray = cam.getRay(pixels_x[idx], pixels_y[idx]);
        acc += ray.direction.data[0];
    }
    const end = std.time.nanoTimestamp();
    doNotOptimizeAway(acc);
    return @as(f64, @floatFromInt(end - start)) / @as(f64, @floatFromInt(iterations));
}

fn benchmarkCameraLookAt(iterations: usize, rng: std.Random) f64 {
    var cam = Camera.fromFOV(60.0, 1920, 1080);

    // Pre-generate random positions
    var eyes: [256][3]f32 = undefined;
    var targets: [256][3]f32 = undefined;
    for (0..256) |j| {
        eyes[j] = .{
            randomFloatRange(f32, rng, -10.0, 10.0),
            randomFloatRange(f32, rng, -10.0, 10.0),
            randomFloatRange(f32, rng, -10.0, 10.0),
        };
        targets[j] = .{
            randomFloatRange(f32, rng, -5.0, 5.0),
            randomFloatRange(f32, rng, -5.0, 5.0),
            randomFloatRange(f32, rng, -5.0, 5.0),
        };
    }

    const up = Vec3.fromArray(&.{ 0, 1, 0 });
    var acc: f32 = 0;
    const start = std.time.nanoTimestamp();
    for (0..iterations) |i| {
        const idx = i % 256;
        const eye = Vec3.fromArray(&eyes[idx]);
        const target = Vec3.fromArray(&targets[idx]);
        cam.lookAt(eye, target, up);
        acc += cam.transform.data[12];
    }
    const end = std.time.nanoTimestamp();
    doNotOptimizeAway(acc);
    return @as(f64, @floatFromInt(end - start)) / @as(f64, @floatFromInt(iterations));
}

fn benchmarkCameraGetKInverse(iterations: usize) f64 {
    const cam = Camera.fromFOV(60.0, 1920, 1080);

    var acc: f32 = 0;
    const start = std.time.nanoTimestamp();
    for (0..iterations) |_| {
        if (cam.getKInverse()) |k_inv| {
            acc += k_inv.data[0];
        }
    }
    const end = std.time.nanoTimestamp();
    doNotOptimizeAway(acc);
    return @as(f64, @floatFromInt(end - start)) / @as(f64, @floatFromInt(iterations));
}

// ============================================================================
// Scene Benchmarks
// ============================================================================

fn createTestModel(allocator: std.mem.Allocator, num_triangles: usize) !Model {
    var model = Model.init(allocator);

    // Create a simple grid mesh with the specified number of triangles
    const grid_size = @as(usize, @intFromFloat(@sqrt(@as(f64, @floatFromInt(num_triangles / 2))))) + 1;
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
                @as(f32, @floatFromInt(x)) / @as(f32, @floatFromInt(grid_size)),
                0.0,
                @as(f32, @floatFromInt(y)) / @as(f32, @floatFromInt(grid_size)),
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

            // First triangle
            indices[tri_idx * 3 + 0] = v0;
            indices[tri_idx * 3 + 1] = v1;
            indices[tri_idx * 3 + 2] = v2;
            tri_idx += 1;

            // Second triangle
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

fn benchmarkSceneAddModel(iterations: usize, allocator: std.mem.Allocator) !f64 {
    // Create test models upfront
    var models = try allocator.alloc(Model, iterations);
    defer {
        for (models) |*m| {
            m.deinit();
        }
        allocator.free(models);
    }

    for (0..iterations) |i| {
        models[i] = try createTestModel(allocator, 100);
    }

    var scene = Scene.init(allocator);
    defer scene.deinit();

    const start = std.time.nanoTimestamp();
    for (0..iterations) |i| {
        // Clone the model since addModel takes ownership
        var model_copy = Model.init(allocator);
        const mesh = try Mesh.initCopy(allocator, models[i].meshes.items[0].vertices, models[i].meshes.items[0].indices);
        try model_copy.addMesh(mesh);
        _ = try scene.addModel(model_copy);
    }
    const end = std.time.nanoTimestamp();

    return @as(f64, @floatFromInt(end - start)) / @as(f64, @floatFromInt(iterations));
}

fn benchmarkSceneInstantiate(iterations: usize, allocator: std.mem.Allocator, rng: std.Random) !f64 {
    var scene = Scene.init(allocator);
    defer scene.deinit();

    // Add a model to instantiate
    const model = try createTestModel(allocator, 1000);
    const model_id = try scene.addModel(model);

    // Pre-generate random transforms
    var transforms: [1024]Mat4 = undefined;
    for (0..1024) |j| {
        const tx = randomFloatRange(f32, rng, -10.0, 10.0);
        const ty = randomFloatRange(f32, rng, -10.0, 10.0);
        const tz = randomFloatRange(f32, rng, -10.0, 10.0);
        transforms[j] = Mat4.translation(tx, ty, tz);
    }

    const start = std.time.nanoTimestamp();
    for (0..iterations) |i| {
        _ = try scene.instantiate(model_id, transforms[i % 1024], {});
    }
    const end = std.time.nanoTimestamp();

    return @as(f64, @floatFromInt(end - start)) / @as(f64, @floatFromInt(iterations));
}

fn benchmarkSceneBuildAS(iterations: usize, allocator: std.mem.Allocator) !f64 {
    // Build a scene with several instances
    var scene = Scene.init(allocator);
    defer scene.deinit();

    const model = try createTestModel(allocator, 1000);
    const model_id = try scene.addModel(model);

    // Add 100 instances
    for (0..100) |i| {
        const x: f32 = @floatFromInt(i % 10);
        const z: f32 = @floatFromInt(i / 10);
        const transform = Mat4.translation(x * 2.0, 0.0, z * 2.0);
        _ = try scene.instantiate(model_id, transform, {});
    }

    var acc: usize = 0;
    const start = std.time.nanoTimestamp();
    for (0..iterations) |_| {
        // Force rebuild by invalidating TLAS
        if (scene.tlas) |*tlas| {
            tlas.deinit();
            scene.tlas = null;
        }
        try scene.buildAccelerationStructure();
        if (scene.tlas) |tlas| {
            acc += tlas.root.primitives.items.len;
        }
    }
    const end = std.time.nanoTimestamp();
    doNotOptimizeAway(acc);

    return @as(f64, @floatFromInt(end - start)) / @as(f64, @floatFromInt(iterations));
}

fn benchmarkSceneCastRay(iterations: usize, allocator: std.mem.Allocator, rng: std.Random) !f64 {
    var scene = Scene.init(allocator);
    defer scene.deinit();

    // Create a scene with several models and instances
    const model = try createTestModel(allocator, 1000);
    const model_id = try scene.addModel(model);

    // Add instances in a grid
    for (0..25) |i| {
        const x: f32 = @floatFromInt(i % 5);
        const z: f32 = @floatFromInt(i / 5);
        const transform = Mat4.translation(x * 0.3 - 0.6, 0.0, z * 0.3 - 0.6);
        _ = try scene.instantiate(model_id, transform, {});
    }

    try scene.buildAccelerationStructure();

    // Pre-generate random rays
    var rays: [1024]Ray = undefined;
    for (0..1024) |j| {
        rays[j] = .{
            .origin = Vec3.fromArray(&.{
                randomFloatRange(f32, rng, -1.0, 1.0),
                randomFloatRange(f32, rng, 1.0, 5.0),
                randomFloatRange(f32, rng, -1.0, 1.0),
            }),
            .direction = Vec3.fromArray(&.{
                randomFloatRange(f32, rng, -0.3, 0.3),
                -1.0,
                randomFloatRange(f32, rng, -0.3, 0.3),
            }).normalized(),
        };
    }

    var hits = std.ArrayListUnmanaged(Scene.Hit){};
    defer hits.deinit(allocator);

    var hit_count: usize = 0;
    const start = std.time.nanoTimestamp();
    for (0..iterations) |i| {
        hits.clearRetainingCapacity();
        try scene.castRay(rays[i % 1024], .forward, .closest, &hits);
        hit_count += hits.items.len;
    }
    const end = std.time.nanoTimestamp();
    doNotOptimizeAway(hit_count);

    return @as(f64, @floatFromInt(end - start)) / @as(f64, @floatFromInt(iterations));
}

fn benchmarkSceneCastRayAll(iterations: usize, allocator: std.mem.Allocator, rng: std.Random) !f64 {
    var scene = Scene.init(allocator);
    defer scene.deinit();

    const model = try createTestModel(allocator, 500);
    const model_id = try scene.addModel(model);

    // Stack instances to get multiple hits per ray
    for (0..10) |i| {
        const y: f32 = @floatFromInt(i);
        const transform = Mat4.translation(0.0, y * 0.2, 0.0);
        _ = try scene.instantiate(model_id, transform, {});
    }

    try scene.buildAccelerationStructure();

    // Rays pointing down through the stack
    var rays: [256]Ray = undefined;
    for (0..256) |j| {
        rays[j] = .{
            .origin = Vec3.fromArray(&.{
                randomFloatRange(f32, rng, 0.2, 0.8),
                5.0,
                randomFloatRange(f32, rng, 0.2, 0.8),
            }),
            .direction = Vec3.fromArray(&.{ 0, -1, 0 }),
        };
    }

    var hits = std.ArrayListUnmanaged(Scene.Hit){};
    defer hits.deinit(allocator);

    var total_hits: usize = 0;
    const start = std.time.nanoTimestamp();
    for (0..iterations) |i| {
        hits.clearRetainingCapacity();
        try scene.castRay(rays[i % 256], .forward, .all, &hits);
        total_hits += hits.items.len;
    }
    const end = std.time.nanoTimestamp();
    doNotOptimizeAway(total_hits);

    return @as(f64, @floatFromInt(end - start)) / @as(f64, @floatFromInt(iterations));
}

fn benchmarkSceneGetHitNormal(iterations: usize, allocator: std.mem.Allocator, rng: std.Random) !f64 {
    var scene = Scene.init(allocator);
    defer scene.deinit();

    const model = try createTestModel(allocator, 1000);
    const model_id = try scene.addModel(model);
    _ = try scene.instantiate(model_id, Mat4.identity(), {});

    try scene.buildAccelerationStructure();

    // Generate some valid hits
    var hits_list = std.ArrayListUnmanaged(Scene.Hit){};
    defer hits_list.deinit(allocator);

    const ray = Ray{
        .origin = Vec3.fromArray(&.{ 0.5, 1.0, 0.5 }),
        .direction = Vec3.fromArray(&.{ 0, -1, 0 }),
    };
    try scene.castRay(ray, .forward, .closest, &hits_list);

    if (hits_list.items.len == 0) {
        // Fallback: create a synthetic hit
        _ = rng;
        return 0.0;
    }

    const hit = hits_list.items[0];

    var acc: f32 = 0;
    const start = std.time.nanoTimestamp();
    for (0..iterations) |_| {
        const normal = scene.getHitNormal(hit);
        acc += normal.data[1];
    }
    const end = std.time.nanoTimestamp();
    doNotOptimizeAway(acc);

    return @as(f64, @floatFromInt(end - start)) / @as(f64, @floatFromInt(iterations));
}

// ============================================================================
// Main
// ============================================================================

pub fn main() !void {
    const args = try std.process.argsAlloc(std.heap.page_allocator);
    defer std.process.argsFree(std.heap.page_allocator, args);

    const parsed = common.parseArgs(args);
    const iterations = parsed.iterations;

    var prng = common.initRng();
    const rng = prng.random();

    const allocator = std.heap.page_allocator;

    // Camera benchmarks
    printHeader("Camera API Benchmarks", iterations);

    printRow("Camera.fromFOV", benchmarkCameraFromFOV(iterations));
    printRow("Camera.fromIntrinsics", benchmarkCameraFromIntrinsics(iterations));
    printRow("Camera.getRay", benchmarkCameraGetRay(iterations, rng));
    printRow("Camera.lookAt", benchmarkCameraLookAt(iterations, rng));
    printRow("Camera.getKInverse", benchmarkCameraGetKInverse(iterations));

    printFooter();

    // Scene benchmarks (fewer iterations due to higher cost)
    const scene_iterations = @min(iterations, 1000);
    printHeader("Scene API Benchmarks", scene_iterations);

    printRow("Scene.addModel (100 tris)", try benchmarkSceneAddModel(scene_iterations, allocator));
    printRow("Scene.instantiate", try benchmarkSceneInstantiate(scene_iterations, allocator, rng));

    const build_iterations = @min(iterations, 100);
    printRow("Scene.buildAccelerationStructure", try benchmarkSceneBuildAS(build_iterations, allocator));

    printRow("Scene.castRay (closest)", try benchmarkSceneCastRay(scene_iterations, allocator, rng));
    printRow("Scene.castRay (all)", try benchmarkSceneCastRayAll(scene_iterations, allocator, rng));
    printRow("Scene.getHitNormal", try benchmarkSceneGetHitNormal(iterations, allocator, rng));

    printFooter();
}
