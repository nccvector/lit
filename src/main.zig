const std = @import("std");
const lit = @import("lit");

const Vec3 = lit.Vec3;
const Mat4 = lit.Mat4;
const Scene = lit.Scene;
const Camera32 = lit.Camera32;
const Ray = lit.Ray;

// Primitives
const Sphere = lit.Sphere;
const Cube = lit.Cube;
const Cylinder = lit.Cylinder;
const Cone = lit.Cone;
const Pyramid = lit.Pyramid;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("lit - Raytracing Demo with Primitives\n", .{});
    std.debug.print("======================================\n\n", .{});

    // Create scene
    var scene = Scene.init(allocator);
    defer scene.deinit();

    // Try to load a model - check multiple locations
    const model_paths = [_][]const u8{
        "assets/bunny.obj",
        "../chad/resources/bunny.obj",
    };

    var has_bunny = false;
    for (model_paths) |model_path| {
        const model_id = scene.loadOBJ(model_path, 1.0) catch |err| {
            std.debug.print("Could not load '{s}': {}\n", .{ model_path, err });
            continue;
        };
        std.debug.print("Loaded model: {s}\n", .{model_path});
        _ = try scene.instantiate(model_id, Mat4.identity());
        has_bunny = true;
        break;
    }

    if (!has_bunny) {
        std.debug.print("No bunny model found, rendering primitives only.\n", .{});
    }

    // Define primitives around the bunny (or at origin if no bunny)
    // Bunny is centered around (0, 0.11, 0) with size ~0.16
    const primitives = .{
        // Sphere to the left of the bunny
        .{ .shape = Sphere.init(Vec3.fromArray(&.{ -0.15, 0.05, 0 }), 0.04), .color = .{ 255, 100, 100 } },
        // Cube to the right of the bunny
        .{ .shape = Cube.initUnit(Vec3.fromArray(&.{ 0.15, 0.05, 0 }), 0.06), .color = .{ 100, 255, 100 } },
        // Cylinder behind and to the left
        .{ .shape = Cylinder.init(Vec3.fromArray(&.{ -0.1, 0.04, -0.1 }), 0.025, 0.08), .color = .{ 100, 100, 255 } },
        // Cone behind and to the right
        .{ .shape = Cone.init(Vec3.fromArray(&.{ 0.1, 0.0, -0.1 }), 0.03, 0.08), .color = .{ 255, 255, 100 } },
        // Pyramid in front
        .{ .shape = Pyramid.init(Vec3.fromArray(&.{ 0, 0.0, 0.12 }), 0.05, 0.06), .color = .{ 255, 100, 255 } },
    };

    std.debug.print("Added {} primitives to the scene\n\n", .{primitives.len});

    // Create and setup camera
    var camera = Camera32.fromFOV(60.0, 640, 480);
    camera.lookAt(
        Vec3.fromArray(&.{ 0, 0.15, 0.35 }), // eye - slightly higher and further back to see all primitives
        Vec3.fromArray(&.{ 0, 0.08, 0 }), // target - center of scene
        Vec3.fromArray(&.{ 0, 1, 0 }), // up
    );

    std.debug.print("Camera setup:\n", .{});
    std.debug.print("  Resolution: {}x{}\n", .{ camera.image_width, camera.image_height });
    std.debug.print("  FOV: 60 degrees\n\n", .{});

    // Build acceleration structure for mesh models
    if (has_bunny) {
        std.debug.print("Building acceleration structure for mesh...\n", .{});
        try scene.buildAccelerationStructure();
    }

    // Allocate image buffer
    try camera.allocateImageBuffer(allocator);
    defer camera.freeImageBuffer();

    // Custom render loop that combines mesh and primitive intersection
    std.debug.print("Rendering...\n", .{});
    const start_time = std.time.milliTimestamp();

    for (0..camera.image_height) |y| {
        for (0..camera.image_width) |x| {
            const pixel_x: f32 = @floatFromInt(x);
            const pixel_y: f32 = @floatFromInt(y);

            // Generate ray for this pixel
            const ray = camera.getRay(pixel_x + 0.5, pixel_y + 0.5);

            var best_t: f32 = std.math.inf(f32);
            var best_normal: Vec3 = undefined;
            var best_color: [3]u8 = .{ 0, 0, 0 };

            // Test mesh models (bunny)
            if (has_bunny) {
                var hits = try scene.castRay(ray, .forward, .closest);
                defer hits.deinit(allocator);

                if (hits.items.len > 0) {
                    const hit = hits.items[0];
                    if (hit.t < best_t) {
                        best_t = hit.t;
                        best_normal = scene.getHitNormal(hit);
                        // Use normal visualization for bunny (cyan-ish)
                        best_color = .{ 128, 180, 200 };
                    }
                }
            }

            // Test each primitive
            inline for (primitives) |prim| {
                if (prim.shape.intersectRay(ray)) |hit| {
                    if (hit.t < best_t) {
                        best_t = hit.t;
                        best_normal = hit.normal;
                        best_color = prim.color;
                    }
                }
            }

            // Shade the pixel
            if (best_t < std.math.inf(f32)) {
                // Simple diffuse shading with a light from upper-right-front
                const light_dir = Vec3.fromArray(&.{ 0.5, 0.7, 0.5 }).normalized();
                const ndotl = @max(0.0, best_normal.dotProduct(light_dir));
                const ambient: f32 = 0.3;
                const shade = ambient + (1.0 - ambient) * ndotl;

                const r: u8 = @intFromFloat(@min(255.0, @as(f32, @floatFromInt(best_color[0])) * shade));
                const g: u8 = @intFromFloat(@min(255.0, @as(f32, @floatFromInt(best_color[1])) * shade));
                const b: u8 = @intFromFloat(@min(255.0, @as(f32, @floatFromInt(best_color[2])) * shade));

                camera.setPixel(@intCast(x), @intCast(y), r, g, b);
            } else {
                // Background gradient (sky blue to white)
                const t_bg = @as(f32, @floatFromInt(y)) / @as(f32, @floatFromInt(camera.image_height));
                const r: u8 = @intFromFloat(255.0 * (1.0 - t_bg) + 135.0 * t_bg);
                const g: u8 = @intFromFloat(255.0 * (1.0 - t_bg) + 206.0 * t_bg);
                const b: u8 = @intFromFloat(255.0 * (1.0 - t_bg) + 235.0 * t_bg);
                camera.setPixel(@intCast(x), @intCast(y), r, g, b);
            }
        }
    }

    const end_time = std.time.milliTimestamp();
    std.debug.print("  Render time: {} ms\n\n", .{end_time - start_time});

    // Save output
    const output_path = "output.ppm";
    try lit.writePPM(output_path, camera.image_buffer.?, camera.image_width, camera.image_height);
    std.debug.print("Output saved to: {s}\n", .{output_path});
}

fn runTestScene(allocator: std.mem.Allocator) !void {
    var scene = Scene.init(allocator);
    defer scene.deinit();

    // Create a simple model with two triangles (a quad)
    var model = lit.Model.init(allocator);

    const vertices = [_][3]f32{
        // Triangle 1 (front-facing)
        .{ -0.5, -0.5, 0.0 },
        .{ 0.5, -0.5, 0.0 },
        .{ 0.0, 0.5, 0.0 },
        // Triangle 2 (tilted)
        .{ -0.3, -0.3, 0.2 },
        .{ 0.3, -0.3, 0.2 },
        .{ 0.0, 0.3, -0.2 },
    };
    const indices = [_]u32{ 0, 1, 2, 3, 4, 5 };

    const mesh = try lit.Mesh.initCopy(allocator, &vertices, &indices);
    try model.addMesh(mesh);

    const model_id = try scene.addModel(model);
    std.debug.print("Created test model with {} triangles\n", .{model.meshes.items[0].triangleCount()});

    // Instance the model
    _ = try scene.instantiate(model_id, Mat4.identity());

    // Setup camera - positioned to look at the triangles at z=0 and z=0.2
    var camera = Camera32.fromFOV(60.0, 320, 240);
    camera.lookAt(
        Vec3.fromArray(&.{ 0, 0, 2 }), // eye - in front of triangles
        Vec3.fromArray(&.{ 0, 0, 0 }), // target - center of scene
        Vec3.fromArray(&.{ 0, 1, 0 }), // up
    );
    _ = try scene.addCamera(camera);

    std.debug.print("Camera resolution: {}x{}\n\n", .{ camera.image_width, camera.image_height });

    // Build acceleration structure
    std.debug.print("Building acceleration structure...\n", .{});
    try scene.buildAccelerationStructure();

    // Render
    std.debug.print("Rendering...\n", .{});
    const start_time = std.time.milliTimestamp();
    try scene.render();
    const end_time = std.time.milliTimestamp();
    std.debug.print("Render time: {} ms\n\n", .{end_time - start_time});

    // Save output
    const output_path = "test_output.ppm";
    const cam = scene.getActiveCamera().?;
    try lit.writePPM(output_path, cam.image_buffer.?, cam.image_width, cam.image_height);
    std.debug.print("Output saved to: {s}\n", .{output_path});
}

test "basic rendering" {
    const allocator = std.testing.allocator;

    var test_scene = Scene.init(allocator);
    defer test_scene.deinit();

    // Create a simple triangle model
    var model = lit.Model.init(allocator);

    const vertices = [_][3]f32{
        .{ -1.0, -1.0, -2.0 },
        .{ 1.0, -1.0, -2.0 },
        .{ 0.0, 1.0, -2.0 },
    };
    const indices = [_]u32{ 0, 1, 2 };

    const mesh = try lit.Mesh.initCopy(allocator, &vertices, &indices);
    try model.addMesh(mesh);

    const model_id = try test_scene.addModel(model);
    _ = try test_scene.instantiate(model_id, Mat4.identity());

    // Setup camera
    const test_camera = Camera32.fromFOV(90.0, 64, 64);
    _ = try test_scene.addCamera(test_camera);

    // Render
    try test_scene.render();

    // Check that we got an image
    const cam = test_scene.getActiveCamera().?;
    try std.testing.expect(cam.image_buffer != null);
}

test "camera center 3x3" {
    const allocator = std.testing.allocator;

    // Sphere of radius 1.0 at world origin
    const sphere = Sphere.init(Vec3.fromArray(&.{ 0, 0, 0 }), 2.0);

    // Camera at (5, 0, 0) looking at origin, 45 degree FOV, 3x3 image
    var camera = Camera32.fromFOV(45.0, 3, 3);
    camera.lookAt(
        Vec3.fromArray(&.{ 0, 0, 5 }), // eye - on positive X axis
        Vec3.fromArray(&.{ 0, 0, 0 }), // target - origin (sphere center)
        Vec3.fromArray(&.{ 0, 1, 0 }), // up
    );

    // Allocate image buffer
    try camera.allocateImageBuffer(allocator);
    defer camera.freeImageBuffer();

    // Render the 3x3 image
    for (0..camera.image_height) |y| {
        for (0..camera.image_width) |x| {
            const pixel_x: f32 = @floatFromInt(x);
            const pixel_y: f32 = @floatFromInt(y);

            // Generate ray for this pixel
            // Note: principal point is at integer coordinates (cx, cy) = ((w-1)/2, (h-1)/2)
            const ray = camera.getRay(pixel_x, pixel_y);

            // Test sphere intersection
            if (sphere.intersectRay(ray)) |hit| {
                // Shade based on normal (white color with normal-based shading)
                // Light direction pointing from camera toward sphere
                const light_dir = Vec3.fromArray(&.{ 0, 0, 1 }).normalized();
                const ndotl = @max(0.0, hit.normal.dotProduct(light_dir));
                const shade = @abs(ndotl);

                const intensity: u8 = @intFromFloat(@min(255.0, 255.0 * shade));
                camera.setPixel(@intCast(x), @intCast(y), intensity, intensity, intensity);
            } else {
                // Black background
                camera.setPixel(@intCast(x), @intCast(y), 0, 0, 0);
            }
        }
    }

    // Save output to PPM file
    try lit.writePPM("camera_center_3x3.ppm", camera.image_buffer.?, camera.image_width, camera.image_height);

    // Helper to get pixel intensity (R channel, since grayscale R=G=B)
    const buf = camera.image_buffer.?;
    const getPixel = struct {
        fn get(buffer: []u8, x: usize, y: usize, width: usize) u8 {
            const idx = (y * width + x) * 3;
            return buffer[idx];
        }
    }.get;

    // Verify plus shape pattern:
    // The center cross (vertical and horizontal lines) should hit the sphere (non-zero)
    // The corners should miss the sphere (zero)

    // Center pixel (1,1) - must hit
    try std.testing.expect(getPixel(buf, 1, 1, 3) > 0);

    // Horizontal line through center: (0,1) and (2,1) - must hit
    try std.testing.expect(getPixel(buf, 0, 1, 3) > 0);
    try std.testing.expect(getPixel(buf, 2, 1, 3) > 0);

    // Vertical line through center: (1,0) and (1,2) - must hit
    try std.testing.expect(getPixel(buf, 1, 0, 3) > 0);
    try std.testing.expect(getPixel(buf, 1, 2, 3) > 0);

    // Corners should be zero (miss the sphere)
    try std.testing.expectEqual(@as(u8, 0), getPixel(buf, 0, 0, 3)); // top-left
    try std.testing.expectEqual(@as(u8, 0), getPixel(buf, 2, 0, 3)); // top-right
    try std.testing.expectEqual(@as(u8, 0), getPixel(buf, 0, 2, 3)); // bottom-left
    try std.testing.expectEqual(@as(u8, 0), getPixel(buf, 2, 2, 3)); // bottom-right
}

test "camera center 4x4" {
    const allocator = std.testing.allocator;

    // Sphere of radius 2.0 at world origin
    const sphere = Sphere.init(Vec3.fromArray(&.{ 0, 0, 0 }), 2.0);

    // Camera at (0, 0, 5) looking at origin, 45 degree FOV, 4x4 image
    var camera = Camera32.fromFOV(45.0, 4, 4);
    camera.lookAt(
        Vec3.fromArray(&.{ 0, 0, 5 }), // eye - on positive Z axis
        Vec3.fromArray(&.{ 0, 0, 0 }), // target - origin (sphere center)
        Vec3.fromArray(&.{ 0, 1, 0 }), // up
    );

    // Allocate image buffer
    try camera.allocateImageBuffer(allocator);
    defer camera.freeImageBuffer();

    // Render the 4x4 image
    for (0..camera.image_height) |y| {
        for (0..camera.image_width) |x| {
            const pixel_x: f32 = @floatFromInt(x);
            const pixel_y: f32 = @floatFromInt(y);

            // Generate ray for this pixel
            const ray = camera.getRay(pixel_x, pixel_y);

            // Test sphere intersection
            if (sphere.intersectRay(ray)) |hit| {
                // Shade based on normal
                const light_dir = Vec3.fromArray(&.{ 0, 0, 1 }).normalized();
                const ndotl = @max(0.0, hit.normal.dotProduct(light_dir));
                const shade = @abs(ndotl);

                const intensity: u8 = @intFromFloat(@min(255.0, 255.0 * shade));
                camera.setPixel(@intCast(x), @intCast(y), intensity, intensity, intensity);
            } else {
                // Black background
                camera.setPixel(@intCast(x), @intCast(y), 0, 0, 0);
            }
        }
    }

    // Save output to PPM file
    try lit.writePPM("camera_center_4x4.ppm", camera.image_buffer.?, camera.image_width, camera.image_height);

    // Helper to get pixel intensity (R channel, since grayscale R=G=B)
    const buf = camera.image_buffer.?;
    const getPixel = struct {
        fn get(buffer: []u8, x: usize, y: usize, width: usize) u8 {
            const idx = (y * width + x) * 3;
            return buffer[idx];
        }
    }.get;

    // Verify center 2x2 grid (pixels (1,1), (2,1), (1,2), (2,2)) are non-zero and equal
    const center_tl = getPixel(buf, 1, 1, 4); // top-left of center 2x2
    const center_tr = getPixel(buf, 2, 1, 4); // top-right of center 2x2
    const center_bl = getPixel(buf, 1, 2, 4); // bottom-left of center 2x2
    const center_br = getPixel(buf, 2, 2, 4); // bottom-right of center 2x2

    // All center pixels must be non-zero (hit the sphere)
    try std.testing.expect(center_tl > 0);
    try std.testing.expect(center_tr > 0);
    try std.testing.expect(center_bl > 0);
    try std.testing.expect(center_br > 0);

    // All center pixels must have equal values (symmetric around center)
    try std.testing.expectEqual(center_tl, center_tr);
    try std.testing.expectEqual(center_tl, center_bl);
    try std.testing.expectEqual(center_tl, center_br);
}

test "camera center 8x8" {
    const allocator = std.testing.allocator;

    // Sphere of radius 2.0 at world origin
    const sphere = Sphere.init(Vec3.fromArray(&.{ 0, 0, 0 }), 2.0);

    // Camera at (0, 0, 5) looking at origin, 45 degree FOV, 8x8 image
    var camera = Camera32.fromFOV(45.0, 8, 8);
    camera.lookAt(
        Vec3.fromArray(&.{ 0, 0, 5 }), // eye - on positive Z axis
        Vec3.fromArray(&.{ 0, 0, 0 }), // target - origin (sphere center)
        Vec3.fromArray(&.{ 0, 1, 0 }), // up
    );

    // Allocate image buffer
    try camera.allocateImageBuffer(allocator);
    defer camera.freeImageBuffer();

    // Render the 8x8 image
    for (0..camera.image_height) |y| {
        for (0..camera.image_width) |x| {
            const pixel_x: f32 = @floatFromInt(x);
            const pixel_y: f32 = @floatFromInt(y);

            // Generate ray for this pixel
            const ray = camera.getRay(pixel_x, pixel_y);

            // Test sphere intersection
            if (sphere.intersectRay(ray)) |hit| {
                // Shade based on normal
                const light_dir = Vec3.fromArray(&.{ 0, 0, 1 }).normalized();
                const ndotl = @max(0.0, hit.normal.dotProduct(light_dir));
                const shade = @abs(ndotl);

                const intensity: u8 = @intFromFloat(@min(255.0, 255.0 * shade));
                camera.setPixel(@intCast(x), @intCast(y), intensity, intensity, intensity);
            } else {
                // Black background
                camera.setPixel(@intCast(x), @intCast(y), 0, 0, 0);
            }
        }
    }

    // Save output to PPM file
    try lit.writePPM("camera_center_8x8.ppm", camera.image_buffer.?, camera.image_width, camera.image_height);

    // Helper to get pixel intensity (R channel, since grayscale R=G=B)
    const buf = camera.image_buffer.?;
    const getPixel = struct {
        fn get(buffer: []u8, x: usize, y: usize, width: usize) u8 {
            const idx = (y * width + x) * 3;
            return buffer[idx];
        }
    }.get;

    // Verify center 2x2 grid (pixels (3,3), (4,3), (3,4), (4,4)) are non-zero
    const center_tl = getPixel(buf, 3, 3, 8);
    const center_tr = getPixel(buf, 4, 3, 8);
    const center_bl = getPixel(buf, 3, 4, 8);
    const center_br = getPixel(buf, 4, 4, 8);

    try std.testing.expect(center_tl > 0);
    try std.testing.expect(center_tr > 0);
    try std.testing.expect(center_bl > 0);
    try std.testing.expect(center_br > 0);

    // Verify four quadrants are symmetric
    // For an 8x8 image centered on sphere, pixels should be symmetric across both axes
    // Quadrant mapping: (x, y) should equal (7-x, y), (x, 7-y), and (7-x, 7-y)
    for (0..4) |y| {
        for (0..4) |x| {
            const tl = getPixel(buf, x, y, 8); // top-left quadrant
            const tr = getPixel(buf, 7 - x, y, 8); // top-right quadrant (mirrored horizontally)
            const bl = getPixel(buf, x, 7 - y, 8); // bottom-left quadrant (mirrored vertically)
            const br = getPixel(buf, 7 - x, 7 - y, 8); // bottom-right quadrant (mirrored both)

            try std.testing.expectEqual(tl, tr);
            try std.testing.expectEqual(tl, bl);
            try std.testing.expectEqual(tl, br);
        }
    }
}
