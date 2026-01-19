const std = @import("std");
const lit = @import("lit");

const Vec3 = lit.Vec3;
const Mat4 = lit.Mat4;
const Camera32 = lit.Camera32;
const Ray = lit.Ray;
const Octree = lit.Octree;
const Aabb = lit.Aabb;
const Scene = lit.Scene;
const UserDataBindingTable = lit.UserDataBindingTable;
const IntersectionResult = lit.IntersectionResult;
const PrimId = lit.PrimId;
const HitResult = lit.HitResult;
const Triangle = lit.Triangle;
const Mesh = lit.Mesh;
const Model = lit.Model;

// Primitives
const Sphere = lit.Sphere;
const Cube = lit.Cube;
const Cylinder = lit.Cylinder;
const Cone = lit.Cone;
const Pyramid = lit.Pyramid;

// ============================================================================
// User Data Types
// ============================================================================

/// Combined user data - can be either a triangle or a primitive
const GeometryData = struct {
    geo_type: GeoType,
    color: [3]u8,

    // Triangle data
    mesh: ?*const Mesh = null,
    tri_idx: usize = 0,

    // Primitive data
    sphere: ?Sphere = null,
    cube: ?Cube = null,
    cylinder: ?Cylinder = null,
    cone: ?Cone = null,
    pyramid: ?Pyramid = null,

    const GeoType = enum { triangle, sphere, cube, cylinder, cone, pyramid };

    fn intersectRay(self: *const GeometryData, ray: Ray) ?struct { t: f32, normal: Vec3 } {
        switch (self.geo_type) {
            .triangle => {
                if (self.mesh) |mesh| {
                    const tri = Triangle.fromMesh(mesh, self.tri_idx);
                    if (tri.intersectRay(ray, false)) |hit| {
                        return .{ .t = hit.t, .normal = tri.normalNormalized() };
                    }
                }
                return null;
            },
            .sphere => if (self.sphere) |s| if (s.intersectRay(ray)) |hit| return .{ .t = hit.t, .normal = hit.normal } else return null else return null,
            .cube => if (self.cube) |c| if (c.intersectRay(ray)) |hit| return .{ .t = hit.t, .normal = hit.normal } else return null else return null,
            .cylinder => if (self.cylinder) |c| if (c.intersectRay(ray)) |hit| return .{ .t = hit.t, .normal = hit.normal } else return null else return null,
            .cone => if (self.cone) |c| if (c.intersectRay(ray)) |hit| return .{ .t = hit.t, .normal = hit.normal } else return null else return null,
            .pyramid => if (self.pyramid) |p| if (p.intersectRay(ray)) |hit| return .{ .t = hit.t, .normal = hit.normal } else return null else return null,
        }
    }

    fn getBounds(self: *const GeometryData) Aabb {
        return switch (self.geo_type) {
            .triangle => blk: {
                if (self.mesh) |mesh| {
                    const tri = Triangle.fromMesh(mesh, self.tri_idx);
                    break :blk tri.bounds();
                }
                break :blk Aabb{ .bmin = Vec3.fromArray(&.{ 0, 0, 0 }), .bmax = Vec3.fromArray(&.{ 0, 0, 0 }) };
            },
            .sphere => self.sphere.?.bounds(),
            .cube => self.cube.?.bounds(),
            .cylinder => self.cylinder.?.bounds(),
            .cone => self.cone.?.bounds(),
            .pyramid => self.pyramid.?.bounds(),
        };
    }
};

// ============================================================================
// Intersection Function (comptime known)
// ============================================================================

/// Intersection function for all geometry types
/// This function is inlined at compile time for maximum performance
fn geometryIntersect(ray: Ray, prim_id: PrimId, udbt: *const UserDataBindingTable(GeometryData)) ?IntersectionResult {
    const geo = udbt.get(prim_id) orelse return null;

    if (geo.intersectRay(ray)) |hit| {
        // Encode normal in u, v, hit_kind
        return .{
            .t = hit.t,
            .u = hit.normal.data[0],
            .v = hit.normal.data[1],
            .hit_kind = @as(u8, @intFromFloat((hit.normal.data[2] + 1.0) * 127.5)),
        };
    }
    return null;
}

// Create Scene type at comptime with our intersection function
const MyScene = Scene(GeometryData, geometryIntersect);

// ============================================================================
// Helper to compute model bounds
// ============================================================================

fn computeModelBounds(model: *const Model) Aabb {
    var bmin = Vec3.fromArray(&.{ std.math.inf(f32), std.math.inf(f32), std.math.inf(f32) });
    var bmax = Vec3.fromArray(&.{ -std.math.inf(f32), -std.math.inf(f32), -std.math.inf(f32) });

    for (model.meshes.items) |*mesh| {
        for (mesh.vertices) |v| {
            const vert = Vec3.fromArray(&v);
            bmin = bmin.min(vert);
            bmax = bmax.max(vert);
        }
    }

    return .{ .bmin = bmin, .bmax = bmax };
}

// ============================================================================
// Main
// ============================================================================

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("lit - Raytracing Demo with Comptime Scene\n", .{});
    std.debug.print("==========================================\n\n", .{});

    // Create User Data Binding Table
    var udbt = UserDataBindingTable(GeometryData).init(allocator);
    defer udbt.deinit();

    var next_prim_id: PrimId = 0;

    // Try to load a model
    const model_paths = [_][]const u8{
        "assets/bunny.obj",
        "../chad/resources/bunny.obj",
    };

    var model: ?Model = null;
    defer if (model) |*m| m.deinit();

    for (model_paths) |model_path| {
        model = lit.loadOBJ(allocator, model_path, 1.0) catch |err| {
            std.debug.print("Could not load '{s}': {}\n", .{ model_path, err });
            continue;
        };
        std.debug.print("Loaded model: {s}\n", .{model_path});
        break;
    }

    // Add triangles from model to UDBT
    var mesh_triangle_count: usize = 0;
    if (model) |*m| {
        for (m.meshes.items) |*mesh| {
            const tri_count = mesh.triangleCount();
            for (0..tri_count) |tri_idx| {
                try udbt.put(next_prim_id, .{
                    .geo_type = .triangle,
                    .color = .{ 128, 180, 200 }, // Cyan-ish for bunny
                    .mesh = mesh,
                    .tri_idx = tri_idx,
                });
                next_prim_id += 1;
                mesh_triangle_count += 1;
            }
        }
        std.debug.print("Added {} triangles from mesh\n", .{mesh_triangle_count});
    } else {
        std.debug.print("No model loaded, rendering primitives only.\n", .{});
    }

    // Add primitives to UDBT
    const primitive_start_id = next_prim_id;

    try udbt.put(next_prim_id, .{
        .geo_type = .sphere,
        .color = .{ 255, 100, 100 },
        .sphere = Sphere.init(Vec3.fromArray(&.{ -0.15, 0.05, 0 }), 0.04),
    });
    next_prim_id += 1;

    try udbt.put(next_prim_id, .{
        .geo_type = .cube,
        .color = .{ 100, 255, 100 },
        .cube = Cube.initUnit(Vec3.fromArray(&.{ 0.15, 0.05, 0 }), 0.06),
    });
    next_prim_id += 1;

    try udbt.put(next_prim_id, .{
        .geo_type = .cylinder,
        .color = .{ 100, 100, 255 },
        .cylinder = Cylinder.init(Vec3.fromArray(&.{ -0.1, 0.04, -0.1 }), 0.025, 0.08),
    });
    next_prim_id += 1;

    try udbt.put(next_prim_id, .{
        .geo_type = .cone,
        .color = .{ 255, 255, 100 },
        .cone = Cone.init(Vec3.fromArray(&.{ 0.1, 0.0, -0.1 }), 0.03, 0.08),
    });
    next_prim_id += 1;

    try udbt.put(next_prim_id, .{
        .geo_type = .pyramid,
        .color = .{ 255, 100, 255 },
        .pyramid = Pyramid.init(Vec3.fromArray(&.{ 0, 0.0, 0.12 }), 0.05, 0.06),
    });
    next_prim_id += 1;

    std.debug.print("Added {} primitives\n", .{next_prim_id - primitive_start_id});
    std.debug.print("Total geometry count: {}\n\n", .{next_prim_id});

    // Build BLAS
    // Compute scene bounds
    var scene_min = Vec3.fromArray(&.{ std.math.inf(f32), std.math.inf(f32), std.math.inf(f32) });
    var scene_max = Vec3.fromArray(&.{ -std.math.inf(f32), -std.math.inf(f32), -std.math.inf(f32) });

    var udbt_iter = udbt.data.iterator();
    while (udbt_iter.next()) |entry| {
        const bounds = entry.value_ptr.getBounds();
        scene_min = scene_min.min(bounds.bmin);
        scene_max = scene_max.max(bounds.bmax);
    }

    // Expand bounds slightly
    const epsilon: @Vector(3, f32) = @splat(0.01);
    scene_min.data -= epsilon;
    scene_max.data += epsilon;

    var blas = Octree.init(allocator, .{
        .bmin = scene_min,
        .bmax = scene_max,
    }, .{ .max_depth = 10, .max_primitives_per_node = 8 });
    defer blas.deinit();

    // Insert all geometry into BLAS
    var insert_iter = udbt.data.iterator();
    while (insert_iter.next()) |entry| {
        try blas.insert(entry.key_ptr.*, entry.value_ptr.getBounds());
    }

    std.debug.print("Built BLAS with {} primitives\n", .{udbt.data.count()});

    // Create scene with comptime-known intersection function
    var scene = MyScene.init(allocator, &udbt);
    defer scene.deinit();

    // Add instance (identity transform)
    _ = try scene.addInstance(&blas, Mat4.identity());
    try scene.buildTLAS();

    std.debug.print("Built TLAS with 1 instance\n\n", .{});

    // Create and setup camera
    var camera = Camera32.fromFOV(60.0, 640, 480);
    camera.lookAt(
        Vec3.fromArray(&.{ 0, 0.15, 0.35 }),
        Vec3.fromArray(&.{ 0, 0.08, 0 }),
        Vec3.fromArray(&.{ 0, 1, 0 }),
    );

    std.debug.print("Camera setup:\n", .{});
    std.debug.print("  Resolution: {}x{}\n", .{ camera.image_width, camera.image_height });
    std.debug.print("  FOV: 60 degrees\n\n", .{});

    // Allocate image buffer
    try camera.allocateImageBuffer(allocator);
    defer camera.freeImageBuffer();

    // Render
    std.debug.print("Rendering...\n", .{});
    const start_time = std.time.milliTimestamp();

    var hits = std.ArrayListUnmanaged(HitResult){};
    defer hits.deinit(allocator);

    for (0..camera.image_height) |y| {
        for (0..camera.image_width) |x| {
            const pixel_x: f32 = @floatFromInt(x);
            const pixel_y: f32 = @floatFromInt(y);

            const ray = camera.getRay(pixel_x + 0.5, pixel_y + 0.5);

            hits.clearRetainingCapacity();
            try scene.castRay(ray, .forward, .closest, &hits);

            if (hits.items.len > 0) {
                const hit = hits.items[0];

                // Decode normal from u, v, hit_kind
                const normal = Vec3.fromArray(&.{
                    hit.u,
                    hit.v,
                    (@as(f32, @floatFromInt(hit.hit_kind)) / 127.5) - 1.0,
                }).normalized();

                // Get color from UDBT
                const geo = udbt.get(hit.prim_id) orelse continue;
                const base_color = geo.color;

                // Simple diffuse shading
                const light_dir = Vec3.fromArray(&.{ 0.5, 0.7, 0.5 }).normalized();
                const ndotl = @max(0.0, normal.dotProduct(light_dir));
                const ambient: f32 = 0.3;
                const shade = ambient + (1.0 - ambient) * ndotl;

                const r: u8 = @intFromFloat(@min(255.0, @as(f32, @floatFromInt(base_color[0])) * shade));
                const g: u8 = @intFromFloat(@min(255.0, @as(f32, @floatFromInt(base_color[1])) * shade));
                const b: u8 = @intFromFloat(@min(255.0, @as(f32, @floatFromInt(base_color[2])) * shade));

                camera.setPixel(@intCast(x), @intCast(y), r, g, b);
            } else {
                // Background gradient
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

// ============================================================================
// Tests - Camera tests that don't use Scene
// ============================================================================

test "camera center 3x3" {
    const allocator = std.testing.allocator;

    const sphere = Sphere.init(Vec3.fromArray(&.{ 0, 0, 0 }), 2.0);

    var camera = Camera32.fromFOV(45.0, 3, 3);
    camera.lookAt(
        Vec3.fromArray(&.{ 0, 0, 5 }),
        Vec3.fromArray(&.{ 0, 0, 0 }),
        Vec3.fromArray(&.{ 0, 1, 0 }),
    );

    try camera.allocateImageBuffer(allocator);
    defer camera.freeImageBuffer();

    for (0..camera.image_height) |y| {
        for (0..camera.image_width) |x| {
            const pixel_x: f32 = @floatFromInt(x);
            const pixel_y: f32 = @floatFromInt(y);

            const ray = camera.getRay(pixel_x, pixel_y);

            if (sphere.intersectRay(ray)) |hit| {
                const light_dir = Vec3.fromArray(&.{ 0, 0, 1 }).normalized();
                const ndotl = @max(0.0, hit.normal.dotProduct(light_dir));
                const shade = @abs(ndotl);

                const gray: u8 = @intFromFloat(@round(255.0 * shade));
                camera.setPixel(@intCast(x), @intCast(y), gray, gray, gray);
            }
        }
    }

    const center_idx: usize = 1 * camera.image_width * 3 + 1 * 3;
    const center_r = camera.image_buffer.?[center_idx];
    try std.testing.expect(center_r > 200);
}

test "camera center 4x4" {
    const allocator = std.testing.allocator;

    const sphere = Sphere.init(Vec3.fromArray(&.{ 0, 0, 0 }), 2.0);

    var camera = Camera32.fromFOV(45.0, 4, 4);
    camera.lookAt(
        Vec3.fromArray(&.{ 0, 0, 5 }),
        Vec3.fromArray(&.{ 0, 0, 0 }),
        Vec3.fromArray(&.{ 0, 1, 0 }),
    );

    try camera.allocateImageBuffer(allocator);
    defer camera.freeImageBuffer();

    for (0..camera.image_height) |y| {
        for (0..camera.image_width) |x| {
            const pixel_x: f32 = @floatFromInt(x);
            const pixel_y: f32 = @floatFromInt(y);

            const ray = camera.getRay(pixel_x, pixel_y);

            if (sphere.intersectRay(ray)) |hit| {
                const light_dir = Vec3.fromArray(&.{ 0, 0, 1 }).normalized();
                const ndotl = @max(0.0, hit.normal.dotProduct(light_dir));
                const shade = @abs(ndotl);

                const gray: u8 = @intFromFloat(@round(255.0 * shade));
                camera.setPixel(@intCast(x), @intCast(y), gray, gray, gray);
            }
        }
    }

    const center_pixels = [_]struct { x: usize, y: usize }{
        .{ .x = 1, .y = 1 },
        .{ .x = 2, .y = 1 },
        .{ .x = 1, .y = 2 },
        .{ .x = 2, .y = 2 },
    };

    for (center_pixels) |p| {
        const idx: usize = p.y * camera.image_width * 3 + p.x * 3;
        const r = camera.image_buffer.?[idx];
        try std.testing.expect(r > 200);
    }
}

test "camera center 8x8" {
    const allocator = std.testing.allocator;

    const sphere = Sphere.init(Vec3.fromArray(&.{ 0, 0, 0 }), 2.0);

    var camera = Camera32.fromFOV(45.0, 8, 8);
    camera.lookAt(
        Vec3.fromArray(&.{ 0, 0, 5 }),
        Vec3.fromArray(&.{ 0, 0, 0 }),
        Vec3.fromArray(&.{ 0, 1, 0 }),
    );

    try camera.allocateImageBuffer(allocator);
    defer camera.freeImageBuffer();

    for (0..camera.image_height) |y| {
        for (0..camera.image_width) |x| {
            const pixel_x: f32 = @floatFromInt(x);
            const pixel_y: f32 = @floatFromInt(y);

            const ray = camera.getRay(pixel_x, pixel_y);

            if (sphere.intersectRay(ray)) |hit| {
                const light_dir = Vec3.fromArray(&.{ 0, 0, 1 }).normalized();
                const ndotl = @max(0.0, hit.normal.dotProduct(light_dir));
                const shade = @abs(ndotl);

                const gray: u8 = @intFromFloat(@round(255.0 * shade));
                camera.setPixel(@intCast(x), @intCast(y), gray, gray, gray);
            }
        }
    }

    const center_idx: usize = 4 * camera.image_width * 3 + 4 * 3;
    const center_r = camera.image_buffer.?[center_idx];
    try std.testing.expect(center_r > 200);
}

test "camera center 16x16" {
    const allocator = std.testing.allocator;

    const sphere = Sphere.init(Vec3.fromArray(&.{ 0, 0, 0 }), 2.0);

    var camera = Camera32.fromFOV(45.0, 16, 16);
    camera.lookAt(
        Vec3.fromArray(&.{ 0, 0, 5 }),
        Vec3.fromArray(&.{ 0, 0, 0 }),
        Vec3.fromArray(&.{ 0, 1, 0 }),
    );

    try camera.allocateImageBuffer(allocator);
    defer camera.freeImageBuffer();

    for (0..camera.image_height) |y| {
        for (0..camera.image_width) |x| {
            const pixel_x: f32 = @floatFromInt(x);
            const pixel_y: f32 = @floatFromInt(y);

            const ray = camera.getRay(pixel_x, pixel_y);

            if (sphere.intersectRay(ray)) |hit| {
                const light_dir = Vec3.fromArray(&.{ 0, 0, 1 }).normalized();
                const ndotl = @max(0.0, hit.normal.dotProduct(light_dir));
                const shade = @abs(ndotl);

                const gray: u8 = @intFromFloat(@round(255.0 * shade));
                camera.setPixel(@intCast(x), @intCast(y), gray, gray, gray);
            }
        }
    }

    const center_idx: usize = 8 * camera.image_width * 3 + 8 * 3;
    const center_r = camera.image_buffer.?[center_idx];
    try std.testing.expect(center_r > 200);
}

test "camera center 32x32" {
    const allocator = std.testing.allocator;

    const sphere = Sphere.init(Vec3.fromArray(&.{ 0, 0, 0 }), 2.0);

    var camera = Camera32.fromFOV(45.0, 32, 32);
    camera.lookAt(
        Vec3.fromArray(&.{ 0, 0, 5 }),
        Vec3.fromArray(&.{ 0, 0, 0 }),
        Vec3.fromArray(&.{ 0, 1, 0 }),
    );

    try camera.allocateImageBuffer(allocator);
    defer camera.freeImageBuffer();

    for (0..camera.image_height) |y| {
        for (0..camera.image_width) |x| {
            const pixel_x: f32 = @floatFromInt(x);
            const pixel_y: f32 = @floatFromInt(y);

            const ray = camera.getRay(pixel_x, pixel_y);

            if (sphere.intersectRay(ray)) |hit| {
                const light_dir = Vec3.fromArray(&.{ 0, 0, 1 }).normalized();
                const ndotl = @max(0.0, hit.normal.dotProduct(light_dir));
                const shade = @abs(ndotl);

                const gray: u8 = @intFromFloat(@round(255.0 * shade));
                camera.setPixel(@intCast(x), @intCast(y), gray, gray, gray);
            }
        }
    }

    const center_idx: usize = 16 * camera.image_width * 3 + 16 * 3;
    const center_r = camera.image_buffer.?[center_idx];
    try std.testing.expect(center_r > 200);
}
