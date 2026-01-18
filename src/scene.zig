const std = @import("std");
const lmao = @import("lmao");
const chad = @import("chad");

const geometry = chad.geometry;
const octree_mod = chad.octree;
const objloader = chad.objloader;

const Vec3 = lmao.Vec3f;
const Mat4 = lmao.Mat4f;
const Ray = geometry.Ray;
const RayHit = geometry.RayHit;
const Triangle = geometry.Triangle;
const Aabb = geometry.Aabb;
const Mesh = geometry.Mesh;
const Model = geometry.Model;
const Octree = octree_mod.Octree;

const Camera = @import("camera.zig").Camera32;

/// Casting direction for ray queries
pub const CastDirection = enum {
    forward, // Only positive t values (in front of ray origin)
    backward, // Only negative t values (behind ray origin)
    both, // Both positive and negative t values
};

/// Filter mode for ray queries
pub const FilterMode = enum {
    closest, // Return only the closest intersection
    all, // Return all intersections
};

/// Unique identifiers
pub const ModelId = u32;
pub const InstanceId = u32;
pub const PrimId = u32;
pub const CameraId = u32;
pub const LightId = u32;

/// Result of a ray cast operation
pub const HitResult = struct {
    model_id: ModelId,
    instance_id: InstanceId,
    mesh_idx: u32,
    prim_id: PrimId, // Triangle index within mesh
    t: f32, // Distance along ray
    u: f32, // Barycentric u
    v: f32, // Barycentric v
    user_data: ?*anyopaque = null,

    /// Compute the hit point in world space
    pub fn hitPoint(self: HitResult, ray: Ray) Vec3 {
        return ray.at(self.t);
    }

    /// Get the third barycentric coordinate
    pub fn w(self: HitResult) f32 {
        return 1.0 - self.u - self.v;
    }
};

/// Model instance in the scene
const Instance = struct {
    model_id: ModelId,
    transform: Mat4,
    inv_transform: Mat4,
    bounds: Aabb, // World-space bounds
    user_data: ?*anyopaque = null,
};

/// Bottom-Level Acceleration Structure (per-model)
const BLAS = struct {
    octree: Octree,
    model: *const Model,

    fn init(allocator: std.mem.Allocator, model: *const Model) !BLAS {
        // Compute bounds of the model
        var bounds = computeModelBounds(model);

        // Slightly expand bounds to avoid numerical issues
        const epsilon: @Vector(3, f32) = @splat(0.001);
        bounds.bmin.data -= epsilon;
        bounds.bmax.data += epsilon;

        var tree = Octree.init(allocator, bounds, .{
            .max_depth = 10,
            .max_primitives_per_node = 8,
        });

        // Insert all triangles from all meshes
        var global_prim_id: PrimId = 0;
        for (model.meshes.items) |*mesh| {
            const tri_count = mesh.triangleCount();
            for (0..tri_count) |tri_idx| {
                const tri = Triangle.fromMesh(mesh, tri_idx);
                try tree.insert(global_prim_id, tri.bounds());
                global_prim_id += 1;
            }
        }

        return .{
            .octree = tree,
            .model = model,
        };
    }

    fn deinit(self: *BLAS) void {
        self.octree.deinit();
    }

    /// Ray-model intersection (in model local space)
    fn intersect(self: *const BLAS, allocator: std.mem.Allocator, ray: Ray, direction: CastDirection, filter: FilterMode, results: *std.ArrayListUnmanaged(HitResult), instance_id: InstanceId, model_id: ModelId) !void {
        // Query octree for candidate primitives
        var candidates = std.ArrayListUnmanaged(PrimId){};
        defer candidates.deinit(self.octree.allocator);
        try self.octree.queryRayIntersection(ray, &candidates);

        // Test each candidate triangle
        for (candidates.items) |global_prim_id| {
            // Find which mesh and triangle this primitive belongs to
            var mesh_idx: u32 = 0;
            var local_tri_idx: usize = global_prim_id;
            for (self.model.meshes.items, 0..) |*mesh, idx| {
                const tri_count = mesh.triangleCount();
                if (local_tri_idx < tri_count) {
                    mesh_idx = @intCast(idx);
                    break;
                }
                local_tri_idx -= tri_count;
            }

            const mesh = &self.model.meshes.items[mesh_idx];
            const tri = Triangle.fromMesh(mesh, local_tri_idx);

            const hit_opt: ?RayHit = switch (direction) {
                .forward => tri.intersectRay(ray, false),
                .backward => blk: {
                    // Flip ray direction and check
                    const flipped_ray = Ray{
                        .origin = ray.origin,
                        .direction = ray.direction.negate(),
                    };
                    if (tri.intersectRay(flipped_ray, false)) |h| {
                        break :blk RayHit{ .t = -h.t, .u = h.u, .v = h.v };
                    }
                    break :blk null;
                },
                .both => tri.intersectRayBidirectional(ray),
            };

            if (hit_opt) |hit| {
                const result = HitResult{
                    .model_id = model_id,
                    .instance_id = instance_id,
                    .mesh_idx = mesh_idx,
                    .prim_id = @intCast(local_tri_idx),
                    .t = hit.t,
                    .u = hit.u,
                    .v = hit.v,
                };

                if (filter == .closest) {
                    // Keep only closest
                    if (results.items.len == 0 or @abs(result.t) < @abs(results.items[0].t)) {
                        results.clearRetainingCapacity();
                        try results.append(allocator, result);
                    }
                } else {
                    try results.append(allocator, result);
                }
            }
        }
    }
};

/// Compute the axis-aligned bounding box of a model
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

/// Transform an AABB by a matrix (returns a conservative AABB)
fn transformAabb(bounds: Aabb, transform: Mat4) Aabb {
    // Get the 8 corners of the AABB
    const min = bounds.bmin.toArray();
    const max = bounds.bmax.toArray();

    var new_min = Vec3.fromArray(&.{ std.math.inf(f32), std.math.inf(f32), std.math.inf(f32) });
    var new_max = Vec3.fromArray(&.{ -std.math.inf(f32), -std.math.inf(f32), -std.math.inf(f32) });

    // Transform all 8 corners
    inline for (0..8) |i| {
        const corner = Vec3.fromArray(&.{
            if (i & 1 != 0) max[0] else min[0],
            if (i & 2 != 0) max[1] else min[1],
            if (i & 4 != 0) max[2] else min[2],
        });
        const transformed = transform.transformPoint(corner);
        new_min = new_min.min(transformed);
        new_max = new_max.max(transformed);
    }

    return .{ .bmin = new_min, .bmax = new_max };
}

/// The main Scene struct
pub const Scene = struct {
    allocator: std.mem.Allocator,

    // Model storage (templates)
    models: std.ArrayListUnmanaged(Model) = .empty,
    blas_list: std.ArrayListUnmanaged(BLAS) = .empty,

    // Instance storage
    instances: std.ArrayListUnmanaged(Instance) = .empty,

    // Top-Level Acceleration Structure
    tlas: ?Octree = null,

    // Cameras
    cameras: std.ArrayListUnmanaged(Camera) = .empty,
    active_camera_id: ?CameraId = null,

    pub fn init(allocator: std.mem.Allocator) Scene {
        return .{
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Scene) void {
        // Free TLAS
        if (self.tlas) |*tlas| {
            tlas.deinit();
        }

        // Free BLAS
        for (self.blas_list.items) |*blas| {
            blas.deinit();
        }
        self.blas_list.deinit(self.allocator);

        // Free models
        for (self.models.items) |*model| {
            model.deinit();
        }
        self.models.deinit(self.allocator);

        // Free instances
        self.instances.deinit(self.allocator);

        // Free cameras (and their image buffers)
        for (self.cameras.items) |*cam| {
            cam.freeImageBuffer();
        }
        self.cameras.deinit(self.allocator);
    }

    /// Load an OBJ model from file and add it to the scene
    pub fn loadOBJ(self: *Scene, path: []const u8, scale: ?f32) !ModelId {
        const model = try objloader.ObjLoader.loadModel(self.allocator, path, scale orelse 1.0);
        return self.addModel(model);
    }

    /// Add a model to the scene (takes ownership)
    pub fn addModel(self: *Scene, model: Model) !ModelId {
        const model_id: ModelId = @intCast(self.models.items.len);
        try self.models.append(self.allocator, model);

        // Build BLAS for this model
        const blas = try BLAS.init(self.allocator, &self.models.items[model_id]);
        try self.blas_list.append(self.allocator, blas);

        return model_id;
    }

    /// Create an instance of a model with the given transform
    pub fn instantiate(self: *Scene, model_id: ModelId, transform: Mat4) !InstanceId {
        if (model_id >= self.models.items.len) {
            return error.InvalidModelId;
        }

        const model = &self.models.items[model_id];
        const model_bounds = computeModelBounds(model);
        const world_bounds = transformAabb(model_bounds, transform);

        const inv_transform = transform.inverse() orelse Mat4.identity();

        const instance = Instance{
            .model_id = model_id,
            .transform = transform,
            .inv_transform = inv_transform,
            .bounds = world_bounds,
        };

        const instance_id: InstanceId = @intCast(self.instances.items.len);
        try self.instances.append(self.allocator, instance);

        // Invalidate TLAS (needs rebuild)
        if (self.tlas) |*tlas| {
            tlas.deinit();
            self.tlas = null;
        }

        return instance_id;
    }

    /// Update the transform of an instance
    pub fn setInstanceTransform(self: *Scene, instance_id: InstanceId, transform: Mat4) !void {
        if (instance_id >= self.instances.items.len) {
            return error.InvalidInstanceId;
        }

        var instance = &self.instances.items[instance_id];
        const model = &self.models.items[instance.model_id];
        const model_bounds = computeModelBounds(model);

        instance.transform = transform;
        instance.inv_transform = transform.inverse() orelse Mat4.identity();
        instance.bounds = transformAabb(model_bounds, transform);

        // Invalidate TLAS
        if (self.tlas) |*tlas| {
            tlas.deinit();
            self.tlas = null;
        }
    }

    /// Build or rebuild the Top-Level Acceleration Structure
    pub fn buildAccelerationStructure(self: *Scene) !void {
        if (self.tlas) |*tlas| {
            tlas.deinit();
        }

        if (self.instances.items.len == 0) {
            self.tlas = null;
            return;
        }

        // Compute scene bounds
        var scene_min = Vec3.fromArray(&.{ std.math.inf(f32), std.math.inf(f32), std.math.inf(f32) });
        var scene_max = Vec3.fromArray(&.{ -std.math.inf(f32), -std.math.inf(f32), -std.math.inf(f32) });

        for (self.instances.items) |*instance| {
            scene_min = scene_min.min(instance.bounds.bmin);
            scene_max = scene_max.max(instance.bounds.bmax);
        }

        // Expand bounds slightly
        const epsilon: @Vector(3, f32) = @splat(0.001);
        scene_min.data -= epsilon;
        scene_max.data += epsilon;

        var tlas = Octree.init(self.allocator, .{ .bmin = scene_min, .bmax = scene_max }, .{
            .max_depth = 6,
            .max_primitives_per_node = 4,
        });

        // Insert all instances
        for (self.instances.items, 0..) |*instance, idx| {
            try tlas.insert(@intCast(idx), instance.bounds);
        }

        self.tlas = tlas;
    }

    /// Add a camera to the scene
    pub fn addCamera(self: *Scene, cam: Camera) !CameraId {
        const camera_id: CameraId = @intCast(self.cameras.items.len);
        try self.cameras.append(self.allocator, cam);

        // Set as active if first camera
        if (self.active_camera_id == null) {
            self.active_camera_id = camera_id;
        }

        return camera_id;
    }

    /// Set the active camera
    pub fn setActiveCamera(self: *Scene, camera_id: CameraId) !void {
        if (camera_id >= self.cameras.items.len) {
            return error.InvalidCameraId;
        }
        self.active_camera_id = camera_id;
    }

    /// Get the active camera
    pub fn getActiveCamera(self: *Scene) ?*Camera {
        if (self.active_camera_id) |id| {
            if (id < self.cameras.items.len) {
                return &self.cameras.items[id];
            }
        }
        return null;
    }

    /// Cast a ray and return hit results
    /// Caller must call results.deinit(allocator) when done
    pub fn castRay(
        self: *Scene,
        ray: Ray,
        direction: CastDirection,
        filter: FilterMode,
    ) !std.ArrayListUnmanaged(HitResult) {
        var results = std.ArrayListUnmanaged(HitResult){};

        // Ensure TLAS is built
        if (self.tlas == null) {
            try self.buildAccelerationStructure();
        }

        const tlas = self.tlas orelse return results;

        // Query TLAS for candidate instances
        var candidate_instances = std.ArrayListUnmanaged(PrimId){};
        defer candidate_instances.deinit(self.allocator);
        try tlas.queryRayIntersection(ray, &candidate_instances);

        // Test each candidate instance
        for (candidate_instances.items) |instance_id| {
            const instance = &self.instances.items[instance_id];
            const blas = &self.blas_list.items[instance.model_id];

            // Transform ray to model local space
            const local_origin = instance.inv_transform.transformPoint(ray.origin);
            const local_dir = instance.inv_transform.transformDirection(ray.direction).normalized();
            const local_ray = Ray{
                .origin = local_origin,
                .direction = local_dir,
            };

            // Intersect with BLAS
            try blas.intersect(self.allocator, local_ray, direction, filter, &results, instance_id, instance.model_id);
        }

        // Sort by distance if returning all
        if (filter == .all and results.items.len > 1) {
            std.mem.sort(HitResult, results.items, {}, struct {
                fn lessThan(_: void, a: HitResult, b: HitResult) bool {
                    return @abs(a.t) < @abs(b.t);
                }
            }.lessThan);
        }

        // For closest filter with multiple results, keep only the closest
        if (filter == .closest and results.items.len > 1) {
            var closest_idx: usize = 0;
            var closest_t = @abs(results.items[0].t);
            for (results.items, 0..) |hit, idx| {
                if (@abs(hit.t) < closest_t) {
                    closest_t = @abs(hit.t);
                    closest_idx = idx;
                }
            }
            const closest = results.items[closest_idx];
            results.clearRetainingCapacity();
            try results.append(self.allocator, closest);
        }

        return results;
    }

    /// Get triangle normal at a hit point
    pub fn getHitNormal(self: *const Scene, hit: HitResult) Vec3 {
        const instance = &self.instances.items[hit.instance_id];
        const model = &self.models.items[hit.model_id];
        const mesh = &model.meshes.items[hit.mesh_idx];
        const tri = Triangle.fromMesh(mesh, hit.prim_id);

        // Get local normal and transform to world space
        const local_normal = tri.normalNormalized();
        const world_normal = instance.transform.transformDirection(local_normal).normalized();

        return world_normal;
    }

    /// Render the scene to the active camera's image buffer
    pub fn render(self: *Scene) !void {
        const camera = self.getActiveCamera() orelse return error.NoActiveCamera;

        // Allocate image buffer if not already allocated
        if (camera.image_buffer == null) {
            try camera.allocateImageBuffer(self.allocator);
        }

        // Ensure acceleration structure is built
        if (self.tlas == null) {
            try self.buildAccelerationStructure();
        }

        // Render each pixel
        for (0..camera.image_height) |y| {
            for (0..camera.image_width) |x| {
                const pixel_x: f32 = @floatFromInt(x);
                const pixel_y: f32 = @floatFromInt(y);

                // Generate ray for this pixel (center of pixel)
                const ray = camera.getRay(pixel_x + 0.5, pixel_y + 0.5);

                // Cast ray
                var hits = try self.castRay(ray, .forward, .closest);
                defer hits.deinit(self.allocator);

                // Shade the pixel
                var r: u8 = 0;
                var g: u8 = 0;
                var b: u8 = 0;

                if (hits.items.len > 0) {
                    const hit = hits.items[0];
                    const normal = self.getHitNormal(hit);

                    // Simple normal visualization (map [-1,1] to [0,255])
                    r = @intFromFloat(@max(0, @min(255, (normal.data[0] * 0.5 + 0.5) * 255)));
                    g = @intFromFloat(@max(0, @min(255, (normal.data[1] * 0.5 + 0.5) * 255)));
                    b = @intFromFloat(@max(0, @min(255, (normal.data[2] * 0.5 + 0.5) * 255)));
                }

                camera.setPixel(@intCast(x), @intCast(y), r, g, b);
            }
        }
    }
};

// Tests
test "scene basic operations" {
    const allocator = std.testing.allocator;

    var scene = Scene.init(allocator);
    defer scene.deinit();

    // Create a simple triangle model
    var model = Model.init(allocator);
    const vertices = [_][3]f32{
        .{ -1.0, -1.0, 0.0 },
        .{ 1.0, -1.0, 0.0 },
        .{ 0.0, 1.0, 0.0 },
    };
    const indices = [_]u32{ 0, 1, 2 };
    const mesh = try Mesh.initCopy(allocator, &vertices, &indices);
    try model.addMesh(mesh);

    // Add model to scene
    const model_id = try scene.addModel(model);
    try std.testing.expectEqual(@as(ModelId, 0), model_id);

    // Create an instance
    const instance_id = try scene.instantiate(model_id, Mat4.identity());
    try std.testing.expectEqual(@as(InstanceId, 0), instance_id);

    // Build acceleration structure
    try scene.buildAccelerationStructure();
    try std.testing.expect(scene.tlas != null);
}
