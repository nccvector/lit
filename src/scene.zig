const std = @import("std");
const lmao = @import("lmao");
const chad = @import("chad");

const geometry = chad.geometry;
const octree_mod = chad.octree;

const Vec3 = lmao.Vec3f;
const Mat4 = lmao.Mat4f;
const Ray = geometry.Ray;
const Aabb = geometry.Aabb;
const Octree = octree_mod.Octree;

/// Primitive ID type - used as key for UDBT lookups
pub const PrimId = geometry.PrimId;

/// Unique identifiers
pub const InstanceId = u32;
pub const CameraId = u32;

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

/// Result from an intersection function
/// Returned by user-defined intersection functions
pub const IntersectionResult = struct {
    t: f32, // Distance along ray
    u: f32, // First barycentric/parametric coordinate
    v: f32, // Second barycentric/parametric coordinate
    hit_kind: u8 = 0, // User-defined hit type (e.g., front face = 0, back face = 1)
};

/// Hit result returned from ray casting
/// Contains only IDs - user looks up actual data via UDBT using prim_id
pub const HitResult = struct {
    instance_id: InstanceId,
    prim_id: PrimId, // Key for UDBT lookup
    t: f32, // Distance along ray (in world space)
    u: f32, // First barycentric/parametric coordinate
    v: f32, // Second barycentric/parametric coordinate
    hit_kind: u8, // User-defined hit type from intersection function

    /// Compute the hit point in world space
    pub fn hitPoint(self: HitResult, ray: Ray) Vec3 {
        return ray.at(self.t);
    }

    /// Get the third barycentric coordinate (for triangles: w = 1 - u - v)
    pub fn w(self: HitResult) f32 {
        return 1.0 - self.u - self.v;
    }
};

/// Instance of geometry in the scene
/// References an external BLAS (user-owned) with a transform
pub const Instance = struct {
    blas: *const Octree, // User-provided BLAS (not owned by scene)
    transform: Mat4,
    inv_transform: Mat4,
    bounds: Aabb, // World-space bounds (for TLAS)
};

/// User Data Binding Table - maps PrimId to user-defined data
/// Generic over the data type stored per primitive
pub fn UserDataBindingTable(comptime T: type) type {
    return struct {
        const Self = @This();

        data: std.AutoHashMap(PrimId, T),

        pub fn init(allocator: std.mem.Allocator) Self {
            return .{
                .data = std.AutoHashMap(PrimId, T).init(allocator),
            };
        }

        pub fn deinit(self: *Self) void {
            self.data.deinit();
        }

        /// Add or update data for a primitive
        pub fn put(self: *Self, prim_id: PrimId, value: T) !void {
            try self.data.put(prim_id, value);
        }

        /// Get data for a primitive (returns null if not found)
        pub fn get(self: *const Self, prim_id: PrimId) ?T {
            return self.data.get(prim_id);
        }

        /// Get pointer to data for a primitive (returns null if not found)
        pub fn getPtr(self: *Self, prim_id: PrimId) ?*T {
            return self.data.getPtr(prim_id);
        }

        /// Get const pointer to data for a primitive (returns null if not found)
        pub fn getConstPtr(self: *const Self, prim_id: PrimId) ?*const T {
            return self.data.getPtr(prim_id);
        }

        /// Check if a primitive has data
        pub fn contains(self: *const Self, prim_id: PrimId) bool {
            return self.data.contains(prim_id);
        }

        /// Remove data for a primitive
        pub fn remove(self: *Self, prim_id: PrimId) bool {
            return self.data.remove(prim_id);
        }
    };
}

/// Transform an AABB by a matrix (returns a conservative AABB)
fn transformAabb(bounds: Aabb, transform: Mat4) Aabb {
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

/// Comptime-parameterized Scene
/// The intersection function and UDBT type are known at compile time,
/// allowing full inlining of intersection logic.
///
/// Parameters:
///   - UserData: The type stored in the UDBT for each primitive
///   - intersectFn: Compile-time known intersection function
///
/// Example:
/// ```zig
/// const MyScene = Scene(GeometryData, geometryIntersect);
/// var scene = MyScene.init(allocator, &udbt);
/// ```
pub fn Scene(
    comptime UserData: type,
    comptime intersectFn: fn (ray: Ray, prim_id: PrimId, udbt: *const UserDataBindingTable(UserData)) ?IntersectionResult,
) type {
    return struct {
        const Self = @This();
        const UDBT = UserDataBindingTable(UserData);

        allocator: std.mem.Allocator,

        /// Instances referencing external BLAS structures
        instances: std.ArrayListUnmanaged(Instance) = .empty,

        /// Top-Level Acceleration Structure (over instances)
        tlas: ?Octree = null,

        /// User Data Binding Table - provided by user
        udbt: *const UDBT,

        pub fn init(allocator: std.mem.Allocator, udbt: *const UDBT) Self {
            return .{
                .allocator = allocator,
                .udbt = udbt,
            };
        }

        pub fn deinit(self: *Self) void {
            if (self.tlas) |*tlas| {
                tlas.deinit();
            }
            self.instances.deinit(self.allocator);
        }

        /// Add an instance to the scene
        /// The BLAS must outlive the scene (user-owned)
        pub fn addInstance(self: *Self, blas: *const Octree, transform: Mat4) !InstanceId {
            const world_bounds = transformAabb(blas.root.bounds, transform);
            const inv_transform = transform.inverse() orelse Mat4.identity();

            const instance = Instance{
                .blas = blas,
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
        pub fn setInstanceTransform(self: *Self, instance_id: InstanceId, transform: Mat4) !void {
            if (instance_id >= self.instances.items.len) {
                return error.InvalidInstanceId;
            }

            var instance = &self.instances.items[instance_id];
            instance.transform = transform;
            instance.inv_transform = transform.inverse() orelse Mat4.identity();
            instance.bounds = transformAabb(instance.blas.root.bounds, transform);

            // Invalidate TLAS
            if (self.tlas) |*tlas| {
                tlas.deinit();
                self.tlas = null;
            }
        }

        /// Get instance by ID
        pub fn getInstance(self: *const Self, instance_id: InstanceId) ?*const Instance {
            if (instance_id >= self.instances.items.len) {
                return null;
            }
            return &self.instances.items[instance_id];
        }

        /// Build or rebuild the Top-Level Acceleration Structure
        pub fn buildTLAS(self: *Self) !void {
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

        /// Cast a ray and collect hit results
        /// Results are appended to the provided list (caller owns the list)
        /// The intersection function is inlined at compile time for maximum performance.
        pub fn castRay(
            self: *Self,
            ray: Ray,
            direction: CastDirection,
            filter: FilterMode,
            results: *std.ArrayListUnmanaged(HitResult),
        ) !void {
            // Ensure TLAS is built
            if (self.tlas == null) {
                try self.buildTLAS();
            }

            const tlas = self.tlas orelse return;

            // Query TLAS for candidate instances
            var candidate_instances = std.ArrayListUnmanaged(PrimId){};
            defer candidate_instances.deinit(self.allocator);
            try tlas.queryRayIntersection(ray, &candidate_instances);

            // Test each candidate instance
            for (candidate_instances.items) |instance_id| {
                const instance = &self.instances.items[instance_id];

                // Transform ray to BLAS local space
                const local_origin = instance.inv_transform.transformPoint(ray.origin);
                const local_dir = instance.inv_transform.transformDirection(ray.direction).normalized();
                const local_ray = Ray{
                    .origin = local_origin,
                    .direction = local_dir,
                };

                // Query BLAS for candidate primitives
                var candidate_prims = std.ArrayListUnmanaged(PrimId){};
                defer candidate_prims.deinit(self.allocator);
                try instance.blas.queryRayIntersection(local_ray, &candidate_prims);

                // Test each candidate primitive using comptime-known intersection function
                for (candidate_prims.items) |prim_id| {
                    // Call intersection function (inlined at compile time)
                    const intersection_result: ?IntersectionResult = switch (direction) {
                        .forward => intersectFn(local_ray, prim_id, self.udbt),
                        .backward => blk: {
                            const flipped_ray = Ray{
                                .origin = local_ray.origin,
                                .direction = local_ray.direction.negate(),
                            };
                            if (intersectFn(flipped_ray, prim_id, self.udbt)) |res| {
                                break :blk IntersectionResult{
                                    .t = -res.t,
                                    .u = res.u,
                                    .v = res.v,
                                    .hit_kind = res.hit_kind,
                                };
                            }
                            break :blk null;
                        },
                        .both => blk: {
                            // Try forward first
                            if (intersectFn(local_ray, prim_id, self.udbt)) |res| {
                                break :blk res;
                            }
                            // Try backward
                            const flipped_ray = Ray{
                                .origin = local_ray.origin,
                                .direction = local_ray.direction.negate(),
                            };
                            if (intersectFn(flipped_ray, prim_id, self.udbt)) |res| {
                                break :blk IntersectionResult{
                                    .t = -res.t,
                                    .u = res.u,
                                    .v = res.v,
                                    .hit_kind = res.hit_kind,
                                };
                            }
                            break :blk null;
                        },
                    };

                    if (intersection_result) |isect| {
                        const hit = HitResult{
                            .instance_id = instance_id,
                            .prim_id = prim_id,
                            .t = isect.t,
                            .u = isect.u,
                            .v = isect.v,
                            .hit_kind = isect.hit_kind,
                        };

                        if (filter == .closest) {
                            // Keep only closest
                            if (results.items.len == 0 or @abs(hit.t) < @abs(results.items[0].t)) {
                                results.clearRetainingCapacity();
                                try results.append(self.allocator, hit);
                            }
                        } else {
                            try results.append(self.allocator, hit);
                        }
                    }
                }
            }

            // Sort by distance if returning all
            if (filter == .all and results.items.len > 1) {
                std.mem.sort(HitResult, results.items, {}, struct {
                    fn lessThan(_: void, a: HitResult, b: HitResult) bool {
                        return @abs(a.t) < @abs(b.t);
                    }
                }.lessThan);
            }

            // For closest filter with multiple results from different instances, keep only the closest
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
        }
    };
}

// ============================================================================
// Tests
// ============================================================================

test "UserDataBindingTable basic operations" {
    const allocator = std.testing.allocator;

    const SphereData = struct {
        center: [3]f32,
        radius: f32,
    };

    var udbt = UserDataBindingTable(SphereData).init(allocator);
    defer udbt.deinit();

    // Add some data
    try udbt.put(0, .{ .center = .{ 0, 0, 0 }, .radius = 1.0 });
    try udbt.put(1, .{ .center = .{ 2, 0, 0 }, .radius = 0.5 });

    // Retrieve data
    const sphere0 = udbt.get(0).?;
    try std.testing.expectEqual(@as(f32, 1.0), sphere0.radius);

    const sphere1 = udbt.get(1).?;
    try std.testing.expectEqual(@as(f32, 0.5), sphere1.radius);

    // Non-existent
    try std.testing.expect(udbt.get(99) == null);
}

test "Scene with comptime intersection" {
    const allocator = std.testing.allocator;

    // Simple sphere data
    const SphereData = struct {
        center: Vec3,
        radius: f32,
    };

    // User data binding table
    var udbt_table = UserDataBindingTable(SphereData).init(allocator);
    defer udbt_table.deinit();

    try udbt_table.put(0, .{ .center = Vec3.fromArray(&.{ 0, 0, 0 }), .radius = 1.0 });

    // Sphere intersection function (comptime known)
    const sphereIntersect = struct {
        fn func(ray: Ray, prim_id: PrimId, udbt: *const UserDataBindingTable(SphereData)) ?IntersectionResult {
            const sphere = udbt.get(prim_id) orelse return null;

            // Ray-sphere intersection
            const oc = ray.origin.sub(sphere.center);
            const dir = ray.direction;
            const a = dir.dotProduct(dir);
            const b = 2.0 * oc.dotProduct(dir);
            const c = oc.dotProduct(oc) - sphere.radius * sphere.radius;
            const discriminant = b * b - 4.0 * a * c;

            if (discriminant < 0) return null;

            const t = (-b - @sqrt(discriminant)) / (2.0 * a);
            if (t < 0) return null;

            return .{ .t = t, .u = 0, .v = 0, .hit_kind = 0 };
        }
    }.func;

    // Create scene with comptime-known intersection function
    const MyScene = Scene(SphereData, sphereIntersect);

    // Build BLAS (just an octree with the sphere's AABB)
    var blas = Octree.init(allocator, .{
        .bmin = Vec3.fromArray(&.{ -2, -2, -2 }),
        .bmax = Vec3.fromArray(&.{ 2, 2, 2 }),
    }, .{});
    defer blas.deinit();

    try blas.insert(0, .{
        .bmin = Vec3.fromArray(&.{ -1, -1, -1 }),
        .bmax = Vec3.fromArray(&.{ 1, 1, 1 }),
    });

    // Create scene
    var scene = MyScene.init(allocator, &udbt_table);
    defer scene.deinit();

    // Add instance
    _ = try scene.addInstance(&blas, Mat4.identity());
    try scene.buildTLAS();

    // Cast ray
    var results = std.ArrayListUnmanaged(HitResult){};
    defer results.deinit(allocator);

    const ray = Ray{
        .origin = Vec3.fromArray(&.{ 0, 0, -5 }),
        .direction = Vec3.fromArray(&.{ 0, 0, 1 }),
    };

    try scene.castRay(ray, .forward, .closest, &results);

    try std.testing.expectEqual(@as(usize, 1), results.items.len);
    try std.testing.expectEqual(@as(PrimId, 0), results.items[0].prim_id);

    // t should be approximately 4.0 (ray starts at z=-5, sphere surface at z=-1)
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), results.items[0].t, 0.001);
}
