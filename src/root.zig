//! lit - A geometry-agnostic raytracing library
//!
//! This library provides an OptiX-inspired raytracing architecture:
//! - Geometry-agnostic scene with TLAS (top-level acceleration structure)
//! - User-provided BLAS (bottom-level acceleration structure) per object
//! - Comptime-known intersection functions for maximum performance (inlined)
//! - User Data Binding Table (UDBT) maps primitive IDs to custom data
//!
//! Example usage:
//! ```zig
//! const lit = @import("lit");
//!
//! // Define your primitive data
//! const SphereData = struct { center: lit.Vec3, radius: f32 };
//!
//! // Create UDBT with per-primitive data
//! var udbt = lit.UserDataBindingTable(SphereData).init(allocator);
//! try udbt.put(0, .{ .center = lit.Vec3.zero(), .radius = 1.0 });
//!
//! // Define intersection function (comptime known)
//! fn sphereIntersect(ray: lit.Ray, prim_id: lit.PrimId, udbt: *const lit.UserDataBindingTable(SphereData)) ?lit.IntersectionResult {
//!     const sphere = udbt.get(prim_id) orelse return null;
//!     // ... intersection math ...
//! }
//!
//! // Create Scene type with comptime intersection function
//! const MyScene = lit.Scene(SphereData, sphereIntersect);
//!
//! // Build BLAS (octree with primitive AABBs)
//! var blas = lit.Octree.init(allocator, bounds, .{});
//! try blas.insert(0, sphere_aabb);
//!
//! // Create scene and add instance
//! var scene = MyScene.init(allocator, &udbt);
//! _ = try scene.addInstance(&blas, lit.Mat4.identity());
//!
//! // Cast rays
//! var hits = std.ArrayListUnmanaged(lit.HitResult){};
//! try scene.castRay(ray, .forward, .closest, &hits);
//! ```

const std = @import("std");
pub const lmao = @import("lmao");
pub const chad = @import("chad");

// Re-export commonly used types from lmao
pub const Vec3f = lmao.Vec3f;
pub const Vec4f = lmao.Vec4f;
pub const Mat3f = lmao.Mat3f;
pub const Mat4f = lmao.Mat4f;
pub const Vec3d = lmao.Vec3d;
pub const Vec4d = lmao.Vec4d;
pub const Mat3d = lmao.Mat3d;
pub const Mat4d = lmao.Mat4d;
pub const Matrix = lmao.Matrix;

// Default precision aliases
pub const Vec3 = Vec3f;
pub const Vec4 = Vec4f;
pub const Mat3 = Mat3f;
pub const Mat4 = Mat4f;

// Re-export commonly used types from chad
pub const Ray = chad.geometry.Ray;
pub const RayHit = chad.geometry.RayHit;
pub const PrimitiveHit = chad.geometry.PrimitiveHit;
pub const Triangle = chad.geometry.Triangle;
pub const Aabb = chad.geometry.Aabb;
pub const Mesh = chad.geometry.Mesh;
pub const Model = chad.geometry.Model;

// Primitive shapes
pub const Sphere = chad.geometry.Sphere;
pub const Cube = chad.geometry.Cube;
pub const Cylinder = chad.geometry.Cylinder;
pub const Cone = chad.geometry.Cone;
pub const Pyramid = chad.geometry.Pyramid;

// Octree (for building BLAS)
pub const Octree = chad.octree.Octree;

// lit modules
pub const camera = @import("camera.zig");
pub const scene = @import("scene.zig");
pub const image = @import("image.zig");

// Re-export main types at top level
pub const Camera = camera.Camera;
pub const Camera16 = camera.Camera16;
pub const Camera32 = camera.Camera32;
pub const Camera64 = camera.Camera64;

// Scene types (comptime-parameterized)
pub const Scene = scene.Scene;
pub const CastDirection = scene.CastDirection;
pub const FilterMode = scene.FilterMode;
pub const HitResult = scene.HitResult;
pub const Instance = scene.Instance;
pub const InstanceId = scene.InstanceId;
pub const PrimId = scene.PrimId;

// UDBT and intersection result
pub const UserDataBindingTable = scene.UserDataBindingTable;
pub const IntersectionResult = scene.IntersectionResult;

// Image utilities
pub const ImageBuffer = image.ImageBuffer;
pub const writePPM = image.writePPM;
pub const writePPMAscii = image.writePPMAscii;

// OBJ loading convenience
pub const loadOBJ = chad.objloader.ObjLoader.loadModel;

test {
    std.testing.refAllDecls(@This());
}
