//! lit - A high-performance raytracing library for computer vision research
//!
//! This library provides:
//! - Flexible camera models with intrinsic matrix computation
//! - Scene management with model loading and instancing
//! - Two-level acceleration structures (TLAS/BLAS) using octrees
//! - Ray casting with configurable direction and filtering
//! - PPM image output
//!
//! Example usage:
//! ```zig
//! const lit = @import("lit");
//!
//! // Create a scene
//! var scene = lit.Scene.init(allocator);
//! defer scene.deinit();
//!
//! // Load a model
//! const model_id = try scene.loadOBJ("model.obj", 1.0);
//!
//! // Create an instance
//! _ = try scene.instantiate(model_id, lit.Mat4.identity());
//!
//! // Add a camera
//! var camera = lit.Camera.fromFOV(60.0, 640, 480);
//! camera.lookAt(
//!     lit.Vec3.fromArray(&.{0, 0, 5}),
//!     lit.Vec3.fromArray(&.{0, 0, 0}),
//!     lit.Vec3.fromArray(&.{0, 1, 0}),
//! );
//! _ = try scene.addCamera(camera);
//!
//! // Render
//! try scene.render();
//!
//! // Save output
//! const cam = scene.getActiveCamera().?;
//! try lit.image.writePPM("output.ppm", cam.image_buffer.?, cam.image_width, cam.image_height);
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

// lit modules
pub const camera = @import("camera.zig");
pub const scene = @import("scene.zig");
pub const image = @import("image.zig");

// Re-export main types at top level
pub const Camera = camera.Camera;
pub const Camera16 = camera.Camera16;
pub const Camera32 = camera.Camera32;
pub const Camera64 = camera.Camera64;

pub const Scene = scene.Scene;
pub const CastDirection = scene.CastDirection;
pub const FilterMode = scene.FilterMode;
pub const HitResult = scene.HitResult;
pub const ModelId = scene.ModelId;
pub const InstanceId = scene.InstanceId;
pub const CameraId = scene.CameraId;

pub const ImageBuffer = image.ImageBuffer;
pub const writePPM = image.writePPM;
pub const writePPMAscii = image.writePPMAscii;

// OBJ loading convenience
pub const loadOBJ = chad.objloader.ObjLoader.loadModel;

test {
    std.testing.refAllDecls(@This());
}
