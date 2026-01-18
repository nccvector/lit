const std = @import("std");
const lmao = @import("lmao");
const chad = @import("chad");

const geometry = chad.geometry;
const Ray = geometry.Ray;

/// Generic camera supporting f16, f32, and f64 precision
pub fn Camera(comptime T: type) type {
    comptime {
        if (@typeInfo(T) != .float) @compileError("Camera requires a floating point type");
    }

    const Vec3 = lmao.Matrix(T, 3, 1);
    const Mat3 = lmao.Matrix(T, 3, 3);
    const Mat4 = lmao.Matrix(T, 4, 4);

    return struct {
        const Self = @This();

        /// Camera intrinsics matrix (3x3)
        /// [fx  0  cx]
        /// [ 0 fy  cy]
        /// [ 0  0   1]
        K: Mat3,

        /// Camera-to-world transformation (4x4)
        transform: Mat4,

        /// Image dimensions
        image_width: u32,
        image_height: u32,

        /// Image buffer (RGB, 3 bytes per pixel)
        image_buffer: ?[]u8 = null,
        allocator: ?std.mem.Allocator = null,

        /// Create a camera from vertical FOV and image dimensions
        pub fn fromFOV(vertical_fov_degrees: T, image_width: u32, image_height: u32) Self {
            const K = computeCameraMatrix(vertical_fov_degrees, @floatFromInt(image_width), @floatFromInt(image_height));
            return .{
                .K = K,
                .transform = Mat4.identity(),
                .image_width = image_width,
                .image_height = image_height,
            };
        }

        /// Create a camera from explicit intrinsics
        pub fn fromIntrinsics(fx: T, fy: T, cx: T, cy: T, image_width: u32, image_height: u32) Self {
            return .{
                .K = Mat3.fromArray(&.{
                    fx, 0,  cx,
                    0,  fy, cy,
                    0,  0,  1,
                }),
                .transform = Mat4.identity(),
                .image_width = image_width,
                .image_height = image_height,
            };
        }

        /// Compute the camera intrinsic matrix from vertical FOV
        pub fn computeCameraMatrix(vertical_fov_degrees: T, image_width: T, image_height: T) Mat3 {
            const pi: T = std.math.pi;
            const vertical_fov_radians = vertical_fov_degrees * pi / 180.0;

            // Compute image plane distance and horizontal FOV
            const half_fov = vertical_fov_radians / 2.0;
            const base_distance = (image_height / 2.0) / @tan(half_fov);
            const horizontal_fov_radians = 2.0 * std.math.atan((image_width / 2.0) / base_distance);

            // Compute focal lengths
            const fy = (image_height - 1.0) / (2.0 * @tan(half_fov));
            const fx = (image_width - 1.0) / (2.0 * @tan(horizontal_fov_radians / 2.0));

            // Compute principal point (image center)
            const cx = (image_width - 1.0) / 2.0;
            const cy = (image_height - 1.0) / 2.0;

            return Mat3.fromArray(&.{
                fx, 0,  cx,
                0,  fy, cy,
                0,  0,  1,
            });
        }

        /// Set the camera transform (camera-to-world)
        pub fn setTransform(self: *Self, transform: Mat4) void {
            self.transform = transform;
        }

        /// Set camera position using look-at
        pub fn lookAt(self: *Self, eye: Vec3, target: Vec3, up: Vec3) void {
            self.transform = Mat4.lookAt(eye, target, up).inverse() orelse Mat4.identity();
        }

        /// Allocate the image buffer
        pub fn allocateImageBuffer(self: *Self, allocator: std.mem.Allocator) !void {
            const size = @as(usize, self.image_width) * @as(usize, self.image_height) * 3;
            self.image_buffer = try allocator.alloc(u8, size);
            self.allocator = allocator;
            @memset(self.image_buffer.?, 0);
        }

        /// Free the image buffer
        pub fn freeImageBuffer(self: *Self) void {
            if (self.image_buffer) |buf| {
                if (self.allocator) |alloc| {
                    alloc.free(buf);
                }
            }
            self.image_buffer = null;
            self.allocator = null;
        }

        /// Set pixel color in the image buffer
        pub fn setPixel(self: *Self, x: u32, y: u32, r: u8, g: u8, b: u8) void {
            if (self.image_buffer) |buf| {
                const idx = (@as(usize, y) * @as(usize, self.image_width) + @as(usize, x)) * 3;
                buf[idx + 0] = r;
                buf[idx + 1] = g;
                buf[idx + 2] = b;
            }
        }

        /// Get the inverse of the intrinsic matrix
        pub fn getKInverse(self: *const Self) ?Mat3 {
            return self.K.inverse();
        }

        /// Generate a ray for the given pixel coordinates (in pixel space)
        /// Returns the ray in world space
        pub fn getRay(self: *const Self, pixel_x: T, pixel_y: T) Ray {
            // Get inverse intrinsics
            const K_inv = self.K.inverse() orelse Mat3.identity();

            // Create homogeneous pixel coordinate
            // Negate x to correct horizontal flip (mirror image correction)
            const pixel_homogeneous = Vec3.fromArray(&.{ pixel_x, pixel_y, 1.0 });

            // Transform to camera space direction
            // K_inv maps pixel coords to a point on the z=1 plane in camera space
            // Negate the entire direction to point into the scene (-Z forward convention)
            const dir_camera_inv = K_inv.dotSIMD(pixel_homogeneous).normalized();
            const dir_camera = Vec3.fromArray(&.{ dir_camera_inv.data[0], -dir_camera_inv.data[1], -dir_camera_inv.data[2] });

            // Transform direction and origin to world space
            const dir_world = self.transform.transformDirection(dir_camera).normalized();
            const origin_world = self.transform.transformPoint(Vec3.fromArray(&.{ 0, 0, 0 }));

            // Convert to f32 for Ray (chad uses f32)
            return .{
                .origin = toVec3f(origin_world),
                .direction = toVec3f(dir_world),
            };
        }

        /// Generate all rays for the image
        /// Returns a slice of rays, one per pixel (row-major order)
        pub fn getRays(self: *const Self, allocator: std.mem.Allocator) ![]Ray {
            const total_pixels = @as(usize, self.image_width) * @as(usize, self.image_height);
            const rays = try allocator.alloc(Ray, total_pixels);

            for (0..self.image_height) |y| {
                for (0..self.image_width) |x| {
                    const pixel_x: T = @floatFromInt(x);
                    const pixel_y: T = @floatFromInt(y);
                    const idx = y * self.image_width + x;
                    rays[idx] = self.getRay(pixel_x + 0.5, pixel_y + 0.5); // Center of pixel
                }
            }

            return rays;
        }

        /// Convert from generic Vec3 to f32 Vec3 (for chad compatibility)
        fn toVec3f(v: Vec3) lmao.Vec3f {
            if (T == f32) {
                return v;
            } else {
                return lmao.Vec3f.fromArray(&.{
                    @floatCast(v.data[0]),
                    @floatCast(v.data[1]),
                    @floatCast(v.data[2]),
                });
            }
        }
    };
}

// Default camera types
pub const Camera16 = Camera(f16);
pub const Camera32 = Camera(f32);
pub const Camera64 = Camera(f64);

// Tests
test "camera ray generation" {
    const cam = Camera32.fromFOV(60.0, 640, 480);

    // Test center pixel ray
    const center_ray = cam.getRay(320.0, 240.0);

    // Center ray should point roughly in -Z direction (forward in camera space)
    // with the default identity transform
    try std.testing.expect(@abs(center_ray.direction.data[0]) < 0.01);
    try std.testing.expect(@abs(center_ray.direction.data[1]) < 0.01);
    try std.testing.expect(center_ray.direction.data[2] < 0); // Negative Z (forward)
}

test "camera intrinsics" {
    const K = Camera32.computeCameraMatrix(90.0, 640.0, 480.0);

    // With 90 degree vertical FOV, fy should equal height/2 (approximately)
    const fy = K.data[4]; // K[1,1]
    const cy = K.data[5]; // K[1,2]

    try std.testing.expect(@abs(cy - 239.5) < 0.1); // Center should be at (height-1)/2
    try std.testing.expect(fy > 200 and fy < 300); // Reasonable focal length
}
