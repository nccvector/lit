const std = @import("std");

/// Write an RGB image to a PPM file (P6 binary format)
/// Data is expected in row-major RGB format, 3 bytes per pixel
pub fn writePPM(path: []const u8, data: []const u8, width: u32, height: u32) !void {
    const file = try std.fs.cwd().createFile(path, .{});
    defer file.close();

    // Write PPM header
    var header_buf: [64]u8 = undefined;
    const header = std.fmt.bufPrint(&header_buf, "P6\n{d} {d}\n255\n", .{ width, height }) catch unreachable;
    _ = try file.write(header);

    // Write pixel data
    _ = try file.write(data);
}

/// Write an RGB image to a PPM file (P3 ASCII format)
/// Useful for debugging as the file is human-readable
pub fn writePPMAscii(path: []const u8, data: []const u8, width: u32, height: u32) !void {
    const file = try std.fs.cwd().createFile(path, .{});
    defer file.close();

    // Write PPM header
    var header_buf: [64]u8 = undefined;
    const header = std.fmt.bufPrint(&header_buf, "P3\n{d} {d}\n255\n", .{ width, height }) catch unreachable;
    _ = try file.write(header);

    // Write pixel data as ASCII
    var line_buf: [4096]u8 = undefined;
    var idx: usize = 0;
    for (0..height) |_| {
        var line_len: usize = 0;
        for (0..width) |x| {
            const r = data[idx];
            const g = data[idx + 1];
            const b = data[idx + 2];
            idx += 3;

            if (x > 0) {
                line_buf[line_len] = ' ';
                line_len += 1;
            }
            const pixel_str = std.fmt.bufPrint(line_buf[line_len..], "{d} {d} {d}", .{ r, g, b }) catch break;
            line_len += pixel_str.len;
        }
        line_buf[line_len] = '\n';
        line_len += 1;
        _ = try file.write(line_buf[0..line_len]);
    }
}

/// Image buffer for accumulating samples and outputting
pub const ImageBuffer = struct {
    width: u32,
    height: u32,
    data: []u8,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, width: u32, height: u32) !ImageBuffer {
        const size = @as(usize, width) * @as(usize, height) * 3;
        const data = try allocator.alloc(u8, size);
        @memset(data, 0);

        return .{
            .width = width,
            .height = height,
            .data = data,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *ImageBuffer) void {
        self.allocator.free(self.data);
    }

    /// Set a pixel's RGB value
    pub fn setPixel(self: *ImageBuffer, x: u32, y: u32, r: u8, g: u8, b: u8) void {
        const idx_val = (@as(usize, y) * @as(usize, self.width) + @as(usize, x)) * 3;
        self.data[idx_val + 0] = r;
        self.data[idx_val + 1] = g;
        self.data[idx_val + 2] = b;
    }

    /// Set a pixel's RGB value from floats in [0, 1] range
    pub fn setPixelF(self: *ImageBuffer, x: u32, y: u32, r: f32, g: f32, b: f32) void {
        const r_clamped = @max(0.0, @min(1.0, r));
        const g_clamped = @max(0.0, @min(1.0, g));
        const b_clamped = @max(0.0, @min(1.0, b));

        self.setPixel(
            x,
            y,
            @intFromFloat(r_clamped * 255.0),
            @intFromFloat(g_clamped * 255.0),
            @intFromFloat(b_clamped * 255.0),
        );
    }

    /// Get a pixel's RGB value
    pub fn getPixel(self: *const ImageBuffer, x: u32, y: u32) struct { r: u8, g: u8, b: u8 } {
        const idx_val = (@as(usize, y) * @as(usize, self.width) + @as(usize, x)) * 3;
        return .{
            .r = self.data[idx_val + 0],
            .g = self.data[idx_val + 1],
            .b = self.data[idx_val + 2],
        };
    }

    /// Fill the image with a solid color
    pub fn fill(self: *ImageBuffer, r: u8, g: u8, b: u8) void {
        var idx_val: usize = 0;
        while (idx_val < self.data.len) : (idx_val += 3) {
            self.data[idx_val + 0] = r;
            self.data[idx_val + 1] = g;
            self.data[idx_val + 2] = b;
        }
    }

    /// Save to PPM file (binary format)
    pub fn savePPM(self: *const ImageBuffer, path: []const u8) !void {
        try writePPM(path, self.data, self.width, self.height);
    }

    /// Save to PPM file (ASCII format)
    pub fn savePPMAscii(self: *const ImageBuffer, path: []const u8) !void {
        try writePPMAscii(path, self.data, self.width, self.height);
    }
};

// Tests
test "PPM write" {
    const allocator = std.testing.allocator;

    var img = try ImageBuffer.init(allocator, 2, 2);
    defer img.deinit();

    // Set some pixels
    img.setPixel(0, 0, 255, 0, 0); // Red
    img.setPixel(1, 0, 0, 255, 0); // Green
    img.setPixel(0, 1, 0, 0, 255); // Blue
    img.setPixel(1, 1, 255, 255, 255); // White

    // Test pixel values
    const p00 = img.getPixel(0, 0);
    try std.testing.expectEqual(@as(u8, 255), p00.r);
    try std.testing.expectEqual(@as(u8, 0), p00.g);
    try std.testing.expectEqual(@as(u8, 0), p00.b);
}

test "ImageBuffer float pixels" {
    const allocator = std.testing.allocator;

    var img = try ImageBuffer.init(allocator, 1, 1);
    defer img.deinit();

    img.setPixelF(0, 0, 0.5, 0.25, 1.0);

    const p = img.getPixel(0, 0);
    try std.testing.expectEqual(@as(u8, 127), p.r);
    try std.testing.expectEqual(@as(u8, 63), p.g);
    try std.testing.expectEqual(@as(u8, 255), p.b);
}
