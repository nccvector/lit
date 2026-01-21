const std = @import("std");

// Use std.mem.doNotOptimizeAway to prevent DCE
pub const doNotOptimizeAway = std.mem.doNotOptimizeAway;

pub fn print(comptime fmt: []const u8, args: anytype) void {
    const stdout = std.fs.File.stdout();
    var buf: [4096]u8 = undefined;
    const slice = std.fmt.bufPrint(&buf, fmt, args) catch return;
    _ = stdout.write(slice) catch {};
}

pub fn formatTimeBuf(ns: f64, buf: []u8) []const u8 {
    if (ns < 1_000) {
        return std.fmt.bufPrint(buf, "{d:.2}ns", .{ns}) catch "?";
    } else if (ns < 1_000_000) {
        return std.fmt.bufPrint(buf, "{d:.2}us", .{ns / 1_000.0}) catch "?";
    } else if (ns < 1_000_000_000) {
        return std.fmt.bufPrint(buf, "{d:.2}ms", .{ns / 1_000_000.0}) catch "?";
    } else {
        return std.fmt.bufPrint(buf, "{d:.2}s", .{ns / 1_000_000_000.0}) catch "?";
    }
}

/// Format a throughput value (items per second)
pub fn formatThroughputBuf(ops_per_sec: f64, buf: []u8) []const u8 {
    if (ops_per_sec < 1_000) {
        return std.fmt.bufPrint(buf, "{d:.1}/s", .{ops_per_sec}) catch "?";
    } else if (ops_per_sec < 1_000_000) {
        return std.fmt.bufPrint(buf, "{d:.2}K/s", .{ops_per_sec / 1_000.0}) catch "?";
    } else if (ops_per_sec < 1_000_000_000) {
        return std.fmt.bufPrint(buf, "{d:.2}M/s", .{ops_per_sec / 1_000_000.0}) catch "?";
    } else {
        return std.fmt.bufPrint(buf, "{d:.2}G/s", .{ops_per_sec / 1_000_000_000.0}) catch "?";
    }
}

/// Generate random values for a given floating-point type
pub fn randomFloat(comptime T: type, rng: std.Random) T {
    return switch (@typeInfo(T)) {
        .float => |info| blk: {
            if (info.bits == 16) {
                break :blk @as(T, @floatCast(rng.float(f32) * 2.0 - 1.0));
            } else {
                break :blk rng.float(T) * 2.0 - 1.0;
            }
        },
        else => @compileError("randomFloat requires a floating-point type"),
    };
}

/// Generate a random float in range [min, max)
pub fn randomFloatRange(comptime T: type, rng: std.Random, min_val: T, max_val: T) T {
    const t = rng.float(f32);
    return min_val + @as(T, @floatCast(t)) * (max_val - min_val);
}

/// Initialize RNG from timestamp
pub fn initRng() std.Random.DefaultPrng {
    const ts: i128 = std.time.nanoTimestamp();
    return std.Random.DefaultPrng.init(@truncate(@as(u128, @bitCast(ts))));
}

/// Parse common benchmark arguments (-N=iterations)
pub fn parseArgs(args: []const [:0]const u8) struct { iterations: usize } {
    var iterations: usize = 10_000;

    for (args) |arg| {
        if (std.mem.startsWith(u8, arg, "-N=")) {
            iterations = std.fmt.parseInt(usize, arg[3..], 10) catch 10_000;
        }
    }

    return .{ .iterations = iterations };
}

/// Print a table header
pub fn printHeader(title: []const u8, iterations: usize) void {
    print("\n {s} ({d} iterations, ReleaseFast)\n", .{ title, iterations });
    print("{s}\n", .{"═" ** 70});
    print(" {s: <35} │ {s: >12} │ {s: >15}\n", .{ "Operation", "Time/op", "Throughput" });
    print("{s}\n", .{"─" ** 70});
}

/// Print a table row with time and throughput
pub fn printRow(name: []const u8, ns_per_op: f64) void {
    var time_buf: [32]u8 = undefined;
    var throughput_buf: [32]u8 = undefined;

    const time_str = formatTimeBuf(ns_per_op, &time_buf);
    const ops_per_sec = 1_000_000_000.0 / ns_per_op;
    const throughput_str = formatThroughputBuf(ops_per_sec, &throughput_buf);

    print(" {s: <35} │ {s: >12} │ {s: >15}\n", .{ name, time_str, throughput_str });
}

/// Print a table row with custom value (for things like pixel throughput)
pub fn printRowCustom(name: []const u8, ns_per_op: f64, custom_throughput: []const u8) void {
    var time_buf: [32]u8 = undefined;
    const time_str = formatTimeBuf(ns_per_op, &time_buf);
    print(" {s: <35} │ {s: >12} │ {s: >15}\n", .{ name, time_str, custom_throughput });
}

/// Print a table footer
pub fn printFooter() void {
    print("{s}\n\n", .{"═" ** 70});
}

/// Measure the time to execute a function N times, returning nanoseconds per operation
pub fn benchmark(comptime func: anytype, args: anytype, iterations: usize) f64 {
    const start = std.time.nanoTimestamp();
    for (0..iterations) |_| {
        const result = @call(.auto, func, args);
        doNotOptimizeAway(result);
    }
    const end = std.time.nanoTimestamp();
    return @as(f64, @floatFromInt(end - start)) / @as(f64, @floatFromInt(iterations));
}
