const std = @import("std");
const assert = std.debug.assert;
const Image = @This();

width: u32,
height: u32,
data: []u8,

pub fn save(image: Image, path: []const u8) !void {
    const file = try std.fs.cwd().createFile(path, .{});
    defer file.close();
    var bw = std.io.bufferedWriter(file.writer());
    try image.write(bw.writer());
    try bw.flush();
}

pub fn write(image: Image, writer: anytype) !void {
    const file_size = @sizeOf(FileHeader) + @sizeOf(BMPInfoHeader) + image.data.len;

    const file_header: FileHeader = .{ .file_size = @intCast(file_size) };

    inline for (std.meta.fields(FileHeader)) |field| {
        try writer.writeInt(field.type, @field(file_header, field.name), .little);
    }

    const flipped_height: i32 = @intCast(image.height);
    const bitmap_info: BMPInfoHeader = .{
        .width = @intCast(image.width),
        .height = -flipped_height,
        .bit_count = 32,
        .image_size = @intCast(image.data.len),
    };

    inline for (std.meta.fields(BMPInfoHeader)) |field| {
        try writer.writeInt(field.type, @field(bitmap_info, field.name), .little);
    }

    try writer.writeAll(image.data);
}

const file_header_size = 14;
const FileHeader = struct {
    file_type: u16 = @bitCast([2]u8{ 'B', 'M' }),
    file_size: u32,
    _reserved: u32 = 0,
    image_data_offset: u32 = file_header_size,
};

const BMPInfoHeader = struct {
    header_size: u32 = @sizeOf(BMPInfoHeader),
    width: i32,
    height: i32,
    planes: u16 = 0,
    bit_count: u16,
    compression: u32 = 0,
    image_size: u32,

    x_pels_per_meter: i32 = 0,
    y_pels_per_meter: i32 = 0,

    clr_used: u32 = 0,
    clr_important: u32 = 0,

    comptime {
        assert(@sizeOf(BMPInfoHeader) == 40);
    }
};
