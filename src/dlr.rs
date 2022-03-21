use ash::vk;

pub struct VirtualBuffer {
    backing_buffer: vk::Buffer,
    offset: u64
}