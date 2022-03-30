use ash::vk;

pub struct VirtualBuffer {
    backing_buffer: vk::Buffer,
    offset: u64,
    length: u64
}

//Allocator that can only free its memory all at once
pub struct VirtualBumpAllocator {
    backing_buffer: vk::Buffer,
    current_offset: u64,
    max_size: u64
}

impl VirtualBumpAllocator {
    pub fn clear(&mut self) {
        self.current_offset = 0;
    }

    pub fn new(backing_buffer: vk::Buffer, max_size: u64) -> Self {
        VirtualBumpAllocator {
            backing_buffer,
            current_offset: 0,
            max_size
        }
    }

    pub fn allocate(&mut self, size: u64) -> Result<VirtualBuffer, String> {
        if size + self.current_offset > self.max_size {
            return Err(format!("Backing buffer only has {} bytes remaining.", self.max_size - self.current_offset));
        }
        
        let b = VirtualBuffer {
            backing_buffer: self.backing_buffer,
            offset: self.current_offset,
            length: size
        };
        self.current_offset += size;
        Ok(b)
    }
}