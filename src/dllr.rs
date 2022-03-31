use ash::vk;
use crate::*;

pub unsafe fn allocate_buffer_memory(vk_instance: &ash::Instance, vk_physical_device: vk::PhysicalDevice, vk_device: &ash::Device, buffer: vk::Buffer) -> vk::DeviceMemory {
    let mem_reqs = vk_device.get_buffer_memory_requirements(buffer);
    let memory_type_index = get_memory_type_index(
        &vk_instance,
        vk_physical_device,
        mem_reqs,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT
    );
    if let None = memory_type_index {
        crash_with_error_dialog("Staging buffer memory allocation failed.");
    }
    let memory_type_index = memory_type_index.unwrap();

    let alloc_info = vk::MemoryAllocateInfo {
        allocation_size: mem_reqs.size,
        memory_type_index,
        ..Default::default()
    };
    vk_device.allocate_memory(&alloc_info, VK_MEMORY_ALLOCATOR).unwrap()    
}

pub unsafe fn allocate_image_memory(vk_instance: &ash::Instance, vk_physical_device: vk::PhysicalDevice, vk_device: &ash::Device, image: vk::Image) -> vk::DeviceMemory {
    let mem_reqs = vk_device.get_image_memory_requirements(image);

    //Search for the largest DEVICE_LOCAL heap the device advertises
    let memory_type_index = get_memory_type_index(&vk_instance, vk_physical_device, mem_reqs, vk::MemoryPropertyFlags::DEVICE_LOCAL);
    if let None = memory_type_index {
        crash_with_error_dialog("Depth buffer memory allocation failed.");
    }
    let memory_type_index = memory_type_index.unwrap();

    let allocate_info = vk::MemoryAllocateInfo {
        allocation_size: mem_reqs.size,
        memory_type_index,
        ..Default::default()
    };
    vk_device.allocate_memory(&allocate_info, VK_MEMORY_ALLOCATOR).unwrap()
}

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