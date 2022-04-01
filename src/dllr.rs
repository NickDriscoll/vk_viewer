use std::ffi::c_void;

use ash::vk;
use std::ptr;
use crate::*;

unsafe fn get_memory_type_index(
    vk_instance: &ash::Instance,
    vk_physical_device: vk::PhysicalDevice,
    memory_requirements: vk::MemoryRequirements,
    flags: vk::MemoryPropertyFlags
) -> Option<u32> {
    let mut i = 0;
    let mut memory_type_index = None;
    let mut largest_heap = 0;
    let phys_device_mem_props = vk_instance.get_physical_device_memory_properties(vk_physical_device);
    for mem_type in phys_device_mem_props.memory_types {
        if memory_requirements.memory_type_bits & (1 << i) != 0 && mem_type.property_flags.contains(flags) {
            let heap_size = phys_device_mem_props.memory_heaps[mem_type.heap_index as usize].size;
            if heap_size > largest_heap {
                memory_type_index = Some(i);
                largest_heap = heap_size;
            }
        }
        i += 1;
    }

    memory_type_index
}

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

pub unsafe fn load_shader_stage(vk_device: &ash::Device, shader_stage_flags: vk::ShaderStageFlags, path: &str) -> vk::PipelineShaderStageCreateInfo {
    let mut file = File::open(path).unwrap();
    let spv = ash::util::read_spv(&mut file).unwrap();

    let module_create_info = vk::ShaderModuleCreateInfo {
        code_size: spv.len() * size_of::<u32>(),
        p_code: spv.as_ptr(),
        ..Default::default()
    };
    let module = vk_device.create_shader_module(&module_create_info, VK_MEMORY_ALLOCATOR).unwrap();

    vk::PipelineShaderStageCreateInfo {
        stage: shader_stage_flags,
        p_name: "main\0".as_ptr() as *const i8,
        module,
        ..Default::default()
    }
}

pub struct VirtualBuffer {
    backing_buffer: vk::Buffer,
    buffer_ptr: *mut c_void,
    offset: u64,
    length: u64
}

impl VirtualBuffer {
    //Read-only access to fields
    pub fn backing_buffer(&self) -> vk::Buffer { self.backing_buffer }
    pub fn offset(&self) -> u64 { self.offset }
    pub fn length(&self) -> u64 { self.length }

    pub fn upload_buffer<T>(&self, in_buffer: &[T]) {
        unsafe {
            let dst_ptr = self.buffer_ptr.offset(self.offset.try_into().unwrap());
            ptr::copy_nonoverlapping(in_buffer.as_ptr(), dst_ptr as *mut T, in_buffer.len());
        }
    }
}

//Allocator that can only free its memory all at once
pub struct VirtualBumpAllocator {
    backing_buffer: vk::Buffer,
    buffer_ptr: *mut c_void,
    current_offset: u64,
    max_size: u64,
}

impl VirtualBumpAllocator {
    //Read-only access to fields
    pub fn backing_buffer(&self) -> vk::Buffer { self.backing_buffer }
    pub fn current_offset(&self) -> u64 { self.current_offset }
    pub fn max_size(&self) -> u64 { self.max_size }

    pub fn new(backing_buffer: vk::Buffer, ptr: *mut c_void, max_size: u64) -> Self {
        VirtualBumpAllocator {
            backing_buffer,
            current_offset: 0,
            max_size,
            buffer_ptr: ptr
        }
    }

    pub fn clear(&mut self) {
        self.current_offset = 0;
    }


    pub fn allocate(&mut self, size: u64) -> Result<VirtualBuffer, String> {
        if size + self.current_offset > self.max_size {
            return Err(format!("Backing buffer only has {} bytes remaining.", self.max_size - self.current_offset));
        }
        
        let b = VirtualBuffer {
            backing_buffer: self.backing_buffer,
            buffer_ptr: self.buffer_ptr,
            offset: self.current_offset,
            length: size
        };
        self.current_offset += size;
        Ok(b)
    }
}

pub struct VirtualGeometry {
    vertex_buffer: VirtualBuffer,
    index_buffer: VirtualBuffer
}