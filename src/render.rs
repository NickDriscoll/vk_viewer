use core::slice::Iter;
use ash::vk::DescriptorImageInfo;

use crate::vkutil::VirtualGeometry;
use crate::*;

#[derive(Debug)]
pub struct Material {
    pub color_idx: u32,
    pub normal_idx: u32
}

pub struct DrawCall {
    pub geometry_idx: usize,
    pub pipeline: vk::Pipeline,
    pub instance_count: u32,
    pub first_instance: u32
}

//This is a struct that contains mesh and material data
//In other words, the data required to draw a specific kind of thing
pub struct DrawData {
    pub geometry: VirtualGeometry,
    pub position_offset: u32,
    pub tangent_offset: u32,
    pub bitangent_offset: u32,
    pub uv_offset: u32,
    pub material_idx: u32
}

pub struct InstanceData {
    pub world_from_model: glm::TMat4<f32>,
    pub normal_matrix: glm::TMat4<f32>
}

impl InstanceData {
    pub fn new(world_from_model: glm::TMat4<f32>) -> Self {
        let normal_matrix = glm::mat4_to_mat3(&world_from_model);
        let normal_matrix = glm::transpose(&glm::mat3_to_mat4(&glm::affine_inverse(normal_matrix)));

        InstanceData {
            world_from_model,
            normal_matrix
        }
    }
}

pub struct Renderer {
    models: OptionVec<DrawData>,
    drawlist: Vec<DrawCall>,
    instance_data: Vec<InstanceData>,

    pub position_buffer: GPUBuffer,
    position_offset: u64,
    pub tangent_buffer: GPUBuffer,
    tangent_offset: u64,
    pub bitangent_buffer: GPUBuffer,
    bitangent_offset: u64,
    pub uv_buffer: GPUBuffer,
    uv_offset: u64,
    pub uniform_buffer: GPUBuffer,
    pub instance_buffer: GPUBuffer,
    pub material_buffer: GPUBuffer,
    pub global_textures: FreeList<DescriptorImageInfo>,
    pub descriptor_set_layout: vk::DescriptorSetLayout,
    pub descriptor_sets: Vec<vk::DescriptorSet>,
}

impl Renderer {
    //pub const FRAMES_IN_FLIGHT: usize = 2;

    pub fn get_instance_data(&self) -> &Vec<InstanceData> {
        &self.instance_data
    }

    pub fn init(vk: &mut VulkanAPI) -> Self {
        //Maintain free list for texture allocation
        let global_textures = FreeList::with_capacity(1024);

        //Allocate buffer for frame-constant uniforms
        let uniform_buffer_size = size_of::<FrameUniforms>() as vk::DeviceSize;
        let uniform_buffer_alignment = vk.physical_device_properties.limits.min_uniform_buffer_offset_alignment;
        let uniform_buffer = GPUBuffer::allocate(
            vk,
            uniform_buffer_size,
            uniform_buffer_alignment,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            MemoryLocation::CpuToGpu
        );
        
        //Allocate buffer for instance data
        let storage_buffer_alignment = vk.physical_device_properties.limits.min_storage_buffer_offset_alignment;
        let global_transform_slots = 1024 * 1024;
        let buffer_size = (size_of::<render::InstanceData>() * global_transform_slots) as vk::DeviceSize;
        let instance_buffer = GPUBuffer::allocate(
            vk,
            buffer_size,
            storage_buffer_alignment,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            MemoryLocation::CpuToGpu
        );

        //Allocate material buffer
        let global_material_slots = 1024;
        let material_size = 2 * size_of::<u32>() as u64;
        let material_buffer = GPUBuffer::allocate(
            vk,
            material_size * global_material_slots,
            storage_buffer_alignment,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            MemoryLocation::CpuToGpu
        );

        let max_vertices = 1024 * 1024 * 16;
        let alignment = vk.physical_device_properties.limits.min_storage_buffer_offset_alignment;

        //Allocate position buffer
        let position_buffer = GPUBuffer::allocate(
            vk,
            max_vertices * size_of::<glm::TVec4<f32>>() as u64,
            alignment,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            MemoryLocation::CpuToGpu
        );

        //Allocate tangent buffer
        let tangent_buffer = GPUBuffer::allocate(
            vk,
            max_vertices * size_of::<glm::TVec4<f32>>() as u64,
            alignment,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            MemoryLocation::CpuToGpu
        );

        //Allocate bitangent buffer
        let bitangent_buffer = GPUBuffer::allocate(
            vk,
            max_vertices * size_of::<glm::TVec4<f32>>() as u64,
            alignment,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            MemoryLocation::CpuToGpu
        );

        //Allocate uv buffer
        let uv_buffer = GPUBuffer::allocate(
            vk,
            max_vertices * size_of::<glm::TVec2<f32>>() as u64,
            alignment,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            MemoryLocation::CpuToGpu
        );

        //Set up descriptors
        let descriptor_set_layout;
        let descriptor_sets = unsafe {
            struct BufferDescriptorDesc {
                ty: vk::DescriptorType,
                stage_flags: vk::ShaderStageFlags,
                count: u32,
                buffer: vk::Buffer,
                offset: u64,
                length: u64
            }

            let buffer_descriptor_descs = [
                BufferDescriptorDesc {
                    ty: vk::DescriptorType::UNIFORM_BUFFER,
                    stage_flags: vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                    count: 1,
                    buffer: uniform_buffer.backing_buffer(),
                    offset: 0,
                    length: uniform_buffer.length()
                },
                BufferDescriptorDesc {
                    ty: vk::DescriptorType::STORAGE_BUFFER,
                    stage_flags: vk::ShaderStageFlags::VERTEX,
                    count: 1,
                    buffer: instance_buffer.backing_buffer(),
                    offset: 0,
                    length: instance_buffer.length()
                },
                BufferDescriptorDesc {
                    ty: vk::DescriptorType::STORAGE_BUFFER,
                    stage_flags: vk::ShaderStageFlags::FRAGMENT,
                    count: 1,
                    buffer: material_buffer.backing_buffer(),
                    offset: 0,
                    length: material_buffer.length()
                },
                BufferDescriptorDesc {
                    ty: vk::DescriptorType::STORAGE_BUFFER,
                    stage_flags: vk::ShaderStageFlags::VERTEX,
                    count: 1,
                    buffer: position_buffer.backing_buffer(),
                    offset: 0,
                    length: position_buffer.length()
                },
                BufferDescriptorDesc {
                    ty: vk::DescriptorType::STORAGE_BUFFER,
                    stage_flags: vk::ShaderStageFlags::VERTEX,
                    count: 1,
                    buffer: tangent_buffer.backing_buffer(),
                    offset: 0,
                    length: tangent_buffer.length()
                },
                BufferDescriptorDesc {
                    ty: vk::DescriptorType::STORAGE_BUFFER,
                    stage_flags: vk::ShaderStageFlags::VERTEX,
                    count: 1,
                    buffer: bitangent_buffer.backing_buffer(),
                    offset: 0,
                    length: bitangent_buffer.length()
                },
                BufferDescriptorDesc {
                    ty: vk::DescriptorType::STORAGE_BUFFER,
                    stage_flags: vk::ShaderStageFlags::VERTEX,
                    count: 1,
                    buffer: uv_buffer.backing_buffer(),
                    offset: 0,
                    length: uv_buffer.length()
                }
            ];
            
            let mut bindings = Vec::new();
            let mut pool_sizes = Vec::new();

            for i in 0..buffer_descriptor_descs.len() {
                let desc = &buffer_descriptor_descs[i];
                let binding = vk::DescriptorSetLayoutBinding {
                    binding: i as u32,
                    descriptor_type: desc.ty,
                    descriptor_count: desc.count,
                    stage_flags: desc.stage_flags,
                    ..Default::default()
                };
                bindings.push(binding);
                let pool_size = vk::DescriptorPoolSize {
                    ty: desc.ty,
                    descriptor_count: 1
                };
                pool_sizes.push(pool_size);
            }

            //Add global texture descriptor
            let binding = vk::DescriptorSetLayoutBinding {
                binding: buffer_descriptor_descs.len() as u32,
                descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                descriptor_count: global_textures.size() as u32,
                stage_flags: vk::ShaderStageFlags::FRAGMENT,
                ..Default::default()
            };
            bindings.push(binding);
            let pool_size = vk::DescriptorPoolSize {
                ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                descriptor_count: 1
            };
            pool_sizes.push(pool_size);

            let total_set_count = 1;
            let descriptor_pool_info = vk::DescriptorPoolCreateInfo {
                max_sets: total_set_count,
                pool_size_count: pool_sizes.len() as u32,
                p_pool_sizes: pool_sizes.as_ptr(),
                ..Default::default()
            };
            let descriptor_pool = vk.device.create_descriptor_pool(&descriptor_pool_info, vkutil::MEMORY_ALLOCATOR).unwrap();
            
            let descriptor_layout = vk::DescriptorSetLayoutCreateInfo {
                binding_count: bindings.len() as u32,
                p_bindings: bindings.as_ptr(),
                ..Default::default()
            };

            descriptor_set_layout = vk.device.create_descriptor_set_layout(&descriptor_layout, vkutil::MEMORY_ALLOCATOR).unwrap();

            let vk_alloc_info = vk::DescriptorSetAllocateInfo {
                descriptor_pool,
                descriptor_set_count: total_set_count,
                p_set_layouts: &descriptor_set_layout,
                ..Default::default()
            };
            let descriptor_sets = vk.device.allocate_descriptor_sets(&vk_alloc_info).unwrap();

            let mut desc_writes = Vec::new();
            let mut infos = Vec::new();
            for i in 0..buffer_descriptor_descs.len() {
                let desc = &buffer_descriptor_descs[i];
                let info = vk::DescriptorBufferInfo {
                    buffer: desc.buffer,
                    offset: desc.offset,
                    range: desc.length
                };
                infos.push(info);
                let write = vk::WriteDescriptorSet {
                    dst_set: descriptor_sets[0],
                    descriptor_count: 1,
                    descriptor_type: desc.ty,
                    p_buffer_info: &infos[i],
                    dst_array_element: 0,
                    dst_binding: i as u32,
                    ..Default::default()
                };
                desc_writes.push(write);
            }
            vk.device.update_descriptor_sets(&desc_writes, &[]);

            descriptor_sets
        };

        Renderer {
            models: OptionVec::new(),
            drawlist: Vec::new(),
            instance_data: Vec::new(),
            global_textures,
            descriptor_set_layout,
            descriptor_sets,
            position_buffer,
            position_offset: 0,
            tangent_buffer,
            tangent_offset: 0,
            bitangent_buffer,
            bitangent_offset: 0,
            uv_buffer,
            uv_offset: 0,
            uniform_buffer,
            instance_buffer,
            material_buffer
        }
    }

    pub fn register_model(&mut self, data: DrawData) -> usize {
        self.models.insert(data)
    }

    fn upload_vertex_attribute(data: &[f32], buffer: &GPUBuffer, offset: &mut u64) -> u32 {
        let old_offset = *offset;
        let new_offset = old_offset + (data.len() * size_of::<f32>()) as u64;
        buffer.upload_subbuffer(data, old_offset);
        *offset = new_offset;
        old_offset.try_into().unwrap()
    }
    
    pub fn upload_vertex_positions(&mut self, positions: &[f32]) -> u32 {
        Self::upload_vertex_attribute(positions, &self.position_buffer, &mut self.position_offset) / 16
    }
    
    pub fn upload_vertex_tangents(&mut self, tangents: &[f32]) -> u32 {
        Self::upload_vertex_attribute(tangents, &self.tangent_buffer, &mut self.tangent_offset) / 16
    }
    
    pub fn upload_vertex_bitangents(&mut self, bitangents: &[f32]) -> u32 {
        Self::upload_vertex_attribute(bitangents, &self.bitangent_buffer, &mut self.bitangent_offset) / 16
    }
    
    pub fn upload_vertex_uvs(&mut self, uvs: &[f32]) -> u32 {
        Self::upload_vertex_attribute(uvs, &self.uv_buffer, &mut self.uv_offset) / 8
    }

    pub fn get_model(&self, idx: usize) -> &Option<DrawData> {
        &self.models[idx]
    }

    pub fn queue_drawcall(&mut self, model_idx: usize, pipeline: vk::Pipeline, transforms: &[glm::TMat4<f32>]) {
        if let None = &self.models[model_idx] {
            tfd::message_box_ok("No model at supplied index", &format!("No model loaded at index {}", model_idx), tfd::MessageBoxIcon::Error);
            return;
        }

        let instance_count = transforms.len() as u32;
        let first_instance = self.instance_data.len() as u32;

        for t in transforms {
            let instance_data = InstanceData::new(*t);
            self.instance_data.push(instance_data);
        }
        let drawcall = DrawCall {
            geometry_idx: model_idx,
            pipeline,
            instance_count,
            first_instance
        };

        self.drawlist.push(drawcall);
    }

    pub fn drawlist_iter(&self) -> Iter<DrawCall> {
        self.drawlist.iter()
    }

    pub fn reset(&mut self) {
        self.drawlist.clear();
        self.instance_data.clear();
    }
}

//Values that will be uniform to a particular view. Probably shouldn't be called "FrameUniforms"
pub struct FrameUniforms {
    pub clip_from_screen: glm::TMat4<f32>,
    pub clip_from_world: glm::TMat4<f32>,
    pub clip_from_view: glm::TMat4<f32>,
    pub view_from_world: glm::TMat4<f32>,
    pub clip_from_skybox: glm::TMat4<f32>,
    pub camera_position: glm::TVec4<f32>,
    pub sun_direction: glm::TVec4<f32>,
    pub sun_radiance: glm::TVec3<f32>,
    pub time: f32,
    pub stars_threshold: f32, // modifies the number of stars that are visible
	pub stars_exposure: f32,  // modifies the overall strength of the stars
    pub prog_inter: f32
}
