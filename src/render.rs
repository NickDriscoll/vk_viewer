use core::slice::Iter;
use ash::vk::DescriptorImageInfo;

use crate::*;

#[derive(Clone, Debug)]
pub struct Material {
    pub base_color: [f32; 4],
    pub color_idx: u32,
    pub normal_idx: u32,
    _pad0: u32,
    _pad1: u32
}

impl Material {
    pub fn new(base_color: [f32; 4], color_idx: u32, normal_idx: u32) -> Self {
        Material {
            base_color,
            color_idx,
            normal_idx,
            _pad0: 0,
            _pad1: 0
        }
    }
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
    pub index_buffer: GPUBuffer,
    pub index_count: u32,
    pub position_offset: u32,
    pub tangent_offset: u32,
    pub normal_offset: u32,
    pub uv_offset: u32,
    pub material_idx: u32
}

fn minor(m: &[f32], r0: usize, r1: usize, r2: usize, c0: usize, c1: usize, c2: usize) -> f32 {
    return m[4*r0+c0] * (m[4*r1+c1] * m[4*r2+c2] - m[4*r2+c1] * m[4*r1+c2]) -
           m[4*r0+c1] * (m[4*r1+c0] * m[4*r2+c2] - m[4*r2+c0] * m[4*r1+c2]) +
           m[4*r0+c2] * (m[4*r1+c0] * m[4*r2+c1] - m[4*r2+c0] * m[4*r1+c1]);
}

//TL;DR use cofactor instead of transpose(inverse(world_from_model)) for normal matrix
//https://github.com/graphitemaster/normals_revisited
fn cofactor(src: &[f32], dst: &mut [f32]) {
    dst[ 0] =  minor(src, 1, 2, 3, 1, 2, 3);
    dst[ 1] = -minor(src, 1, 2, 3, 0, 2, 3);
    dst[ 2] =  minor(src, 1, 2, 3, 0, 1, 3);
    dst[ 3] = -minor(src, 1, 2, 3, 0, 1, 2);
    dst[ 4] = -minor(src, 0, 2, 3, 1, 2, 3);
    dst[ 5] =  minor(src, 0, 2, 3, 0, 2, 3);
    dst[ 6] = -minor(src, 0, 2, 3, 0, 1, 3);
    dst[ 7] =  minor(src, 0, 2, 3, 0, 1, 2);
    dst[ 8] =  minor(src, 0, 1, 3, 1, 2, 3);
    dst[ 9] = -minor(src, 0, 1, 3, 0, 2, 3);
    dst[10] =  minor(src, 0, 1, 3, 0, 1, 3);
    dst[11] = -minor(src, 0, 1, 3, 0, 1, 2);
    dst[12] = -minor(src, 0, 1, 2, 1, 2, 3);
    dst[13] =  minor(src, 0, 1, 2, 0, 2, 3);
    dst[14] = -minor(src, 0, 1, 2, 0, 1, 3);
    dst[15] =  minor(src, 0, 1, 2, 0, 1, 2);
}

pub struct InstanceData {
    pub world_from_model: glm::TMat4<f32>,
    pub normal_matrix: glm::TMat4<f32>
}

impl InstanceData {
    pub fn new(world_from_model: glm::TMat4<f32>) -> Self {
        //TL;DR use cofactor instead of transpose(inverse(world_from_model)) for normal matrix
        //https://github.com/graphitemaster/normals_revisited
        let mut normal_matrix: glm::TMat4<f32> = glm::identity();
        cofactor(world_from_model.as_slice(), normal_matrix.as_mut_slice());

        InstanceData {
            world_from_model,
            normal_matrix
        }
    }
}

pub struct Renderer {
    //pub uniforms: FrameUniforms,
    pub default_color_idx: u32,
    pub default_normal_idx: u32,
    pub material_sampler: vk::Sampler,
    pub point_sampler: vk::Sampler,
    primitives: OptionVec<DrawData>,
    drawlist: Vec<DrawCall>,
    instance_data: Vec<InstanceData>,

    pub position_buffer: GPUBuffer,
    position_offset: u64,
    pub tangent_buffer: GPUBuffer,
    tangent_offset: u64,
    pub normal_buffer: GPUBuffer,
    normal_offset: u64,
    pub uv_buffer: GPUBuffer,
    uv_offset: u64,
    pub imgui_buffer: GPUBuffer,
    imgui_offset: u64,
    pub uniform_buffer: GPUBuffer,
    pub instance_buffer: GPUBuffer,
    pub material_buffer: GPUBuffer,
    pub global_textures: FreeList<DescriptorImageInfo>,
    pub global_materials: FreeList<Material>,
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
        let mut global_textures = FreeList::with_capacity(1024);

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
        let usage_flags = vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST;

        //Allocate position buffer
        let position_buffer = GPUBuffer::allocate(
            vk,
            max_vertices * size_of::<glm::TVec4<f32>>() as u64,
            alignment,
            usage_flags,
            MemoryLocation::GpuOnly
        );

        //Allocate tangent buffer
        let tangent_buffer = GPUBuffer::allocate(
            vk,
            max_vertices * size_of::<glm::TVec4<f32>>() as u64,
            alignment,
            usage_flags,
            MemoryLocation::GpuOnly
        );

        //Allocate normal buffer
        let normal_buffer = GPUBuffer::allocate(
            vk,
            max_vertices * size_of::<glm::TVec4<f32>>() as u64,
            alignment,
            usage_flags,
            MemoryLocation::GpuOnly
        );

        //Allocate uv buffer
        let uv_buffer = GPUBuffer::allocate(
            vk,
            max_vertices * size_of::<glm::TVec2<f32>>() as u64,
            alignment,
            usage_flags,
            MemoryLocation::GpuOnly
        );

        //Allocate imgui buffer
        let max_imgui_vertices = 1024 * 1024;
        let imgui_buffer = GPUBuffer::allocate(
            vk,
            8 * max_imgui_vertices * size_of::<f32>() as u64,
            alignment,
            usage_flags,
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
                    buffer: normal_buffer.backing_buffer(),
                    offset: 0,
                    length: normal_buffer.length()
                },
                BufferDescriptorDesc {
                    ty: vk::DescriptorType::STORAGE_BUFFER,
                    stage_flags: vk::ShaderStageFlags::VERTEX,
                    count: 1,
                    buffer: uv_buffer.backing_buffer(),
                    offset: 0,
                    length: uv_buffer.length()
                },
                BufferDescriptorDesc {
                    ty: vk::DescriptorType::STORAGE_BUFFER,
                    stage_flags: vk::ShaderStageFlags::VERTEX,
                    count: 1,
                    buffer: imgui_buffer.backing_buffer(),
                    offset: 0,
                    length: imgui_buffer.length()
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

            for i in 0..buffer_descriptor_descs.len() {
                let desc = &buffer_descriptor_descs[i];
                let info = vk::DescriptorBufferInfo {
                    buffer: desc.buffer,
                    offset: desc.offset,
                    range: desc.length
                };
                let write = vk::WriteDescriptorSet {
                    dst_set: descriptor_sets[0],
                    descriptor_count: 1,
                    descriptor_type: desc.ty,
                    p_buffer_info: &info,
                    dst_array_element: 0,
                    dst_binding: i as u32,
                    ..Default::default()
                };
                vk.device.update_descriptor_sets(&[write], &[]);
            }

            descriptor_sets
        };

        //Create texture samplers
        let (material_sampler, point_sampler) = unsafe {
            let sampler_info = vk::SamplerCreateInfo {
                min_filter: vk::Filter::LINEAR,
                mag_filter: vk::Filter::LINEAR,
                mipmap_mode: vk::SamplerMipmapMode::LINEAR,
                address_mode_u: vk::SamplerAddressMode::REPEAT,
                address_mode_v: vk::SamplerAddressMode::REPEAT,
                address_mode_w: vk::SamplerAddressMode::REPEAT,
                mip_lod_bias: 0.0,
                anisotropy_enable: vk::FALSE,
                compare_enable: vk::FALSE,
                min_lod: 0.0,
                max_lod: vk::LOD_CLAMP_NONE,
                border_color: vk::BorderColor::FLOAT_OPAQUE_BLACK,
                unnormalized_coordinates: vk::FALSE,
                ..Default::default()
            };
            let mat = vk.device.create_sampler(&sampler_info, vkutil::MEMORY_ALLOCATOR).unwrap();
            
            let sampler_info = vk::SamplerCreateInfo {
                min_filter: vk::Filter::NEAREST,
                mag_filter: vk::Filter::NEAREST,
                mipmap_mode: vk::SamplerMipmapMode::NEAREST,
                ..sampler_info
            };
            let font = vk.device.create_sampler(&sampler_info, vkutil::MEMORY_ALLOCATOR).unwrap();
            
            (mat, font)
        };

        let default_color_idx = unsafe { global_textures.insert(vkutil::upload_raw_image(vk, point_sampler, vk::Format::R8G8B8A8_UNORM, 1, 1, &[0xFF, 0xFF, 0xFF, 0xFF])) as u32};
        let default_normal_idx = unsafe { global_textures.insert(vkutil::upload_raw_image(vk, point_sampler, vk::Format::R8G8B8A8_UNORM, 1, 1, &[0x80, 0x80, 0xFF, 0xFF])) as u32};

        //Create free list for materials
        let global_materials = FreeList::with_capacity(256);

        Renderer {
            //uniforms,
            default_color_idx,
            default_normal_idx,
            material_sampler,
            point_sampler,
            primitives: OptionVec::new(),
            drawlist: Vec::new(),
            instance_data: Vec::new(),
            global_textures,
            global_materials,
            descriptor_set_layout,
            descriptor_sets,
            position_buffer,
            position_offset: 0,
            tangent_buffer,
            tangent_offset: 0,
            normal_buffer,
            normal_offset: 0,
            uv_buffer,
            uv_offset: 0,
            imgui_buffer,
            imgui_offset: 0,
            uniform_buffer,
            instance_buffer,
            material_buffer
        }
    }

    pub fn register_model(&mut self, data: DrawData) -> usize {
        self.primitives.insert(data)
    }

    fn upload_vertex_attribute(vk: &mut VulkanAPI, data: &[f32], buffer: &GPUBuffer, offset: &mut u64) -> u32 {
        let old_offset = *offset;
        let new_offset = old_offset + data.len() as u64;
        buffer.upload_subbuffer(vk, data, old_offset);
        *offset = new_offset;
        old_offset.try_into().unwrap()
    }
    
    pub fn append_vertex_positions(&mut self, vk: &mut VulkanAPI, positions: &[f32]) -> u32 {
        Self::upload_vertex_attribute(vk, positions, &self.position_buffer, &mut self.position_offset) / 4
    }
    
    pub fn append_vertex_tangents(&mut self, vk: &mut VulkanAPI, tangents: &[f32]) -> u32 {
        Self::upload_vertex_attribute(vk, tangents, &self.tangent_buffer, &mut self.tangent_offset) / 4
    }
    
    pub fn append_vertex_normals(&mut self, vk: &mut VulkanAPI, normals: &[f32]) -> u32 {
        Self::upload_vertex_attribute(vk, normals, &self.normal_buffer, &mut self.normal_offset) / 4
    }
    
    pub fn append_vertex_uvs(&mut self, vk: &mut VulkanAPI, uvs: &[f32]) -> u32 {
        Self::upload_vertex_attribute(vk, uvs, &self.uv_buffer, &mut self.uv_offset) / 2
    }

    pub fn replace_vertex_positions(&mut self, vk: &mut VulkanAPI, data: &[f32], offset: u64) {
        let mut my_offset = offset * 4;
        Self::upload_vertex_attribute(vk, data, &self.position_buffer, &mut my_offset);
    }

    pub fn replace_vertex_tangents(&mut self, vk: &mut VulkanAPI, data: &[f32], offset: u64) {
        let mut my_offset = offset * 4;
        Self::upload_vertex_attribute(vk, data, &self.tangent_buffer, &mut my_offset);
    }

    pub fn replace_vertex_normals(&mut self, vk: &mut VulkanAPI, data: &[f32], offset: u64) {
        let mut my_offset = offset * 4;
        Self::upload_vertex_attribute(vk, data, &self.normal_buffer, &mut my_offset);
    }

    pub fn replace_vertex_uvs(&mut self, vk: &mut VulkanAPI, data: &[f32], offset: u64) {
        let mut my_offset = offset * 2;
        Self::upload_vertex_attribute(vk, data, &self.uv_buffer, &mut my_offset);
    }

    pub fn replace_imgui_vertices(&mut self, vk: &mut VulkanAPI, data: &[f32], offset: u64) {
        let mut my_offset = offset * DevGui::FLOATS_PER_VERTEX as u64;
        Self::upload_vertex_attribute(vk, data, &self.imgui_buffer, &mut my_offset);
    }

    pub fn get_model(&self, idx: usize) -> &Option<DrawData> {
        &self.primitives[idx]
    }

    pub fn queue_drawcall(&mut self, model_idx: usize, pipeline: vk::Pipeline, transforms: &[glm::TMat4<f32>]) {
        if let None = &self.primitives[model_idx] {
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
	pub stars_exposure: f32,  // modifies the overall strength of the stars,
    pub sunzenith_idx: u32,
    pub viewzenith_idx: u32,
    pub sunview_idx: u32
}
