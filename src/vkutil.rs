use std::{ffi::c_void, io::Read, ops::Index};

use ash::vk;
use gpu_allocator::vulkan::*;
use gpu_allocator::MemoryLocation;
use ozy::io::DDSHeader;
use sdl2::libc::c_char;
use sdl2::video::Window;
use std::ptr;
use routines::*;
use crate::*;

pub const MEMORY_ALLOCATOR: Option<&vk::AllocationCallbacks> = None;

pub const COLOR_CLEAR: vk::ClearValue = {
    let color = vk::ClearColorValue {
        float32: [0.0, 0.0, 0.0, 1.0]           //float32: [0.26, 0.4, 0.46, 1.0]
    };
    vk::ClearValue {
        color
    }
};

pub const DEPTH_STENCIL_CLEAR: vk::ClearValue = {
    let value = vk::ClearDepthStencilValue {
        depth: 1.0,
        stencil: 0
    };
    vk::ClearValue {
        depth_stencil: value
    }
};

pub const COMPONENT_MAPPING_DEFAULT: vk::ComponentMapping = vk::ComponentMapping {
    r: vk::ComponentSwizzle::R,
    g: vk::ComponentSwizzle::G,
    b: vk::ComponentSwizzle::B,
    a: vk::ComponentSwizzle::A,
};

//Macro to fit a given desired size to a given alignment without worrying about the specific integer type
#[macro_export]
macro_rules! size_to_alignment {
    ($in_size:expr, $alignment:expr) => {
        {
            let mut final_size = $in_size;
            if $alignment > 0 {
                final_size = (final_size + ($alignment - 1)) & !($alignment - 1);   //Alignment is 2^N where N is a whole number
            }
            final_size
        }
    };
}

pub struct VertexFetchOffsets {
    pub position_offset: u32,
    pub tangent_offset: u32,
    pub normal_offset: u32,
    pub uv_offset: u32,
}

pub fn load_shader_stage(vk_device: &ash::Device, shader_stage_flags: vk::ShaderStageFlags, path: &str) -> vk::PipelineShaderStageCreateInfo {
    let msg = format!("Unable to read spv file {}\nDid a shader fail to compile?", path);
    let mut file = unwrap_result(File::open(path), &msg);
    let spv = unwrap_result(ash::util::read_spv(&mut file), &msg);

    let module_create_info = vk::ShaderModuleCreateInfo {
        code_size: spv.len() * size_of::<u32>(),
        p_code: spv.as_ptr(),
        ..Default::default()
    };
    let module = unsafe { vk_device.create_shader_module(&module_create_info, MEMORY_ALLOCATOR).unwrap() };

    vk::PipelineShaderStageCreateInfo {
        stage: shader_stage_flags,
        p_name: "main\0".as_ptr() as *const i8,
        module,
        ..Default::default()
    }
}

pub unsafe fn allocate_image_memory(vk: &mut VulkanAPI, image: vk::Image) -> Allocation {
    allocate_named_image_memory(vk, image, "")
}

pub unsafe fn allocate_named_image_memory(vk: &mut VulkanAPI, image: vk::Image, name: &str) -> Allocation {
    let requirements = vk.device.get_image_memory_requirements(image);
    let alloc = vk.allocator.allocate(&AllocationCreateDesc {
        name,
        requirements,
        location: MemoryLocation::GpuOnly,
        linear: false       //We want tiled memory for images
    }).unwrap();

    vk.device.bind_image_memory(image, alloc.memory(), alloc.offset()).unwrap();
    alloc
}

pub fn load_png_texture(vk: &mut VulkanAPI, global_textures: &mut FreeList<vk::DescriptorImageInfo>, sampler: vk::Sampler, path: &str) -> u32 {
    let vim = GPUImage::from_png_file(vk, path);

    let descriptor_info = vk::DescriptorImageInfo {
        sampler,
        image_view: vim.vk_view,
        image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
    };
    global_textures.insert(descriptor_info) as u32
}

pub fn load_bc7_texture(vk: &mut VulkanAPI, global_textures: &mut FreeList<vk::DescriptorImageInfo>, sampler: vk::Sampler, path: &str, color_space: ColorSpace) -> u32 {
    unsafe {
        let vim = GPUImage::from_bc7(vk, path, color_space);

        let descriptor_info = vk::DescriptorImageInfo {
            sampler,
            image_view: vim.vk_view,
            image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
        };
        let index = global_textures.insert(descriptor_info);

        index as u32
    }
}

pub unsafe fn upload_raw_image(vk: &mut VulkanAPI, sampler: vk::Sampler, format: vk::Format, width: u32, height: u32, rgba: &[u8]) -> vk::DescriptorImageInfo {
    let image_create_info = vk::ImageCreateInfo {
        image_type: vk::ImageType::TYPE_2D,
        format,
        extent: vk::Extent3D {
            width,
            height,
            depth: 1
        },
        mip_levels: 1,
        array_layers: 1,
        samples: vk::SampleCountFlags::TYPE_1,
        tiling: vk::ImageTiling::OPTIMAL,
        usage: vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED,
        sharing_mode: vk::SharingMode::EXCLUSIVE,
        queue_family_index_count: 1,
        p_queue_family_indices: &vk.queue_family_index,
        initial_layout: vk::ImageLayout::UNDEFINED,
        ..Default::default()
    };
    let normal_image = vk.device.create_image(&image_create_info, MEMORY_ALLOCATOR).unwrap();
    let allocation = allocate_image_memory(vk, normal_image);

    let vim = GPUImage {
        vk_image: normal_image,
        vk_view: vk::ImageView::default(),
        width,
        height,
        mip_count: 1,
        allocation
    };
    upload_image(vk, &vim, &rgba);

    //Then create the image view
    let sampler_subresource_range = vk::ImageSubresourceRange {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        base_mip_level: 0,
        level_count: 1,
        base_array_layer: 0,
        layer_count: 1
    };
    let view_info = vk::ImageViewCreateInfo {
        image: normal_image,
        format,
        view_type: vk::ImageViewType::TYPE_2D,
        components: COMPONENT_MAPPING_DEFAULT,
        subresource_range: sampler_subresource_range,
        ..Default::default()
    };
    let view = vk.device.create_image_view(&view_info, MEMORY_ALLOCATOR).unwrap();
    
    vk::DescriptorImageInfo {
        sampler,
        image_view: view,
        image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
    }
}

pub enum ColorSpace {
    LINEAR,
    SRGB
}

pub struct GPUImage {
    pub vk_image: vk::Image,
    pub vk_view: vk::ImageView,
    pub width: u32,
    pub height: u32,
    pub mip_count: u32,
    pub allocation: Allocation
}

impl GPUImage {
    pub fn from_png_file(vk: &mut VulkanAPI, path: &str) -> Self {
        let mut file = unwrap_result(File::open(path), &format!("Error opening png {}", path));
        let mut png_bytes = vec![0u8; file.metadata().unwrap().len().try_into().unwrap()];
        file.read_exact(&mut png_bytes).unwrap();
        Self::from_png_bytes(vk, &png_bytes)
    }

    pub fn from_png_bytes(vk: &mut VulkanAPI, png_bytes: &[u8]) -> Self {
        use png::BitDepth;
        use png::ColorType;

        let decoder = png::Decoder::new(png_bytes);
        let mut reader = decoder.read_info().unwrap();
        let info = reader.info();

        //We're given a depth in bits, so we set up an integer divide
        let byte_size_divisor = match info.bit_depth {
            BitDepth::One => { 8 }
            BitDepth::Two => { 4 }
            BitDepth::Four => { 2 }
            BitDepth::Eight => { 1 }
            _ => { crash_with_error_dialog("Unsupported PNG bitdepth"); }
        };

        let width = info.width;
        let height = info.height;
        let pixel_count = (width * height / byte_size_divisor) as usize;
        let format;
        let bytes = match info.color_type {
            ColorType::Rgb => {
                format = match info.srgb {
                    Some(_) => { vk::Format::R8G8B8A8_SRGB }
                    None => { vk::Format::R8G8B8A8_UNORM }
                };

                let mut raw_bytes = vec![0u8; 3 * pixel_count];
                reader.next_frame(&mut raw_bytes).unwrap();

                //Convert to RGBA by adding an alpha of 1.0 to each pixel
                let mut bytes = vec![0xFFu8; 4 * pixel_count];
                for i in 0..pixel_count {
                    let idx = 4 * i;
                    let r_idx = 3 * i;
                    bytes[idx] = raw_bytes[r_idx];
                    bytes[idx + 1] = raw_bytes[r_idx + 1];
                    bytes[idx + 2] = raw_bytes[r_idx + 2];
                }
                bytes
            }
            ColorType::Rgba => {
                format = match info.srgb {
                    Some(_) => { vk::Format::R8G8B8A8_SRGB }
                    None => { vk::Format::R8G8B8A8_UNORM }
                };

                let mut bytes = vec![0xFFu8; 4 * pixel_count];
                reader.next_frame(&mut bytes).unwrap();
                bytes
            }
            t => { crash_with_error_dialog(&format!("Unsupported color type: {:?}", t)); }
        };

        unsafe {
            let mip_levels =  (f32::floor(f32::log2(u32::max(width, height) as f32))) as u32;

            let image_create_info = vk::ImageCreateInfo {
                image_type: vk::ImageType::TYPE_2D,
                format,
                extent: vk::Extent3D {
                    width,
                    height,
                    depth: 1
                },
                mip_levels,
                array_layers: 1,
                samples: vk::SampleCountFlags::TYPE_1,
                tiling: vk::ImageTiling::OPTIMAL,
                usage: vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED,
                sharing_mode: vk::SharingMode::EXCLUSIVE,
                queue_family_index_count: 1,
                p_queue_family_indices: &vk.queue_family_index,
                initial_layout: vk::ImageLayout::UNDEFINED,
                ..Default::default()
            };
            let image = vk.device.create_image(&image_create_info, MEMORY_ALLOCATOR).unwrap();
            let allocation = allocate_image_memory(vk, image);

            let mut vim = GPUImage {
                vk_image: image,
                vk_view: vk::ImageView::default(),
                width,
                height,
                mip_count: mip_levels,
                allocation
            };
            upload_image(vk, &vim, &bytes);

            //Generate mipmaps
            vk.device.begin_command_buffer(vk.general_command_buffer, &vk::CommandBufferBeginInfo::default()).unwrap();

            let image_memory_barrier = vk::ImageMemoryBarrier {
                src_access_mask: vk::AccessFlags::empty(),
                dst_access_mask: vk::AccessFlags::TRANSFER_WRITE,
                old_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                new_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                image,
                subresource_range: vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: mip_levels,
                    base_array_layer: 0,
                    layer_count: 1
                },
                ..Default::default()
            };
            vk.device.cmd_pipeline_barrier(
                vk.general_command_buffer,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[], &[],
                &[image_memory_barrier]
            );
        
            for i in 0..(mip_levels - 1) {
                let src_mip_barrier = vk::ImageMemoryBarrier {
                    src_access_mask: vk::AccessFlags::empty(),
                    dst_access_mask: vk::AccessFlags::TRANSFER_WRITE,
                    old_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    new_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    image,
                    subresource_range: vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: i,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1
                    },
                    ..Default::default()
                };
                vk.device.cmd_pipeline_barrier(
                    vk.general_command_buffer,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    &[], &[],
                    &[src_mip_barrier]
                );

                let src_subresource = vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: i,
                    base_array_layer: 0,
                    layer_count: 1
                };
                let dst_subresource = vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: i + 1,
                    base_array_layer: 0,
                    layer_count: 1
                };
                let src_offsets = [
                    vk::Offset3D {x: 0, y: 0, z: 0},
                    vk::Offset3D {x: (width >> i) as i32, y: (height >> i) as i32, z: 1}
                ];
                let dst_offsets = [
                    vk::Offset3D {x: 0, y: 0, z: 0},
                    vk::Offset3D {x: (width >> (i + 1)) as i32, y: (height >> (i + 1)) as i32, z: 1}
                ];
                let regions = [
                    vk::ImageBlit {
                        src_subresource,
                        src_offsets,
                        dst_subresource,
                        dst_offsets
                    }
                ];
                vk.device.cmd_blit_image(
                    vk.general_command_buffer,
                    image,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &regions,
                    vk::Filter::LINEAR
                );
            }

            let subresource_range = vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: mip_levels,
                base_array_layer: 0,
                layer_count: 1
            };
            let image_memory_barrier = vk::ImageMemoryBarrier {
                src_access_mask: vk::AccessFlags::TRANSFER_WRITE,
                dst_access_mask: vk::AccessFlags::SHADER_READ,
                old_layout: vk::ImageLayout::UNDEFINED,
                new_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                image,
                subresource_range,
                ..Default::default()
            };
            vk.device.cmd_pipeline_barrier(vk.general_command_buffer, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::FRAGMENT_SHADER, vk::DependencyFlags::empty(), &[], &[], &[image_memory_barrier]);

            vk.device.end_command_buffer(vk.general_command_buffer).unwrap();
    
            let submit_info = vk::SubmitInfo {
                command_buffer_count: 1,
                p_command_buffers: &vk.general_command_buffer,
                ..Default::default()
            };
            let queue = vk.device.get_device_queue(vk.queue_family_index, 0);
            vk.device.wait_for_fences(&[vk.general_command_buffer_fence], true, vk::DeviceSize::MAX).unwrap();
            vk.device.reset_fences(&[vk.general_command_buffer_fence]).unwrap();
            vk.device.queue_submit(queue, &[submit_info], vk.general_command_buffer_fence).unwrap();
            vk.device.wait_for_fences(&[vk.general_command_buffer_fence], true, vk::DeviceSize::MAX).unwrap();
            
            let sampler_subresource_range = vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: mip_levels,
                base_array_layer: 0,
                layer_count: 1
            };
            let grass_view_info = vk::ImageViewCreateInfo {
                image,
                format,
                view_type: vk::ImageViewType::TYPE_2D,
                components: COMPONENT_MAPPING_DEFAULT,
                subresource_range: sampler_subresource_range,
                ..Default::default()
            };
            let view = vk.device.create_image_view(&grass_view_info, MEMORY_ALLOCATOR).unwrap();

            vim.vk_view = view;
            vim
        }
    }

    pub unsafe fn from_bc7(vk: &mut VulkanAPI, path: &str, color_space: ColorSpace) -> Self {
        let mut file = unwrap_result(File::open(path), &format!("Error opening bc7 {}", path));
        let dds_header = DDSHeader::from_file(&mut file);       //This also advances the file read head to the beginning of the raw data section

        let width = dds_header.width;
        let height = dds_header.height;
        let mipmap_count = dds_header.mipmap_count;

        let mut bytes_size = 0;
        for i in 0..mipmap_count {
            let w = width / (1 << i);
            let h = height / (1 << i);
            bytes_size += w * h;

            bytes_size = size_to_alignment!(bytes_size, 16);
        }

        let mut raw_bytes = vec![0u8; bytes_size as usize];
        file.read_exact(&mut raw_bytes).unwrap();
        
        let format = match color_space {
            ColorSpace::LINEAR => {
                vk::Format::BC7_UNORM_BLOCK
            }
            ColorSpace::SRGB => {
                vk::Format::BC7_SRGB_BLOCK
            }
        };

        let image_extent = vk::Extent3D {
            width,
            height,
            depth: 1
        };
        let image_create_info = vk::ImageCreateInfo {
            image_type: vk::ImageType::TYPE_2D,
            format,
            extent: image_extent,
            mip_levels: mipmap_count,
            array_layers: 1,
            samples: vk::SampleCountFlags::TYPE_1,
            tiling: vk::ImageTiling::OPTIMAL,
            usage: vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            queue_family_index_count: 1,
            p_queue_family_indices: &vk.queue_family_index,
            initial_layout: vk::ImageLayout::UNDEFINED,
            ..Default::default()
        };
        let image = vk.device.create_image(&image_create_info, MEMORY_ALLOCATOR).unwrap();
        let allocation = allocate_image_memory(vk, image);

        let mut vim = GPUImage {
            vk_image: image,
            vk_view: vk::ImageView::default(),
            width,
            height,
            mip_count: mipmap_count,
            allocation
        };
        upload_image(vk, &vim, &raw_bytes);
        
        let sampler_subresource_range = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: mipmap_count,
            base_array_layer: 0,
            layer_count: 1
        };
        let grass_view_info = vk::ImageViewCreateInfo {
            image: image,
            format,
            view_type: vk::ImageViewType::TYPE_2D,
            components: COMPONENT_MAPPING_DEFAULT,
            subresource_range: sampler_subresource_range,
            ..Default::default()
        };
        let view = vk.device.create_image_view(&grass_view_info, MEMORY_ALLOCATOR).unwrap();

        vim.vk_view = view;
        vim
    }
}

pub unsafe fn upload_GPU_buffer<T>(vk: &mut VulkanAPI, dst_buffer: vk::Buffer, offset: u64, raw_data: &[T]) {
    //Create staging buffer and upload raw buffer data
    let bytes_size = (raw_data.len() * size_of::<T>()) as vk::DeviceSize;
    let staging_buffer = GPUBuffer::allocate(vk, bytes_size, 0, vk::BufferUsageFlags::TRANSFER_SRC, MemoryLocation::CpuToGpu);
    staging_buffer.upload_buffer(vk, &raw_data);

    //Wait on the fence before beginning command recording
    vk.device.wait_for_fences(&[vk.general_command_buffer_fence], true, vk::DeviceSize::MAX).unwrap();
    vk.device.reset_fences(&[vk.general_command_buffer_fence]).unwrap();
    vk.device.begin_command_buffer(vk.general_command_buffer, &vk::CommandBufferBeginInfo::default()).unwrap();

    let copy = vk::BufferCopy {
        src_offset: 0,
        dst_offset: offset * size_of::<T>() as u64,
        size: bytes_size
    };
    vk.device.cmd_copy_buffer(vk.general_command_buffer, staging_buffer.backing_buffer(), dst_buffer, &[copy]);

    vk.device.end_command_buffer(vk.general_command_buffer).unwrap();

    let submit_info = vk::SubmitInfo {
        command_buffer_count: 1,
        p_command_buffers: &vk.general_command_buffer,
        ..Default::default()
    };
    let queue = vk.device.get_device_queue(vk.queue_family_index, 0);
    vk.device.queue_submit(queue, &[submit_info], vk.general_command_buffer_fence).unwrap();
    vk.device.wait_for_fences(&[vk.general_command_buffer_fence], true, vk::DeviceSize::MAX).unwrap();
    staging_buffer.free(vk);
}

pub unsafe fn upload_image(vk: &mut VulkanAPI, image: &GPUImage, raw_bytes: &[u8]) {
    //Create staging buffer and upload raw image data
    let bytes_size = raw_bytes.len() as vk::DeviceSize;
    let staging_buffer = GPUBuffer::allocate(vk, bytes_size, 0, vk::BufferUsageFlags::TRANSFER_SRC, MemoryLocation::CpuToGpu);
    staging_buffer.upload_buffer(vk, &raw_bytes);

    //Wait on the fence before beginning command recording
    vk.device.wait_for_fences(&[vk.general_command_buffer_fence], true, vk::DeviceSize::MAX).unwrap();
    vk.device.reset_fences(&[vk.general_command_buffer_fence]).unwrap();
    vk.device.begin_command_buffer(vk.general_command_buffer, &vk::CommandBufferBeginInfo::default()).unwrap();

    let image_memory_barrier = vk::ImageMemoryBarrier {
        src_access_mask: vk::AccessFlags::empty(),
        dst_access_mask: vk::AccessFlags::TRANSFER_WRITE,
        old_layout: vk::ImageLayout::UNDEFINED,
        new_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        image: image.vk_image,
        subresource_range: vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: image.mip_count,
            base_array_layer: 0,
            layer_count: 1
        },
        ..Default::default()
    };
    vk.device.cmd_pipeline_barrier(vk.general_command_buffer, vk::PipelineStageFlags::TOP_OF_PIPE, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[], &[], &[image_memory_barrier]);

    let mut cumulative_offset = 0;
    let mut copy_regions = vec![vk::BufferImageCopy::default(); image.mip_count as usize];
    for i in 0..image.mip_count {
        let w = image.width / (1 << i);
        let h = image.height / (1 << i);
        let subresource_layers = vk::ImageSubresourceLayers {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            mip_level: i,
            base_array_layer: 0,
            layer_count: 1

        };
        let image_extent = vk::Extent3D {
            width: u32::max(w, 1),
            height: u32::max(h, 1),
            depth: 1
        };

        cumulative_offset = size_to_alignment!(cumulative_offset, 16);
        let copy_region = vk::BufferImageCopy {
            buffer_offset: cumulative_offset as u64,
            buffer_row_length: 0,
            buffer_image_height: 0,
            image_extent,
            image_offset: vk::Offset3D::default(),
            image_subresource: subresource_layers
        };
        copy_regions[i as usize] = copy_region;
        cumulative_offset += w * h;
    }

    vk.device.cmd_copy_buffer_to_image(vk.general_command_buffer, staging_buffer.backing_buffer(), image.vk_image, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &copy_regions);

    let subresource_range = vk::ImageSubresourceRange {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        base_mip_level: 0,
        level_count: image.mip_count,
        base_array_layer: 0,
        layer_count: 1
    };
    let image_memory_barrier = vk::ImageMemoryBarrier {
        src_access_mask: vk::AccessFlags::TRANSFER_WRITE,
        dst_access_mask: vk::AccessFlags::SHADER_READ,
        old_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        new_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        image: image.vk_image,
        subresource_range,
        ..Default::default()
    };
    vk.device.cmd_pipeline_barrier(vk.general_command_buffer, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::FRAGMENT_SHADER, vk::DependencyFlags::empty(), &[], &[], &[image_memory_barrier]);

    vk.device.end_command_buffer(vk.general_command_buffer).unwrap();
    
    let submit_info = vk::SubmitInfo {
        command_buffer_count: 1,
        p_command_buffers: &vk.general_command_buffer,
        ..Default::default()
    };
    let queue = vk.device.get_device_queue(vk.queue_family_index, 0);
    vk.device.queue_submit(queue, &[submit_info], vk.general_command_buffer_fence).unwrap();
    vk.device.wait_for_fences(&[vk.general_command_buffer_fence], true, vk::DeviceSize::MAX).unwrap();
    staging_buffer.free(vk);
}

pub fn make_index_buffer(vk: &mut VulkanAPI, indices: &[u32]) -> GPUBuffer {
    let index_buffer = GPUBuffer::allocate(
        vk,
        (indices.len() * size_of::<u32>()) as vk::DeviceSize,
        0,
        vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
        MemoryLocation::GpuOnly
    );
    index_buffer.upload_buffer(vk, indices);
    index_buffer
}

//All the variables that Vulkan needs
pub struct VulkanAPI {
    pub instance: ash::Instance,
    pub physical_device: vk::PhysicalDevice,
    pub physical_device_properties: vk::PhysicalDeviceProperties,
    pub device: ash::Device,
    pub allocator: Allocator,
    pub ext_surface: ash::extensions::khr::Surface,
    pub ext_swapchain: ash::extensions::khr::Swapchain,
    pub queue_family_index: u32,
    pub command_pool: vk::CommandPool,
    pub general_command_buffer: vk::CommandBuffer,
    pub general_command_buffer_fence: vk::Fence
}

impl VulkanAPI {
    pub fn init(window: &Window) -> Self {
        let vk_entry = ash::Entry::linked();
        let vk_instance = {
            let app_info = vk::ApplicationInfo {
                api_version: vk::make_api_version(0, 1, 3, 0),
                ..Default::default()
            };

            #[cfg(target_os = "windows")]
            let platform_surface_extension = ash::extensions::khr::Win32Surface::name().as_ptr();
            
            #[cfg(target_os = "macos")]
            let platform_surface_extension = ash::extensions::mvk::MacOSSurface::name().as_ptr();

            #[cfg(target_os = "linux")]
            let platform_surface_extension = ash::extensions::khr::XlibSurface::name().as_ptr();

            let extension_names = [
                ash::extensions::khr::Surface::name().as_ptr(),
                platform_surface_extension
            ];

            let layer_names = unsafe  {[
                CStr::from_bytes_with_nul_unchecked(b"VK_LAYER_KHRONOS_validation\0").as_ptr()
            ]};
            let vk_create_info = vk::InstanceCreateInfo {
                p_application_info: &app_info,
                enabled_extension_count: extension_names.len() as u32,
                pp_enabled_extension_names: &extension_names as *const *const c_char,
                enabled_layer_count: layer_names.len() as u32,
                pp_enabled_layer_names: &layer_names as *const *const c_char,
                ..Default::default()
            };

            unsafe { vk_entry.create_instance(&vk_create_info, MEMORY_ALLOCATOR).unwrap() }
        };

        let vk_ext_surface = ash::extensions::khr::Surface::new(&vk_entry, &vk_instance);

        //Create the Vulkan device
        let vk_physical_device;
        let vk_physical_device_properties;
        let mut queue_family_index = 0;
        let buffer_device_address;
        let vk_device = unsafe {
            let phys_devices = vk_instance.enumerate_physical_devices();
            if let Err(e) = phys_devices {
                crash_with_error_dialog(&format!("Unable to enumerate physical devices: {}", e));
            }
            let phys_devices = phys_devices.unwrap();

            //Search for the physical device
            const DEVICE_TYPES: [vk::PhysicalDeviceType; 3] = [
                vk::PhysicalDeviceType::DISCRETE_GPU,
                vk::PhysicalDeviceType::INTEGRATED_GPU,
                vk::PhysicalDeviceType::CPU
            ];
            let mut phys_device = None;
            'gpu_search: for d_type in DEVICE_TYPES {
                for device in phys_devices.iter() {
                    let props = vk_instance.get_physical_device_properties(*device);
                    if props.device_type == d_type {
                        let name = CStr::from_ptr(props.device_name.as_ptr()).to_str().unwrap();
                        println!("\"{}\" was chosen as 3D accelerator.", name);
                        phys_device = Some(*device);
                        break 'gpu_search;
                    }
                }
            }

            vk_physical_device = phys_device.unwrap();
            vk_physical_device_properties = vk_instance.get_physical_device_properties(vk_physical_device);
            
            //Get physical device features
            let mut indexing_features = vk::PhysicalDeviceDescriptorIndexingFeatures::default();
            let mut buffer_address_features = vk::PhysicalDeviceBufferDeviceAddressFeatures::default();
            indexing_features.p_next = &mut buffer_address_features as *mut _ as *mut c_void;
            let mut physical_device_features = vk::PhysicalDeviceFeatures2 {
                p_next: &mut indexing_features as *mut _ as *mut c_void,
                ..Default::default()
            };
            vk_instance.get_physical_device_features2(vk_physical_device, &mut physical_device_features);
            if physical_device_features.features.texture_compression_bc == vk::FALSE {
                tfd::message_box_ok("WARNING", "GPU compressed textures are not supported by this GPU.\nYou may be able to get away with this...", tfd::MessageBoxIcon::Warning);
            }
            buffer_device_address = buffer_address_features.buffer_device_address != 0;
            
            if indexing_features.descriptor_binding_partially_bound == 0 || indexing_features.runtime_descriptor_array == 0 {
                crash_with_error_dialog("Your GPU lacks the specific features required to do bindless rendering. Sorry.");
            }

            let mut i = 0;
            let qfps = vk_instance.get_physical_device_queue_family_properties(vk_physical_device);
            for qfp in qfps {
                if qfp.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
                    queue_family_index = i;
                    break;
                }
                i += 1;
            }

            let queue_create_info = vk::DeviceQueueCreateInfo {
                queue_family_index,
                queue_count: 1,
                p_queue_priorities: [1.0].as_ptr(),
                ..Default::default()
            };

            let extension_names = [ash::extensions::khr::Swapchain::name().as_ptr()];
            let create_info = vk::DeviceCreateInfo {
                queue_create_info_count: 1,
                p_queue_create_infos: [queue_create_info].as_ptr(),
                enabled_extension_count: extension_names.len() as u32,
                pp_enabled_extension_names: extension_names.as_ptr(),
                p_enabled_features: &physical_device_features.features,
                p_next: &mut indexing_features as *mut _ as *mut c_void,
                ..Default::default()
            };

            vk_instance.create_device(vk_physical_device, &create_info, MEMORY_ALLOCATOR).unwrap()
        };
        
        let vk_ext_swapchain = ash::extensions::khr::Swapchain::new(&vk_instance, &vk_device);

        //Initialize gpu_allocator
        let allocator = Allocator::new(&AllocatorCreateDesc {
            instance: vk_instance.clone(),
            device: vk_device.clone(),
            physical_device: vk_physical_device,
            debug_settings: Default::default(),
            buffer_device_address
        }).unwrap();

        let command_pool = unsafe {
            let pool_create_info = vk::CommandPoolCreateInfo {
                queue_family_index,
                flags: vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
                ..Default::default()
            };
    
            vk_device.create_command_pool(&pool_create_info, MEMORY_ALLOCATOR).unwrap()
        };

        //Create command buffer
        let graphics_command_buffer = unsafe {
    
            let command_buffer_alloc_info = vk::CommandBufferAllocateInfo {
                command_pool,
                command_buffer_count: 1,
                level: vk::CommandBufferLevel::PRIMARY,
                ..Default::default()
            };
            vk_device.allocate_command_buffers(&command_buffer_alloc_info).unwrap()[0]
        };

        let graphics_command_buffer_fence = unsafe {
            let create_info = vk::FenceCreateInfo {
                flags: vk::FenceCreateFlags::SIGNALED,
                ..Default::default()
            };
            vk_device.create_fence(&create_info, MEMORY_ALLOCATOR).unwrap()
        };

        VulkanAPI {
            instance: vk_instance,
            physical_device: vk_physical_device,
            physical_device_properties: vk_physical_device_properties,
            device: vk_device,
            allocator,
            ext_surface: vk_ext_surface,
            ext_swapchain: vk_ext_swapchain,
            queue_family_index,
            command_pool,
            general_command_buffer: graphics_command_buffer,
            general_command_buffer_fence: graphics_command_buffer_fence
        }
    }
}

pub struct GPUBuffer {
    buffer: vk::Buffer,
    allocation: Allocation,
    length: vk::DeviceSize
}

impl GPUBuffer {
    //Read-only access to fields
    pub fn backing_buffer(&self) -> vk::Buffer { self.buffer }
    pub fn length(&self) -> vk::DeviceSize { self.length }

    pub fn allocate(vk: &mut VulkanAPI, size: vk::DeviceSize, alignment: vk::DeviceSize, usage_flags: vk::BufferUsageFlags, memory_location: MemoryLocation) -> Self {
        let vk_buffer;
        let actual_size = size_to_alignment!(size, alignment);
        let allocation = unsafe {
            let buffer_create_info = vk::BufferCreateInfo {
                usage: usage_flags,
                size: actual_size,
                sharing_mode: vk::SharingMode::EXCLUSIVE,
                ..Default::default()
            };
            vk_buffer = vk.device.create_buffer(&buffer_create_info, MEMORY_ALLOCATOR).unwrap();
            let mem_reqs = vk.device.get_buffer_memory_requirements(vk_buffer);

            let a = vk.allocator.allocate(&AllocationCreateDesc {
                name: "",
                requirements: mem_reqs,
                location: memory_location,
                linear: true
            }).unwrap();
            vk.device.bind_buffer_memory(vk_buffer, a.memory(), a.offset()).unwrap();
            a
        };

        GPUBuffer {
            buffer: vk_buffer,
            allocation,
            length: actual_size
        }
    }

    pub fn free(self, vk: &mut VulkanAPI) {
        vk.allocator.free(self.allocation).unwrap();
        unsafe { vk.device.destroy_buffer(self.buffer, MEMORY_ALLOCATOR); }
    }

    pub fn upload_buffer<T>(&self, vk: &mut VulkanAPI, in_buffer: &[T]) {
        self.upload_subbuffer_elements(vk, in_buffer, 0);
    }

    pub fn upload_subbuffer_elements<T>(&self, vk: &mut VulkanAPI, in_buffer: &[T], offset: u64) {
        let byte_buffer = unsafe { slice_to_bytes(in_buffer) };
        let byte_offset = offset * size_of::<T>() as u64;
        self.upload_subbuffer_bytes(vk, byte_buffer, byte_offset as u64);
    }

    pub fn upload_subbuffer_bytes(&self, vk: &mut VulkanAPI, in_buffer: &[u8], offset: u64) {
        let end_in_bytes = in_buffer.len() + offset as usize;
        if end_in_bytes as u64 > self.length {
            crash_with_error_dialog("OVERRAN BUFFER AAAAA");
        }

        unsafe {
            match self.allocation.mapped_ptr() {
                Some(p) => {
                    let dst_ptr = p.as_ptr() as *mut u8;
                    let dst_ptr = dst_ptr.offset(offset as isize);
                    ptr::copy_nonoverlapping(in_buffer.as_ptr(), dst_ptr as *mut u8, in_buffer.len());
                }
                None => {
                    upload_GPU_buffer(vk, self.buffer, offset, in_buffer);
                }
            }
        }
    }
}

pub struct VertexInputConfiguration {
    pub binding_descriptions: Vec<vk::VertexInputBindingDescription>,
    pub attribute_descriptions: Vec<vk::VertexInputAttributeDescription>    
}

impl VertexInputConfiguration {
    pub fn empty() -> Self {
        VertexInputConfiguration {
            binding_descriptions: Vec::new(),
            attribute_descriptions: Vec::new()
        }
    }
}

pub struct PipelineCreateInfo {
    pipeline_layout: vk::PipelineLayout,
    vertex_input: VertexInputConfiguration,
    input_assembly: vk::PipelineInputAssemblyStateCreateInfo,
    rasterization_state: vk::PipelineRasterizationStateCreateInfo,
    multisample_state: vk::PipelineMultisampleStateCreateInfo,
    dynamic_state: Vec<vk::DynamicState>,
    viewport_state: vk::PipelineViewportStateCreateInfo,
    depthstencil_state: vk::PipelineDepthStencilStateCreateInfo,
    color_blend_attachment_state: vk::PipelineColorBlendAttachmentState,
    shader_stages: Vec<vk::PipelineShaderStageCreateInfo>,
    render_pass: vk::RenderPass
}

#[derive(Default)]
pub struct GraphicsPipelineBuilder {
    pub dynamic_state_enables: [vk::DynamicState; 2],
    pub input_assembly_state: vk::PipelineInputAssemblyStateCreateInfo,
    pub rasterization_state: vk::PipelineRasterizationStateCreateInfo,
    pub color_blend_attachment_state: vk::PipelineColorBlendAttachmentState,
    pub viewport_state: vk::PipelineViewportStateCreateInfo,
    pub depthstencil_state: vk::PipelineDepthStencilStateCreateInfo,
    pub multisample_state: vk::PipelineMultisampleStateCreateInfo,
    pub pipeline_layout: vk::PipelineLayout,
    shader_stages: Vec<vk::PipelineShaderStageCreateInfo>,
    render_pass: vk::RenderPass
}

impl GraphicsPipelineBuilder {
    pub fn new() -> Self {
        let dynamic_state_enables = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];

        let input_assembly_state = vk::PipelineInputAssemblyStateCreateInfo {
            topology: vk::PrimitiveTopology::TRIANGLE_LIST,
            ..Default::default()
        };

        let rasterization_state = vk::PipelineRasterizationStateCreateInfo {
            polygon_mode: vk::PolygonMode::FILL,
            cull_mode: vk::CullModeFlags::BACK,
            front_face: vk::FrontFace::COUNTER_CLOCKWISE,
            depth_clamp_enable: vk::FALSE,
            rasterizer_discard_enable: vk::FALSE,
            depth_bias_enable: vk::TRUE,
            line_width: 1.0,
            ..Default::default()
        };

        let color_blend_attachment_state = vk::PipelineColorBlendAttachmentState {
            color_write_mask: vk::ColorComponentFlags::from_raw(0xF),   //All components
            blend_enable: vk::TRUE,
            alpha_blend_op: vk::BlendOp::ADD,
            color_blend_op: vk::BlendOp::ADD,
            src_color_blend_factor: vk::BlendFactor::SRC_ALPHA,
            dst_color_blend_factor: vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
            src_alpha_blend_factor: vk::BlendFactor::SRC_ALPHA,
            dst_alpha_blend_factor: vk::BlendFactor::ONE_MINUS_SRC_ALPHA
        };

        let viewport_state = vk::PipelineViewportStateCreateInfo {
            viewport_count: 1,
            scissor_count: 1,
            p_scissors: ptr::null(),
            p_viewports: ptr::null(),
            ..Default::default()
        };

        let depth_stencil_state = vk::PipelineDepthStencilStateCreateInfo {
            depth_test_enable: vk::TRUE,
            depth_write_enable: vk::TRUE,
            depth_compare_op: vk::CompareOp::LESS_OR_EQUAL,
            depth_bounds_test_enable: vk::FALSE,
            stencil_test_enable: vk::FALSE,
            ..Default::default()
        };

        let multisample_state = vk::PipelineMultisampleStateCreateInfo {
            rasterization_samples: vk::SampleCountFlags::TYPE_1,
            ..Default::default()
        };

        GraphicsPipelineBuilder {
            dynamic_state_enables,
            input_assembly_state,
            rasterization_state,
            color_blend_attachment_state,
            depthstencil_state: depth_stencil_state,
            multisample_state,
            viewport_state,
            ..Default::default()
        }
    }

    pub fn init(render_pass: vk::RenderPass, pipeline_layout: vk::PipelineLayout) -> Self {
        let mut tmp = Self::new();
        tmp.pipeline_layout = pipeline_layout;
        tmp.render_pass = render_pass;
        tmp
    }

    pub fn build_info(self) -> PipelineCreateInfo {
        let vertex_input = VertexInputConfiguration::empty();

        PipelineCreateInfo {
            pipeline_layout: self.pipeline_layout,
            vertex_input,
            input_assembly: self.input_assembly_state,
            rasterization_state: self.rasterization_state,
            multisample_state: self.multisample_state,
            dynamic_state: Vec::from(self.dynamic_state_enables),
            viewport_state: self.viewport_state,
            depthstencil_state: self.depthstencil_state,
            color_blend_attachment_state: self.color_blend_attachment_state,
            shader_stages: self.shader_stages.clone(),
            render_pass: self.render_pass
        }
    }

    pub fn set_cull_mode(self, flags: vk::CullModeFlags) -> Self {
        let mut t = self;
        t.rasterization_state.cull_mode = flags;
        t
    }

    pub fn set_front_face(self, front_face: vk::FrontFace) -> Self {
        let mut t = self;
        t.rasterization_state.front_face = front_face;
        t
    }

    pub fn set_depth_test(self, set_it: vk::Bool32) -> Self {
        let mut t = self;
        t.depthstencil_state.depth_test_enable = set_it;
        t
    }

    pub fn set_shader_stages(self, stages: Vec<vk::PipelineShaderStageCreateInfo>) -> Self {
        GraphicsPipelineBuilder {
            shader_stages: stages,
            ..self
        }
    }

    pub unsafe fn create_pipelines(vk: &mut VulkanAPI, infos: &[PipelineCreateInfo]) -> Vec<vk::Pipeline> {
        let mut vertex_input_configs = Vec::with_capacity(infos.len());
        let mut vertex_input_states = Vec::with_capacity(infos.len());
        let mut dynamic_states = Vec::with_capacity(infos.len());
        let mut color_blend_attachment_states = Vec::with_capacity(infos.len());
        let mut real_create_infos = Vec::with_capacity(infos.len());
        for i in 0..infos.len() {
            let info = &infos[i];

            vertex_input_configs.push(&info.vertex_input);
            let vertex_input_state = vk::PipelineVertexInputStateCreateInfo {
                vertex_binding_description_count: vertex_input_configs[i].binding_descriptions.len() as u32,
                p_vertex_binding_descriptions: vertex_input_configs[i].binding_descriptions.as_ptr(),
                vertex_attribute_description_count: vertex_input_configs[i].attribute_descriptions.len() as u32,
                p_vertex_attribute_descriptions: vertex_input_configs[i].attribute_descriptions.as_ptr(),
                ..Default::default()
            };
            vertex_input_states.push(vertex_input_state);

            let dynamic_state = vk::PipelineDynamicStateCreateInfo {
                p_dynamic_states: info.dynamic_state.as_ptr(),
                dynamic_state_count: info.dynamic_state.len() as u32,
                ..Default::default()
            };
            dynamic_states.push(dynamic_state);

            let color_blend = vk::PipelineColorBlendStateCreateInfo {
                attachment_count: 1,
                p_attachments: &info.color_blend_attachment_state,
                logic_op_enable: vk::FALSE,
                logic_op: vk::LogicOp::NO_OP,
                blend_constants: [0.0; 4],
                ..Default::default()
            };
            color_blend_attachment_states.push(color_blend);

            let create_info = vk::GraphicsPipelineCreateInfo {
                layout: info.pipeline_layout,
                p_vertex_input_state: &vertex_input_states[i],
                p_input_assembly_state: &info.input_assembly,
                p_rasterization_state: &info.rasterization_state,
                p_color_blend_state: &color_blend_attachment_states[i],
                p_multisample_state: &info.multisample_state,
                p_dynamic_state: &dynamic_states[i],
                p_viewport_state: &info.viewport_state,
                p_depth_stencil_state: &info.depthstencil_state,
                p_stages: info.shader_stages.as_ptr(),
                stage_count: info.shader_stages.len() as u32,
                render_pass: info.render_pass,
                ..Default::default()
            };
            real_create_infos.push(create_info);
        }

        vk.device.create_graphics_pipelines(vk::PipelineCache::null(), &real_create_infos, MEMORY_ALLOCATOR).unwrap()
    }

}

#[derive(Debug)]
pub struct FreeList<T> {
    list: OptionVec<T>,
    size: u64,
    pub updated: bool
}

impl<T> FreeList<T> {
    pub fn with_capacity(size: usize) -> Self {
        FreeList {
            list: OptionVec::with_capacity(size),
            size: size as u64,
            updated: false
        }
    }

    pub fn len(&self) -> usize { self.list.len() }

    pub fn size(&self) -> u64 { self.size }

    pub fn insert(&mut self, item: T) -> usize {
        self.updated = true;
        self.list.insert(item)
    }

    pub fn remove(&mut self, idx: usize) {
        self.updated = true;
        self.list.delete(idx);
    }
}

impl<T> Index<usize> for FreeList<T> {
    type Output = Option<T>;

    fn index(&self, idx: usize) -> &Self::Output {
        &self.list[idx]
    }
}
