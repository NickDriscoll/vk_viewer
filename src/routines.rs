use ozy::structs::UninterleavedVertexArrays;
use crate::render::{Model, PrimitiveKey};
use crate::vkutil::{DeferredImage, VertexFetchOffsets};
use crate::*;

//Converts any data structure to a slice of bytes
#[inline]
pub fn struct_to_bytes<'a, T>(structure: &'a T) -> &'a [u8] {
    unsafe { std::slice::from_raw_parts(structure as *const _ as *const u8, size_of::<T>()) }
}

#[inline]
pub fn slice_to_bytes<'a, T>(in_array: &'a [T]) -> &'a [u8] {
    unsafe { core::slice::from_raw_parts(in_array.as_ptr() as *const u8, in_array.len() * size_of::<T>()) }
}

#[inline]
pub fn calculate_miplevels(width: u32, height: u32) -> u32 {
    (f32::floor(f32::log2(u32::max(width, height) as f32))) as u32 + 1
}

pub fn crash_with_error_dialog(message: &str) -> ! {
    crash_with_error_dialog_titled("Oops...", message);
}

pub fn crash_with_error_dialog_titled(title: &str, message: &str) -> ! {
    tfd::message_box_ok(title, &message.replace("'", ""), tfd::MessageBoxIcon::Error);
    panic!("{}", message);
}

pub fn unwrap_result<T, E: Display>(res: Result<T, E>, msg: &str) -> T {
    match res {
        Ok(t) => { t }
        Err(e) => {
            crash_with_error_dialog(&format!("{}\n{}", msg, e));
        }
    }
}

pub fn unix_epoch_ms() -> u128 {
    SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis()
}

pub fn compute_terrain_vertices(spec: &mut TerrainSpec, fixed_seed: bool, scale: f32) -> UninterleavedVertexArrays {
    if !fixed_seed {
        spec.seed = unix_epoch_ms();
    }
    spec.generate_vertices(scale)
}

pub fn regenerate_terrain(
    vk: &mut VulkanAPI,
    renderer: &mut Renderer,
    physics_engine: &mut PhysicsEngine,
    terrain_collider_handle: &mut ColliderHandle,
    primitive_key: PrimitiveKey,
    terrain: &mut TerrainSpec,
    terrain_generation_scale: f32
) {
    if let Some(ter) = renderer.get_primitive(primitive_key) {
        let offset = ter.position_offset;
        let verts = compute_terrain_vertices(terrain, terrain.fixed_seed, terrain_generation_scale);
        replace_uploaded_vertices(vk, renderer, &verts, offset.into());

        physics_engine.collider_set.remove(*terrain_collider_handle, &mut physics_engine.island_manager, &mut physics_engine.rigid_body_set, false);

        *terrain_collider_handle = physics_engine.make_terrain_collider(&verts.positions, terrain.vertex_width, terrain.vertex_height);
    }
}

pub fn upload_vertex_attributes(vk: &mut VulkanAPI, renderer: &mut Renderer, attribs: &UninterleavedVertexArrays) -> VertexFetchOffsets {
    let position_offset = renderer.append_vertex_positions(vk, &attribs.positions);
    let tangent_offset = renderer.append_vertex_tangents(vk, &attribs.tangents);
    let normal_offset = renderer.append_vertex_normals(vk, &attribs.normals);
    let uv_offset = renderer.append_vertex_uvs(vk, &attribs.uvs);

    VertexFetchOffsets {
        position_offset,
        tangent_offset,
        normal_offset,
        uv_offset
    }
}

pub fn upload_primitive_vertices(vk: &mut VulkanAPI, renderer: &mut Renderer, prim: &GLTFPrimitive) -> VertexFetchOffsets {
    let position_offset = renderer.append_vertex_positions(vk, &prim.vertex_positions);
    let tangent_offset = renderer.append_vertex_tangents(vk, &prim.vertex_tangents);
    let normal_offset = renderer.append_vertex_normals(vk, &prim.vertex_normals);
    let uv_offset = renderer.append_vertex_uvs(vk, &prim.vertex_uvs);

    VertexFetchOffsets {
        position_offset,
        tangent_offset,
        normal_offset,
        uv_offset
    }
}

pub fn replace_uploaded_vertices(vk: &mut VulkanAPI, renderer: &mut Renderer, attributes: &UninterleavedVertexArrays, offset: u64) {
    renderer.replace_vertex_positions(vk, &attributes.positions, offset);
    renderer.replace_vertex_tangents(vk, &attributes.tangents, offset);
    renderer.replace_vertex_normals(vk, &attributes.normals, offset);
    renderer.replace_vertex_uvs(vk, &attributes.uvs, offset);
}

pub fn reset_totoro(physics_engine: &mut PhysicsEngine, totoro: &Option<PhysicsProp>) {
    let handle = totoro.as_ref().unwrap().rigid_body_handle;
    if let Some(body) = physics_engine.rigid_body_set.get_mut(handle) {
        body.set_linvel(glm::zero(), true);
        body.set_position(Isometry::from_parts(Translation::new(0.0, 0.0, 20.0), *body.rotation()), true);
    }
}

pub unsafe fn upload_image_deferred(vk: &mut VulkanAPI, image_create_info: &vk::ImageCreateInfo, sampler: vk::Sampler, layout: vk::ImageLayout, raw_bytes: &[u8]) -> DeferredImage {
    //Create staging buffer and upload raw image data
    let bytes_size = raw_bytes.len() as vk::DeviceSize;
    let staging_buffer = GPUBuffer::allocate(vk, bytes_size, 0, vk::BufferUsageFlags::TRANSFER_SRC, MemoryLocation::CpuToGpu);
    staging_buffer.write_buffer(vk, &raw_bytes);

    //Create image
    let image = vk.device.create_image(image_create_info, vkutil::MEMORY_ALLOCATOR).unwrap();
    let allocation = vkutil::allocate_image_memory(vk, image);
    let view_info = vk::ImageViewCreateInfo {
        image,
        format: image_create_info.format,
        view_type: vk::ImageViewType::TYPE_2D,
        components: vkutil::COMPONENT_MAPPING_DEFAULT,
        subresource_range: vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: image_create_info.mip_levels,
            base_array_layer: 0,
            layer_count: 1
        },
        ..Default::default()
    };
    let image_view = vk.device.create_image_view(&view_info, vkutil::MEMORY_ALLOCATOR).unwrap();
    let vim = GPUImage {
        image,
        view: image_view,
        width: image_create_info.extent.width,
        height: image_create_info.extent.height,
        mip_count: image_create_info.mip_levels,
        format: image_create_info.format,
        layout,
        sampler,
        allocation
    };

    let fence = vk.device.create_fence(&vk::FenceCreateInfo::default(), vkutil::MEMORY_ALLOCATOR).unwrap();
    let command_buffer_idx = vk.command_buffer_indices.insert(0);

    let command_buffer = vk.command_buffers[command_buffer_idx];
    vk.device.begin_command_buffer(command_buffer, &vk::CommandBufferBeginInfo::default()).unwrap();
    record_image_upload_commands(vk, command_buffer, &vim, layout, &staging_buffer);
    vk.device.end_command_buffer(command_buffer).unwrap();
    
    let submit_info = vk::SubmitInfo {
        command_buffer_count: 1,
        p_command_buffers: &command_buffer,
        ..Default::default()
    };
    let queue = vk.device.get_device_queue(vk.queue_family_index, 0);
    vk.device.queue_submit(queue, &[submit_info], fence).unwrap();

    DeferredImage {
        fence,
        staging_buffer: Some(staging_buffer),
        command_buffer_idx,
        final_image: vim
    }
}

//staging_buffer already has the image bytes uploaded to it
pub unsafe fn record_image_upload_commands(vk: &mut VulkanAPI, command_buffer: vk::CommandBuffer, gpu_image: &GPUImage, layout: vk::ImageLayout, staging_buffer: &GPUBuffer) {
    let image_memory_barrier = vk::ImageMemoryBarrier {
        src_access_mask: vk::AccessFlags::empty(),
        dst_access_mask: vk::AccessFlags::TRANSFER_WRITE,
        old_layout: vk::ImageLayout::UNDEFINED,
        new_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        image: gpu_image.image,
        subresource_range: vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: gpu_image.mip_count,
            base_array_layer: 0,
            layer_count: 1
        },
        ..Default::default()
    };
    vk.device.cmd_pipeline_barrier(command_buffer, vk::PipelineStageFlags::TOP_OF_PIPE, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[], &[], &[image_memory_barrier]);

    let mut cumulative_offset = 0;
    let mut copy_regions = vec![vk::BufferImageCopy::default(); gpu_image.mip_count as usize];
    for i in 0..gpu_image.mip_count {
        let w = gpu_image.width / (1 << i);
        let h = gpu_image.height / (1 << i);
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

    vk.device.cmd_copy_buffer_to_image(command_buffer, staging_buffer.backing_buffer(), gpu_image.image, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &copy_regions);

    let subresource_range = vk::ImageSubresourceRange {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        base_mip_level: 0,
        level_count: gpu_image.mip_count,
        base_array_layer: 0,
        layer_count: 1
    };
    let image_memory_barrier = vk::ImageMemoryBarrier {
        src_access_mask: vk::AccessFlags::TRANSFER_WRITE,
        dst_access_mask: vk::AccessFlags::SHADER_READ,
        old_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        new_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        image: gpu_image.image,
        subresource_range,
        ..Default::default()
    };
    vk.device.cmd_pipeline_barrier(command_buffer, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::FRAGMENT_SHADER, vk::DependencyFlags::empty(), &[], &[], &[image_memory_barrier]);

    //Generate mipmaps
    let image_memory_barrier = vk::ImageMemoryBarrier {
        src_access_mask: vk::AccessFlags::empty(),
        dst_access_mask: vk::AccessFlags::TRANSFER_WRITE,
        old_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        new_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        image: gpu_image.image,
        subresource_range: vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: gpu_image.mip_count,
            base_array_layer: 0,
            layer_count: 1
        },
        ..Default::default()
    };
    vk.device.cmd_pipeline_barrier(
        command_buffer,
        vk::PipelineStageFlags::TOP_OF_PIPE,
        vk::PipelineStageFlags::TRANSFER,
        vk::DependencyFlags::empty(),
        &[], &[],
        &[image_memory_barrier]
    );

    for i in 0..(gpu_image.mip_count - 1) {
        let src_mip_barrier = vk::ImageMemoryBarrier {
            src_access_mask: vk::AccessFlags::empty(),
            dst_access_mask: vk::AccessFlags::TRANSFER_WRITE,
            old_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            new_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            image: gpu_image.image,
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
            command_buffer,
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
            vk::Offset3D {x: (gpu_image.width >> i) as i32, y: (gpu_image.height >> i) as i32, z: 1}
        ];
        let dst_offsets = [
            vk::Offset3D {x: 0, y: 0, z: 0},
            vk::Offset3D {x: (gpu_image.width >> (i + 1)) as i32, y: (gpu_image.height >> (i + 1)) as i32, z: 1}
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
            command_buffer,
            gpu_image.image,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            gpu_image.image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &regions,
            vk::Filter::LINEAR
        );
    }

    let subresource_range = vk::ImageSubresourceRange {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        base_mip_level: 0,
        level_count: gpu_image.mip_count,
        base_array_layer: 0,
        layer_count: 1
    };
    let image_memory_barrier = vk::ImageMemoryBarrier {
        src_access_mask: vk::AccessFlags::TRANSFER_WRITE,
        dst_access_mask: vk::AccessFlags::SHADER_READ,
        old_layout: vk::ImageLayout::UNDEFINED,
        new_layout: layout,
        image: gpu_image.image,
        subresource_range,
        ..Default::default()
    };
    vk.device.cmd_pipeline_barrier(command_buffer, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::FRAGMENT_SHADER, vk::DependencyFlags::empty(), &[], &[], &[image_memory_barrier]);

}

pub unsafe fn upload_image(vk: &mut VulkanAPI, image: &GPUImage, raw_bytes: &[u8]) {
    //Create staging buffer and upload raw image data
    let bytes_size = raw_bytes.len() as vk::DeviceSize;
    let staging_buffer = GPUBuffer::allocate(vk, bytes_size, 0, vk::BufferUsageFlags::TRANSFER_SRC, MemoryLocation::CpuToGpu);
    staging_buffer.write_buffer(vk, &raw_bytes);

    //Wait on the fence before beginning command recording
    vk.device.wait_for_fences(&[vk.command_buffer_fence], true, vk::DeviceSize::MAX).unwrap();
    vk.device.reset_fences(&[vk.command_buffer_fence]).unwrap();
    vk.device.begin_command_buffer(vk.command_buffers[0], &vk::CommandBufferBeginInfo::default()).unwrap();

    let image_memory_barrier = vk::ImageMemoryBarrier {
        src_access_mask: vk::AccessFlags::empty(),
        dst_access_mask: vk::AccessFlags::TRANSFER_WRITE,
        old_layout: vk::ImageLayout::UNDEFINED,
        new_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        image: image.image,
        subresource_range: vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: image.mip_count,
            base_array_layer: 0,
            layer_count: 1
        },
        ..Default::default()
    };
    vk.device.cmd_pipeline_barrier(vk.command_buffers[0], vk::PipelineStageFlags::TOP_OF_PIPE, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[], &[], &[image_memory_barrier]);

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

    vk.device.cmd_copy_buffer_to_image(vk.command_buffers[0], staging_buffer.backing_buffer(), image.image, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &copy_regions);

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
        image: image.image,
        subresource_range,
        ..Default::default()
    };
    vk.device.cmd_pipeline_barrier(vk.command_buffers[0], vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::FRAGMENT_SHADER, vk::DependencyFlags::empty(), &[], &[], &[image_memory_barrier]);

    vk.device.end_command_buffer(vk.command_buffers[0]).unwrap();
    
    let submit_info = vk::SubmitInfo {
        command_buffer_count: 1,
        p_command_buffers: &vk.command_buffers[0],
        ..Default::default()
    };
    let queue = vk.device.get_device_queue(vk.queue_family_index, 0);
    vk.device.queue_submit(queue, &[submit_info], vk.command_buffer_fence).unwrap();
    vk.device.wait_for_fences(&[vk.command_buffer_fence], true, vk::DeviceSize::MAX).unwrap();
    staging_buffer.free(vk);
}

#[named]
pub fn compress_png_synchronous(vk: &mut VulkanAPI, path: &str) {
    unsafe {
        use ozy::io::{D3D10_RESOURCE_DIMENSION, DXGI_FORMAT, compute_pitch_bc};

        //Read png bytes out of file
        let mut file = unwrap_result(File::open(path), &format!("Error opening png with {}", function_name!()));
        let mut png_bytes = vec![0u8; file.metadata().unwrap().len().try_into().unwrap()];
        file.read_exact(&mut png_bytes).unwrap();

        //Extract metadata and decode to raw RGBA bytes
        let decoder = png::Decoder::new(png_bytes.as_slice());
        let read_info = decoder.read_info().unwrap();
        let info = read_info.info();
        let width = info.width;
        let height = info.height;
        let rgb_bitcount = info.bit_depth as u32;
        let dxgi_format = match info.srgb {
            Some(_) => { DXGI_FORMAT::BC7_UNORM_SRGB }
            None => { DXGI_FORMAT::BC7_UNORM }
        };
        let uncompressed_format = match info.srgb {
            Some(_) => { vk::Format::R8G8B8A8_SRGB }
            None => { vk::Format::R8G8B8A8_UNORM }
        };
        let bytes = vkutil::decode_png(read_info);

        //After decoding, upload to GPU for mipmap creation
        let mip_levels = calculate_miplevels(width, height);
        let image_create_info = vk::ImageCreateInfo {
            image_type: vk::ImageType::TYPE_2D,
            format: uncompressed_format,
            extent: vk::Extent3D {
                width,
                height,
                depth: 1
            },
            mip_levels,
            array_layers: 1,
            samples: vk::SampleCountFlags::TYPE_1,
            tiling: vk::ImageTiling::OPTIMAL,
            usage: vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            queue_family_index_count: 1,
            p_queue_family_indices: &vk.queue_family_index,
            initial_layout: vk::ImageLayout::UNDEFINED,
            ..Default::default()
        };

        let sampler = vk.device.create_sampler(&vk::SamplerCreateInfo::default(), vkutil::MEMORY_ALLOCATOR).unwrap();
        let gpu_image_layout = vk::ImageLayout::TRANSFER_SRC_OPTIMAL;
        let def_image = upload_image_deferred(vk, &image_create_info, sampler, gpu_image_layout, &bytes);
        let def_image = &DeferredImage::synchronize(vk, vec![def_image])[0];

        //Get the GPU image back into system RAM
        let finished_image_reqs = vk.device.get_image_memory_requirements(def_image.final_image.image);
        let readback_buffer = GPUBuffer::allocate(vk, finished_image_reqs.size, finished_image_reqs.alignment, vk::BufferUsageFlags::TRANSFER_DST, MemoryLocation::CpuToGpu);
        let cb_idx = vk.command_buffer_indices.insert(0);
        let command_buffer = vk.command_buffers[cb_idx];
        vk.device.begin_command_buffer(command_buffer, &vk::CommandBufferBeginInfo::default()).unwrap();

        let mut regions = Vec::with_capacity(mip_levels as usize);
        let mut current_offset = 0;
        for i in 0..mip_levels {
            let w = def_image.final_image.width / (1 << i);
            let h = def_image.final_image.height / (1 << i);
            let image_subresource = vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: i as u32,
                base_array_layer: 0,
                layer_count: 1
            };
            let copy = vk::BufferImageCopy {
                buffer_offset: current_offset,
                buffer_row_length: 0,
                buffer_image_height: 0,
                image_extent: vk::Extent3D {
                    width: w,
                    height: h,
                    depth: 1
                },
                image_subresource,
                image_offset: vk::Offset3D::default()
            };
            regions.push(copy);
            current_offset += (w * h * 4) as u64;
        }
        vk.device.cmd_copy_image_to_buffer(command_buffer, def_image.final_image.image, gpu_image_layout, readback_buffer.backing_buffer(), &regions);

        vk.device.end_command_buffer(command_buffer).unwrap();

        let submit_info = vk::SubmitInfo {
            command_buffer_count: 1,
            p_command_buffers: &command_buffer,
            ..Default::default()
        };
        let queue = vk.device.get_device_queue(vk.queue_family_index, 0);
        let fence = vk.device.create_fence(&vk::FenceCreateInfo::default(), vkutil::MEMORY_ALLOCATOR).unwrap();
        vk.device.queue_submit(queue, &[submit_info], fence).unwrap();
        vk.device.wait_for_fences(&[fence], true, vk::DeviceSize::MAX).unwrap();
        vk.command_buffer_indices.remove(cb_idx);

        let uncompressed_bytes = readback_buffer.read_buffer_bytes();
        let mut bc7_bytes = vec![0u8; uncompressed_bytes.len() / 4];  //BC7 images are one byte per pixel
        let mut uncompressed_byte_offset = 0;
        let mut bc7_offset = 0;
        for j in 0..mip_levels {
            let w2 = (def_image.final_image.width / (1 << j)) as usize;
            let h2 = (def_image.final_image.height / (1 << j)) as usize;
            let data = &uncompressed_bytes[uncompressed_byte_offset..(uncompressed_byte_offset + w2 * h2 * 4)];
            let surface = ispc::RgbaSurface {
                data,
                width: w2 as u32,
                height: h2 as u32,
                stride: 4 * w2 as u32
            };
            let settings = ispc::bc7::opaque_slow_settings();
            let bc7_range = usize::max(w2 * h2, 16);    //BC7 blocks are 16 bytes at minimum
            ispc::bc7::compress_blocks_into(&settings, &surface, &mut bc7_bytes[bc7_offset..(bc7_offset + bc7_range)]);

            uncompressed_byte_offset += w2 * h2 * 4;
            bc7_offset += bc7_range;
        }

        let dds_pixelformat = DDS_PixelFormat {
            rgb_bitcount,
            flags: DDS_PixelFormat::DDPF_FOURCC,        //We just always wanna use the 
            ..Default::default()
        };
        let dx10_header = DDSHeader_DXT10 {
            dxgi_format,
            resource_dimension: D3D10_RESOURCE_DIMENSION::TEXTURE2D,
            array_size: 1,
            ..Default::default()
        };
        let dds_header = DDSHeader {
            flags: DDSHeader::DDSD_CAPS | DDSHeader::DDSD_WIDTH | DDSHeader::DDSD_HEIGHT | DDSHeader::DDSD_PIXELFORMAT | DDSHeader::DDSD_LINEARSIZE,
            height,
            width,
            pitch_or_linear_size: compute_pitch_bc(width, 16),
            mipmap_count: mip_levels,
            spf: dds_pixelformat,
            dx10_header,
            ..Default::default()
        };

        let path = Path::new(path);
        let out_path = format!("{}/{}.dds", path.parent().unwrap().display(), path.file_stem().unwrap().to_str().unwrap());
        let mut out_file = OpenOptions::new().write(true).create(true).open(out_path).unwrap();
        out_file.write(struct_to_bytes(&dds_header)).unwrap();
        out_file.write(&bc7_bytes).unwrap();

        vk.device.destroy_sampler(sampler, vkutil::MEMORY_ALLOCATOR);
    }
}


pub fn make_index_buffer(vk: &mut VulkanAPI, indices: &[u32]) -> GPUBuffer {
    let index_buffer = GPUBuffer::allocate(
        vk,
        (indices.len() * size_of::<u32>()) as vk::DeviceSize,
        0,
        vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
        MemoryLocation::GpuOnly
    );
    index_buffer.write_buffer(vk, indices);
    index_buffer
}
