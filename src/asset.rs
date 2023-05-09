use gltf::{Gltf, Mesh};
use gltf::accessor::DataType;
use ozy::io::{OzyMaterial, OzyPrimitive, OzyImage, UninterleavedVertexData};
use ozy::render::PositionNormalTangentUvPrimitive;
use std::io::{BufWriter};
use std::ptr;
use render::vkdevice::*;
use crate::*;

use crate::routines::*;

//Returns the uncompressed bytes of a png image to RGBA
pub fn decode_png<R: Read>(mut reader: png::Reader<R>) -> Vec<u8> {
    use png::BitDepth;
    use png::ColorType;

    let info = reader.info().clone();

    match info.bit_depth {
        BitDepth::Eight => { 1 }
        _ => { crash_with_error_dialog(&format!("Unsupported PNG bitdepth\n: {:?}", info.bit_depth)); }
    };
    
    //We shift width*height to the left by 1, then shift right by the result of this match
    // let byte_size_shift = match info.bit_depth {
    //     BitDepth::One => { 4 }
    //     BitDepth::Two => { 3 }
    //     BitDepth::Four => { 2 }
    //     BitDepth::Eight => { 1 }
    //     BitDepth::Sixteen => { 0 }
    // };

    let width = info.width;
    let height = info.height;
    let pixel_count = (width * height) as usize;
    match info.color_type {
        ColorType::Rgb => {
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
            let mut bytes = vec![0xFFu8; 4 * pixel_count];
            reader.next_frame(&mut bytes).unwrap();
            bytes
        }
        t => { crash_with_error_dialog(&format!("Unsupported color type: {:?}", t)); }
    }
}

pub fn decode_jpg<R: Read>(decoder: &mut jpg::Decoder<R>) -> Vec<u8> {
    let raw_bytes = decoder.decode().expect("Unable to decode jpg");
    let info = decoder.info().unwrap();
    let pixel_count = info.width as usize * info.height as usize;
    let mut bytes = vec![0xFF; pixel_count * 4];
    match info.pixel_format {
        jpg::PixelFormat::RGB24 => {
            for i in 0..pixel_count {
                let idx = 4 * i;
                let r_idx = 3 * i;
                bytes[idx] = raw_bytes[r_idx];
                bytes[idx + 1] = raw_bytes[r_idx + 1];
                bytes[idx + 2] = raw_bytes[r_idx + 2];
            }
        }
        _ => { crash_with_error_dialog(&format!("Unsupported JPG color format: {:?}", info.pixel_format)) }
    }

    bytes
}

pub fn load_bc7_info(gpu: &mut VulkanGraphicsDevice, path: &str) -> (vk::ImageCreateInfo, Vec<u8>) {
    let dds_path = path;
    let mut file = unwrap_result(File::open(dds_path), &format!("Error opening bc7 {}", dds_path));
    let dds_header = DDSHeader::from_file(&mut file);       //This also advances the file read head to the beginning of the raw data section

    let width = dds_header.width;
    let height = dds_header.height;
    let mip_levels = dds_header.mipmap_count;
    
    let mut bytes_size = 0;
    for i in 0..mip_levels {
        let w = width / (1 << i);
        let h = height / (1 << i);
        bytes_size += w * h;

        bytes_size = size_to_alignment!(bytes_size, 16);
    }

    let mut raw_bytes = vec![0u8; bytes_size as usize];
    file.read_exact(&mut raw_bytes).unwrap();
    
    let format = match dds_header.dx10_header.dxgi_format {
        DXGI_FORMAT::BC7_UNORM => { vk::Format::BC7_UNORM_BLOCK }
        DXGI_FORMAT::BC7_UNORM_SRGB => { vk::Format::BC7_SRGB_BLOCK }
        _ => { crash_with_error_dialog("Unreachable statement reached in bc7_create_info()"); }
    };
    let info = vk::ImageCreateInfo {
        image_type: vk::ImageType::TYPE_2D,
        format,
        mip_levels,
        array_layers: 1,
        samples: vk::SampleCountFlags::TYPE_1,
        tiling: vk::ImageTiling::OPTIMAL,
        sharing_mode: vk::SharingMode::EXCLUSIVE,
        queue_family_index_count: 1,
        p_queue_family_indices: &gpu.main_queue_family_index,
        usage: vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED,
        initial_layout: vk::ImageLayout::UNDEFINED,
        extent: vk::Extent3D {
            width,
            height,
            depth: 1
        },
        ..Default::default()
    };

    (info, raw_bytes)
}

pub unsafe fn upload_image_deferred(gpu: &mut VulkanGraphicsDevice, image_create_info: &vk::ImageCreateInfo, sampler_key: SamplerKey, layout: vk::ImageLayout, generate_mipmaps: bool, raw_bytes: &[u8]) -> DeferredImage {
    //Create staging buffer and upload raw image data
    let bytes_size = raw_bytes.len() as vk::DeviceSize;
    let staging_buffer = gpu.allocate_buffer(bytes_size, 0, vk::BufferUsageFlags::TRANSFER_SRC, MemoryLocation::CpuToGpu);     //TODO: Here and everywhere use a unifed staging buffer managed by the VulkanGraphicsDevice
    staging_buffer.write_buffer(gpu, &raw_bytes);

    //Create image
    let sampler = gpu.get_sampler(sampler_key).unwrap();
    let image = gpu.device.create_image(image_create_info, vkdevice::MEMORY_ALLOCATOR).unwrap();
    let allocation = vkdevice::allocate_image_memory(gpu, image);
    let vim = GPUImage {
        image,
        view: None,
        width: image_create_info.extent.width,
        height: image_create_info.extent.height,
        mip_count: image_create_info.mip_levels,
        format: image_create_info.format,
        layout,
        usage: image_create_info.usage,
        sampler,
        allocation
    };

    let fence = gpu.device.create_fence(&vk::FenceCreateInfo::default(), vkdevice::MEMORY_ALLOCATOR).unwrap();
    let command_buffer_idx = gpu.command_buffer_indices.insert(0);

    let command_buffer = gpu.command_buffers[command_buffer_idx];
    gpu.device.begin_command_buffer(command_buffer, &vk::CommandBufferBeginInfo::default()).unwrap();
    let image_memory_barrier = vk::ImageMemoryBarrier {
        src_access_mask: vk::AccessFlags::empty(),
        dst_access_mask: vk::AccessFlags::TRANSFER_WRITE,
        old_layout: vk::ImageLayout::UNDEFINED,
        new_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        image: vim.image,
        subresource_range: vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: vim.mip_count,
            base_array_layer: 0,
            layer_count: 1
        },
        ..Default::default()
    };
    gpu.device.cmd_pipeline_barrier(command_buffer, vk::PipelineStageFlags::TOP_OF_PIPE, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[], &[], &[image_memory_barrier]);

    let mut cumulative_offset = 0;
    let mut copy_regions = vec![vk::BufferImageCopy::default(); vim.mip_count as usize];
    for i in 0..vim.mip_count {
        let (w, h) = ozy::routines::mip_resolution(vim.width, vim.height, i);
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

    gpu.device.cmd_copy_buffer_to_image(command_buffer, staging_buffer.buffer(), vim.image, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &copy_regions);

    if generate_mipmaps {
        //Barrier to ensure the copy is finished before mipmap generation
        let image_memory_barrier = vk::ImageMemoryBarrier {
            src_access_mask: vk::AccessFlags::empty(),
            dst_access_mask: vk::AccessFlags::TRANSFER_WRITE,
            old_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            new_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            image: vim.image,
            subresource_range: vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: vim.mip_count,
                base_array_layer: 0,
                layer_count: 1
            },
            ..Default::default()
        };
        gpu.device.cmd_pipeline_barrier(
            command_buffer,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[], &[],
            &[image_memory_barrier]
        );
        
        //Generate mipmaps
        for i in 0..(vim.mip_count - 1) {
            let src_mip_barrier = vk::ImageMemoryBarrier {
                src_access_mask: vk::AccessFlags::empty(),
                dst_access_mask: vk::AccessFlags::TRANSFER_WRITE,
                old_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                new_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                image: vim.image,
                subresource_range: vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: i,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1
                },
                ..Default::default()
            };
            gpu.device.cmd_pipeline_barrier(
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
                vk::Offset3D {x: (vim.width >> i) as i32, y: (vim.height >> i) as i32, z: 1}
            ];
            let dst_offsets = [
                vk::Offset3D {x: 0, y: 0, z: 0},
                vk::Offset3D {x: (vim.width >> (i + 1)) as i32, y: (vim.height >> (i + 1)) as i32, z: 1}
            ];
            let regions = [
                vk::ImageBlit {
                    src_subresource,
                    src_offsets,
                    dst_subresource,
                    dst_offsets
                }
            ];
            gpu.device.cmd_blit_image(
                command_buffer,
                vim.image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                vim.image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &regions,
                vk::Filter::LINEAR
            );
        }

        let subresource_range = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: vim.mip_count,
            base_array_layer: 0,
            layer_count: 1
        };
        let image_memory_barrier = vk::ImageMemoryBarrier {
            src_access_mask: vk::AccessFlags::TRANSFER_WRITE,
            dst_access_mask: vk::AccessFlags::SHADER_READ,
            old_layout: vk::ImageLayout::UNDEFINED,
            new_layout: layout,
            image: vim.image,
            subresource_range,
            ..Default::default()
        };
        gpu.device.cmd_pipeline_barrier(command_buffer, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::FRAGMENT_SHADER, vk::DependencyFlags::empty(), &[], &[], &[image_memory_barrier]);
    }

    gpu.device.end_command_buffer(command_buffer).unwrap();
    
    let submit_info = vk::SubmitInfo {
        command_buffer_count: 1,
        p_command_buffers: &command_buffer,
        ..Default::default()
    };
    let queue = gpu.device.get_device_queue(gpu.main_queue_family_index, 0);
    gpu.device.queue_submit(queue, &[submit_info], fence).unwrap();

    DeferredImage {
        fence,
        staging_buffer: Some(staging_buffer),
        command_buffer_idx,
        gpu_image: vim
    }
}

pub unsafe fn upload_image(gpu: &mut VulkanGraphicsDevice, image: &GPUImage, raw_bytes: &[u8]) {
    //Create staging buffer and upload raw image data
    let bytes_size = raw_bytes.len() as vk::DeviceSize;
    let staging_buffer = gpu.allocate_buffer(bytes_size, 0, vk::BufferUsageFlags::TRANSFER_SRC, MemoryLocation::CpuToGpu);
    staging_buffer.write_buffer(gpu, &raw_bytes);

    //Wait on the fence before beginning command recording
    let cbidx = gpu.command_buffer_indices.insert(0);
    let cb_fence = gpu.command_buffer_fences[cbidx];
    gpu.device.wait_for_fences(&[cb_fence], true, vk::DeviceSize::MAX).unwrap();
    gpu.device.reset_fences(&[cb_fence]).unwrap();
    gpu.device.begin_command_buffer(gpu.command_buffers[cbidx], &vk::CommandBufferBeginInfo::default()).unwrap();

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
    gpu.device.cmd_pipeline_barrier(gpu.command_buffers[cbidx], vk::PipelineStageFlags::TOP_OF_PIPE, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[], &[], &[image_memory_barrier]);

    let mut cumulative_offset = 0;
    let mut copy_regions = vec![vk::BufferImageCopy::default(); image.mip_count as usize];
    for i in 0..image.mip_count {
        let (w, h) = ozy::routines::mip_resolution(image.width, image.height, i);
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

    gpu.device.cmd_copy_buffer_to_image(gpu.command_buffers[cbidx], staging_buffer.buffer(), image.image, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &copy_regions);

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
    gpu.device.cmd_pipeline_barrier(gpu.command_buffers[cbidx], vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::FRAGMENT_SHADER, vk::DependencyFlags::empty(), &[], &[], &[image_memory_barrier]);

    gpu.device.end_command_buffer(gpu.command_buffers[cbidx]).unwrap();
    
    let submit_info = vk::SubmitInfo {
        command_buffer_count: 1,
        p_command_buffers: &gpu.command_buffers[cbidx],
        ..Default::default()
    };
    let queue = gpu.device.get_device_queue(gpu.main_queue_family_index, 0);
    gpu.device.queue_submit(queue, &[submit_info], cb_fence).unwrap();
    gpu.device.wait_for_fences(&[cb_fence], true, vk::DeviceSize::MAX).unwrap();
    gpu.free_buffers(vec![staging_buffer]);
}

pub fn raw2bc7_synchronous(gpu: &mut VulkanGraphicsDevice, raw_bytes: &[u8], width: u32, height: u32, format: vk::Format) -> Vec<u8> {
    let mip_levels = ozy::routines::calculate_mipcount(width, height).saturating_sub(2).clamp(1, u32::MAX);
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
        usage: vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST,
        sharing_mode: vk::SharingMode::EXCLUSIVE,
        queue_family_index_count: 1,
        p_queue_family_indices: &gpu.main_queue_family_index,
        initial_layout: vk::ImageLayout::UNDEFINED,
        ..Default::default()
    };
    
    unsafe {
        let gpu_image_layout = vk::ImageLayout::TRANSFER_SRC_OPTIMAL;
        let sampler = gpu.create_sampler(&vk::SamplerCreateInfo::default()).unwrap();
        let def_image = upload_image_deferred(gpu, &image_create_info, sampler, gpu_image_layout, true, &raw_bytes);
        let def_images = DeferredImage::synchronize(gpu, vec![def_image]);
        let finished_image_reqs = gpu.device.get_image_memory_requirements(def_images[0].gpu_image.image);
        let readback_buffer = gpu.allocate_buffer(finished_image_reqs.size, finished_image_reqs.alignment, vk::BufferUsageFlags::TRANSFER_DST, MemoryLocation::GpuToCpu);
        
        let mut regions = Vec::with_capacity(mip_levels as usize);
        let mut current_offset = 0;
        for i in 0..mip_levels {
            let (w, h) = ozy::routines::mip_resolution(width, height, i);
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
        
        let cb_idx = gpu.command_buffer_indices.insert(0);
        let command_buffer = gpu.command_buffers[cb_idx];
        gpu.device.begin_command_buffer(command_buffer, &vk::CommandBufferBeginInfo::default()).unwrap();
        gpu.device.cmd_copy_image_to_buffer(command_buffer, def_images[0].gpu_image.image, gpu_image_layout, readback_buffer.buffer(), &regions);
        gpu.device.end_command_buffer(command_buffer).unwrap();

        let submit_info = vk::SubmitInfo {
            command_buffer_count: 1,
            p_command_buffers: &command_buffer,
            ..Default::default()
        };
        let queue = gpu.device.get_device_queue(gpu.main_queue_family_index, 0);
        let fence = gpu.device.create_fence(&vk::FenceCreateInfo::default(), vkdevice::MEMORY_ALLOCATOR).unwrap();
        gpu.device.queue_submit(queue, &[submit_info], fence).unwrap();
        gpu.device.wait_for_fences(&[fence], true, vk::DeviceSize::MAX).unwrap();
        for im in def_images {
            im.gpu_image.free(gpu);
        }
        gpu.command_buffer_indices.remove(cb_idx);
        gpu.destroy_sampler(sampler);

        let uncompressed_bytes = readback_buffer.read_buffer_bytes();

        gpu.free_buffers(vec![readback_buffer]);

        let bc7_output_size = {
            let mut total = 0;
            for i in 0..mip_levels {
                let (w, h) = ozy::routines::mip_resolution(width, height, i);
                total += ispc::bc7::calc_output_size(w, h);
            }
            total
        };
        let mut bc7_bytes = vec![0u8; bc7_output_size];
        let mut uncompressed_byte_offset = 0;
        let mut bc7_offset = 0;
        for j in 0..mip_levels {
            let (w2, h2) = ozy::routines::mip_resolution(width, height, j as u32);
            let w2 = w2 as usize;
            let h2 = h2 as usize;
            let mip_byte_stride = w2 * h2 * 4;
            let data = &uncompressed_bytes[uncompressed_byte_offset..(uncompressed_byte_offset + mip_byte_stride)];
            let surface = ispc::RgbaSurface {
                data,
                width: w2 as u32,
                height: h2 as u32,
                stride: 4 * w2 as u32
            };
            let settings = ispc::bc7::alpha_ultra_fast_settings();
            let bc7_range = ispc::bc7::calc_output_size(w2 as u32, h2 as u32);
            ispc::bc7::compress_blocks_into(&settings, &surface, &mut bc7_bytes[bc7_offset..(bc7_offset + bc7_range)]);

            uncompressed_byte_offset += mip_byte_stride;
            bc7_offset += bc7_range;
        }
        bc7_bytes
    }
}

pub fn png2bc7_synchronous(gpu: &mut VulkanGraphicsDevice, png_bytes: &[u8]) -> Vec<u8> {
    //Extract metadata and decode to raw RGBA bytes
    let decoder = png::Decoder::new(png_bytes);
    let read_info = decoder.read_info().unwrap();
    let info = read_info.info();
    let width = info.width;
    let height = info.height;
    //let rgb_bitcount = info.bit_depth as u32;
    let uncompressed_format = match info.srgb {
        Some(_) => { vk::Format::R8G8B8A8_SRGB }
        None => { vk::Format::R8G8B8A8_UNORM }
    };
    let bytes = decode_png(read_info);

    raw2bc7_synchronous(gpu, &bytes, width, height, uncompressed_format)
}

pub fn jpg2bc7_synchronous(gpu: &mut VulkanGraphicsDevice, jpg_bytes: &[u8], format: vk::Format) -> Vec<u8> {
    let mut decoder = jpg::Decoder::new(jpg_bytes);
    let raw_bytes = decoder.decode().unwrap();
    let info = decoder.info().unwrap();
    let width = info.width.into();
    let height = info.height.into();

    raw2bc7_synchronous(gpu, &raw_bytes, width, height, format)
}

#[named]
pub fn compress_png_file_synchronous(gpu: &mut VulkanGraphicsDevice, path: &str) {
    use ozy::io::{D3D10_RESOURCE_DIMENSION, DXGI_FORMAT, compute_pitch_bc};

    //Read png bytes out of file
    let mut file = unwrap_result(File::open(path), &format!("Error opening png with {}", function_name!()));
    let mut png_bytes = vec![0u8; file.metadata().unwrap().len().try_into().unwrap()];
    file.read_exact(&mut png_bytes).unwrap();
    let decoder = png::Decoder::new(png_bytes.as_slice());
    let read_info = decoder.read_info().unwrap();
    let info = read_info.info();
    let width = info.width;
    let height = info.height;
    let mip_levels = ozy::routines::calculate_mipcount(width, height).saturating_sub(2).clamp(1, u32::MAX);
    let rgb_bitcount = info.bit_depth as u32;
    let uncompressed_format = match info.srgb {
        Some(_) => { vk::Format::R8G8B8A8_SRGB }
        None => { vk::Format::R8G8B8A8_UNORM }
    };
    let dxgi_format = match uncompressed_format {
        vk::Format::R8G8B8A8_SRGB => { DXGI_FORMAT::BC7_UNORM_SRGB }
        vk::Format::R8G8B8A8_UNORM => { DXGI_FORMAT::BC7_UNORM }
        _ => { crash_with_error_dialog(&format!("Unreachable statement reached in {}", function_name!())); }
    };

    let bc7_bytes = png2bc7_synchronous(gpu, &png_bytes);

    let dds_pixelformat = DDS_PixelFormat {
        rgb_bitcount,
        flags: DDS_PixelFormat::DDPF_FOURCC,        //We just always wanna use this
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
}


pub struct RawImageData {
    pub color_index: Option<usize>,
    pub color_bytes: Vec<u8>,
    pub color_imagetype: GLTFImageType,
    pub normal_index: Option<usize>,
    pub normal_bytes: Vec<u8>,
    pub normal_imagetype: GLTFImageType,
    pub arm_index: Option<usize>,
    pub arm_bytes: Vec<u8>,
    pub arm_imagetype: GLTFImageType,
    pub emissive_index: Option<usize>,
    pub emissive_bytes: Vec<u8>,
    pub emissive_imagetype: GLTFImageType
}

#[derive(Debug)]
pub enum GLTFImageType {
    PNG,
    JPG,
}

#[derive(Debug)]
pub struct GLTFMaterial {
    pub base_color: [f32; 4],
    pub base_roughness: f32,
    pub base_metalness: f32,
    pub emissive_factor: [f32; 3],
    pub color_index: Option<usize>,
    pub color_imagetype: GLTFImageType,
    pub normal_index: Option<usize>,
    pub normal_imagetype: GLTFImageType,
    pub metallic_roughness_index: Option<usize>,
    pub metallic_roughness_imagetype: GLTFImageType,
    pub emissive_index: Option<usize>,
    pub emissive_imagetype: GLTFImageType
}

#[derive(Debug)]
pub struct GLTFPrimitive {
    pub vertex_positions: Vec<f32>,
    pub vertex_normals: Vec<f32>,
    pub vertex_tangents: Vec<f32>,
    pub vertex_uvs: Vec<f32>,
    pub indices: Vec<u32>,
    pub material: GLTFMaterial
}

impl PositionNormalTangentUvPrimitive for GLTFPrimitive {
    fn vertex_positions(&self) -> &[f32] {
        &self.vertex_positions
    }

    fn vertex_normals(&self) -> &[f32] {
        &self.vertex_normals
    }

    fn vertex_tangents(&self) -> &[f32] {
        &self.vertex_tangents
    }

    fn vertex_uvs(&self) -> &[f32] {
        &self.vertex_uvs
    }
}

#[derive(Debug)]
pub struct GLTFMeshData {
    pub name: String,
    pub primitives: Vec<GLTFPrimitive>,
    pub texture_bytes: Vec<Vec<u8>>,        //Indices correspond to the image index in the JSON
    pub children: Vec<usize>                //Indices correspond to the meshes in GLTFSceneData
}

#[derive(Debug)]
pub struct GLTFSceneData {
    pub name: Option<String>,
    pub meshes: Vec<GLTFMeshData>
}

fn image_bytes_from_source(glb: &Gltf, source: gltf::image::Source) -> (Vec<u8>, GLTFImageType) {
    let mut bytes;
    let image_type;
    match source {
        gltf::image::Source::View {view, mime_type} => unsafe {            
            match &glb.blob {
                Some(blob) => {
                    image_type = match mime_type {
                        "image/png" => { GLTFImageType::PNG }
                        "image/jpeg" => { GLTFImageType::JPG }
                        _ => { crash_with_error_dialog(&format!("Unsupported mime type: {}", mime_type)) }
                    };
                    
                    bytes = vec![0u8; view.length()];
                    let src_ptr = blob.as_ptr() as *const u8;
                    let src_ptr = src_ptr.offset(view.offset() as isize);
                    ptr::copy_nonoverlapping(src_ptr, bytes.as_mut_ptr(), view.length());
                }
                None => { crash_with_error_dialog("GLB had no blob"); }
            }
        }
        gltf::image::Source::Uri {..} => {
            crash_with_error_dialog("Uri not supported");
        }
    }
    (bytes, image_type)
}

fn uninterleaved_primitive_vertex_data(glb: &Gltf, prim: &gltf::Primitive) -> UninterleavedVertexData {
    use gltf::Semantic;
    //We always expect an index buffer to be present
    let indices = load_primitive_index_buffer(glb, &prim);
    //We always expect position data to be present
    let positions = match get_f32_semantic(&glb, &prim, Semantic::Positions) {
        Some(v) => {
            let mut out = vec![0.0; v.len() * 4 / 3];
            for i in (0..v.len()).step_by(3) {
                out[4 * i / 3] =     v[i];
                out[4 * i / 3 + 1] = v[i + 1];
                out[4 * i / 3 + 2] = v[i + 2];
                out[4 * i / 3 + 3] = 1.0;
            }
            out
        }
        None => {
            crash_with_error_dialog("GLTF primitive was missing position attribute.");
        }
    };
    let normals = match get_f32_semantic(&glb, &prim, Semantic::Normals) {
        Some(v) => {
            //Align to float4
            let mut out = vec![0.0; v.len() * 4 / 3];
            for i in (0..v.len()).step_by(3) {
                out[4 * i / 3] =     v[i];
                out[4 * i / 3 + 1] = v[i + 1];
                out[4 * i / 3 + 2] = v[i + 2];
                out[4 * i / 3 + 3] = 0.0;
            }
            out
        }
        None => { vec![0.0; positions.len()] }
    };

    let tangents = match get_f32_semantic(&glb, &prim, Semantic::Tangents) {
        Some(v) => { v }
        None => { vec![0.0; positions.len() / 3 * 4] }
    };
    let uvs = match get_f32_semantic(&glb, &prim, Semantic::TexCoords(0)) {
        Some(v) => { v }
        None => { vec![0.0; positions.len() / 3 * 2] }
    };

    UninterleavedVertexData {
        indices,
        positions,
        normals,
        tangents,
        uvs
    }
}

fn get_f32_semantic(glb: &Gltf, prim: &gltf::Primitive, semantic: gltf::Semantic) -> Option<Vec<f32>> {
    let acc = match prim.get(&semantic) {
        Some(a) => { a }
        None => { return None; }
    };
    let view = acc.view().unwrap();

    #[cfg(debug_assertions)]
    {
        let byte_stride = match view.stride() {
            Some(s) => { s }
            None => { 0 }
        };
        if byte_stride != 0 {
            crash_with_error_dialog("You are trying to load a glTF whose vertex attributes are not tightly packed, violating the assumptions of the loader");
        }
    }

    unsafe {
        match &glb.blob {
            Some(blob) => {
                let mut data = vec![0.0; view.length()/4];
                let src_ptr = blob.as_ptr() as *const u8;
                let src_ptr = src_ptr.offset(view.offset() as isize);
                ptr::copy_nonoverlapping(src_ptr, data.as_mut_ptr() as *mut u8, view.length());
                Some(data)
            }
            None => {
                crash_with_error_dialog("Invalid glb. No blob.");
            }
        }
    }
}

fn load_primitive_index_buffer(glb: &Gltf, prim: &gltf::Primitive) -> Vec<u32> {
    unsafe {
        let acc = prim.indices().unwrap();
        let view = acc.view().unwrap();
        match acc.data_type() {
            DataType::U16 => {
                let index_count = view.length() / 2;
                let mut index_buffer = vec![0u32; index_count];
                if let Some(blob) = &glb.blob {
                    for i in 0..index_count {
                        let current_idx = 2 * i + view.offset();
                        let bytes = [blob[current_idx], blob[current_idx + 1], 0, 0];
                        index_buffer[i] = u32::from_le_bytes(bytes);
                    }
                }
                index_buffer
            }
            DataType::U32 => {
                let index_count = view.length() / 4;
                let mut index_buffer = vec![0u32; index_count];
                if let Some(blob) = &glb.blob {
                    let src_ptr = blob.as_ptr() as *const u8;
                    let src_ptr = src_ptr.offset(view.offset() as isize);
                    ptr::copy_nonoverlapping(src_ptr, index_buffer.as_mut_ptr() as *mut u8, view.length());
                }
                index_buffer
            }
            _ => { crash_with_error_dialog(&format!("Unsupported index type: {:?}", acc.data_type())); }
        }
    }
}

pub fn optimize_glb(gpu: &mut VulkanGraphicsDevice, path: &str) {
    let glb = Gltf::open(path).unwrap();

    let parent_dir = Path::new(path).parent().unwrap();
    let name = Path::new(path).file_stem().unwrap().to_string_lossy();
    let output_location = format!("{}/optimized/{}.ozy", parent_dir.as_os_str().to_str().unwrap(), name);
    let mut textures = vec![OzyImage::default(); glb.images().count()];
    let mut materials = vec![OzyMaterial::default(); glb.materials().count()];
    let mut primitives = Vec::with_capacity(64);
    for mesh in glb.meshes() {
        for prim in mesh.primitives() {
            fn ozy_image_from_png(gpu: &mut VulkanGraphicsDevice, png_bytes: &[u8]) -> OzyImage {
                let decoder = png::Decoder::new(png_bytes).read_info().unwrap();
                let info = decoder.info();
                let width = info.width;
                let height = info.height;
                let mipmap_count = ozy::routines::calculate_mipcount(width, height).saturating_sub(2).max(1);
                let bc7_bytes = png2bc7_synchronous(gpu, png_bytes);
                OzyImage {
                    width,
                    height,
                    mipmap_count,
                    bc7_bytes
                }
            }

            fn ozy_image_from_jpg(gpu: &mut VulkanGraphicsDevice, jpg_bytes: &[u8], format: vk::Format) -> OzyImage {
                let mut decoder = jpg::Decoder::new(jpg_bytes);
                let bytes = decode_jpg(&mut decoder);
                let info = decoder.info().unwrap();
                let mipmap_count = ozy::routines::calculate_mipcount(info.width as u32, info.height as u32).saturating_sub(2).max(1);
                let bc7_bytes = raw2bc7_synchronous(gpu, &bytes, info.width.into(), info.height.into(), format);

                OzyImage {
                    width: info.width.into(),
                    height: info.height.into(),
                    mipmap_count,
                    bc7_bytes
                }
            }

            fn ozy_image_from_imagedata(gpu: &mut VulkanGraphicsDevice, idx: Option<usize>, imagetype: GLTFImageType, bytes: &[u8], bc7_idx: &mut Option<u32>, textures: &mut Vec<OzyImage>, format: vk::Format) {
                if let Some(idx) = idx {
                    *bc7_idx = Some(idx as u32);
                    if textures[idx].bc7_bytes.len() == 0 {
                        match imagetype {
                            GLTFImageType::PNG => {
                                textures[idx] = ozy_image_from_png(gpu, bytes);
                            }
                            GLTFImageType::JPG => {
                                textures[idx] = ozy_image_from_jpg(gpu, bytes, format);
                            }
                        }
                    }
                }
            }

            let vertex_data = uninterleaved_primitive_vertex_data(&glb, &prim);

            let mat = prim.material();
            
            println!("Mesh id: {}\nPrimitive index count: {}\nMaterial name: {:?}\n", mesh.index(), vertex_data.indices.len(), mat.name());

            let mat_idx = mat.index().unwrap();
            if materials[mat_idx].base_color[0] > 68.0 {
                let pbr = mat.pbr_metallic_roughness();

                let image_data = load_raw_image_data(&glb, &prim);
                let mut color_bc7_idx = None;
                let mut normal_bc7_idx = None;
                let mut arm_bc7_idx = None;
                let mut emissive_bc7_idx = None;

                ozy_image_from_imagedata(gpu, image_data.color_index, image_data.color_imagetype, &image_data.color_bytes, &mut color_bc7_idx, &mut textures, vk::Format::R8G8B8A8_SRGB);
                ozy_image_from_imagedata(gpu, image_data.normal_index, image_data.normal_imagetype, &image_data.normal_bytes, &mut normal_bc7_idx, &mut textures, vk::Format::R8G8B8A8_UNORM);
                ozy_image_from_imagedata(gpu, image_data.arm_index, image_data.arm_imagetype, &image_data.arm_bytes, &mut arm_bc7_idx, &mut textures, vk::Format::R8G8B8A8_UNORM);
                ozy_image_from_imagedata(gpu, image_data.emissive_index, image_data.emissive_imagetype, &image_data.emissive_bytes, &mut emissive_bc7_idx, &mut textures, vk::Format::R8G8B8A8_UNORM);

                let ozy_mat = OzyMaterial {
                    base_color: pbr.base_color_factor(),
                    emissive_factor: mat.emissive_factor(),
                    base_roughness: pbr.roughness_factor(),
                    base_metalness: pbr.metallic_factor(),
                    color_bc7_idx,
                    normal_bc7_idx,
                    arm_bc7_idx,
                    emissive_bc7_idx
                };
                materials[mat_idx] = ozy_mat;
            }

            let ozy_prim = OzyPrimitive {
                indices: vertex_data.indices,
                vertex_positions: vertex_data.positions,
                vertex_normals: vertex_data.normals,
                vertex_tangents: vertex_data.tangents,
                vertex_uvs: vertex_data.uvs,
                material_idx: mat_idx as u32
            };
            primitives.push(ozy_prim);
        }
    }

    //File header is gonna be material count + primitive count
    let dir = Path::new(&output_location).parent().unwrap();
    if !dir.exists() {
        std::fs::create_dir(dir).unwrap();
    }
    let mut output_file = BufWriter::new(OpenOptions::new().write(true).create(true).open(output_location).unwrap());
    let material_count = materials.len() as u32;
    let primitive_count = primitives.len() as u32;
    let texture_count = textures.len() as u32;
    output_file.write(&material_count.to_le_bytes()).unwrap();
    output_file.write(&primitive_count.to_le_bytes()).unwrap();
    output_file.write(&texture_count.to_le_bytes()).unwrap();
    for material in materials.iter() {
        output_file.write(slice_to_bytes(&material.base_color)).unwrap();
        output_file.write(slice_to_bytes(&material.emissive_factor)).unwrap();
        output_file.write(&material.base_roughness.to_le_bytes()).unwrap();
        output_file.write(&material.base_metalness.to_le_bytes()).unwrap();
        output_file.write(&material.color_bc7_idx.unwrap_or(0xFFFFFFFF).to_le_bytes()).unwrap();
        output_file.write(&material.normal_bc7_idx.unwrap_or(0xFFFFFFFF).to_le_bytes()).unwrap();
        output_file.write(&material.arm_bc7_idx.unwrap_or(0xFFFFFFFF).to_le_bytes()).unwrap();
        output_file.write(&material.emissive_bc7_idx.unwrap_or(0xFFFFFFFF).to_le_bytes()).unwrap();
    }
    for primitive in primitives.iter() {
        output_file.write(&primitive.material_idx.to_le_bytes()).unwrap();

        let count = primitive.indices.len() as u32;
        output_file.write(&count.to_le_bytes()).unwrap();
        output_file.write(vec_to_bytes(&primitive.indices)).unwrap();

        let count = primitive.vertex_positions.len() as u32;
        output_file.write(&count.to_le_bytes()).unwrap();
        output_file.write(vec_to_bytes(&primitive.vertex_positions)).unwrap();

        let count = primitive.vertex_normals.len() as u32;
        output_file.write(&count.to_le_bytes()).unwrap();
        output_file.write(vec_to_bytes(&primitive.vertex_normals)).unwrap();

        let count = primitive.vertex_tangents.len() as u32;
        output_file.write(&count.to_le_bytes()).unwrap();
        output_file.write(vec_to_bytes(&primitive.vertex_tangents)).unwrap();

        let count = primitive.vertex_uvs.len() as u32;
        output_file.write(&count.to_le_bytes()).unwrap();
        output_file.write(vec_to_bytes(&primitive.vertex_uvs)).unwrap();
    }
    for texture in textures {
        //Images are written out as width, height, mip_levels, then bc7_bytes
        output_file.write(&texture.width.to_le_bytes()).unwrap();
        output_file.write(&texture.height.to_le_bytes()).unwrap();
        output_file.write(&texture.mipmap_count.to_le_bytes()).unwrap();
        output_file.write(&texture.bc7_bytes).unwrap();
    }
}

fn load_raw_image_data(glb: &Gltf, prim: &gltf::Primitive) -> RawImageData {
    let mat = prim.material();
    let pbr_model = mat.pbr_metallic_roughness();

    let mut color_bytes = vec![];
    let mut color_imagetype = GLTFImageType::PNG;
    let color_index = match pbr_model.base_color_texture() {
        Some(t) => {
            let image = t.texture().source();
            let idx = image.index();
            let source = image.source();
            (color_bytes, color_imagetype) = image_bytes_from_source(&glb, source);
            Some(idx)
        }
        None => { None }
    };

    let mut normal_bytes = vec![];
    let mut normal_imagetype = GLTFImageType::PNG;
    let normal_index = match mat.normal_texture() {
        Some(t) => {
            let image = t.texture().source();
            let idx = image.index();
            let source = image.source();
            (normal_bytes, normal_imagetype) = image_bytes_from_source(&glb, source);
            Some(idx)
        }
        None => { None }
    };

    let mut arm_bytes = vec![];
    let mut arm_imagetype = GLTFImageType::PNG;
    let mut metalrough_glb_index = -1;
    let arm_index = match pbr_model.metallic_roughness_texture() {
        Some(t) => {
            let image = t.texture().source();
            let idx = image.index();
            metalrough_glb_index = idx as i32;
            let source = image.source();
            (arm_bytes, arm_imagetype) = image_bytes_from_source(&glb, source);
            Some(idx)
        }
        None => { None }
    };
    if let Some(occulusion_texture) = mat.occlusion_texture() {
        let image = occulusion_texture.texture().source();
        if metalrough_glb_index != image.index() as i32 {
            crash_with_error_dialog("Unimplemented case of ao not being packed with metallic_roughness.");
        }
    }

    let mut emissive_bytes = vec![];
    let mut emissive_imagetype = GLTFImageType::PNG;
    let emissive_index = match mat.emissive_texture() {
        Some(t) => {
            let image = t.texture().source();
            let idx = image.index();
            let source = image.source();
            (emissive_bytes, emissive_imagetype) = image_bytes_from_source(&glb, source);
            Some(idx)
        }
        None => { None }
    };

    RawImageData {
        color_index,
        color_bytes,
        color_imagetype,
        normal_index,
        normal_bytes,
        normal_imagetype,
        arm_index,
        arm_bytes,
        arm_imagetype,
        emissive_index,
        emissive_bytes,
        emissive_imagetype
    }
}

fn load_mesh_primitives(glb: &Gltf, mesh: &Mesh, out_primitive_array: &mut Vec<GLTFPrimitive>, texture_bytes: &mut [Vec<u8>]) {
    for prim in mesh.primitives() {
        let mat = prim.material();
        let pbr_model = mat.pbr_metallic_roughness();
        let vertex_data = uninterleaved_primitive_vertex_data(glb, &prim);

        let image_data = load_raw_image_data(glb, &prim);
        if let Some(idx) = image_data.color_index {
            if texture_bytes[idx].len() == 0 {
                texture_bytes[idx] = image_data.color_bytes;
            }
        }
        if let Some(idx) = image_data.normal_index {
            if texture_bytes[idx].len() == 0 {
                texture_bytes[idx] = image_data.normal_bytes;
            }
        }
        if let Some(idx) = image_data.arm_index {
            if texture_bytes[idx].len() == 0 {
                texture_bytes[idx] = image_data.arm_bytes;
            }
        }
        if let Some(idx) = image_data.emissive_index {
            if texture_bytes[idx].len() == 0 {
                texture_bytes[idx] = image_data.emissive_bytes;
            }
        }

        let mat = GLTFMaterial {
            base_color: pbr_model.base_color_factor(),
            base_roughness: pbr_model.roughness_factor(),
            base_metalness: pbr_model.metallic_factor(),
            emissive_factor: mat.emissive_factor(),
            color_index: image_data.color_index,
            color_imagetype: GLTFImageType::PNG,
            normal_index: image_data.normal_index,
            normal_imagetype: GLTFImageType::PNG,
            metallic_roughness_index: image_data.arm_index,
            metallic_roughness_imagetype: GLTFImageType::PNG,
            emissive_index: image_data.emissive_index,
            emissive_imagetype: GLTFImageType::PNG
        };
        let p = GLTFPrimitive {
            vertex_positions: vertex_data.positions,
            vertex_normals: vertex_data.normals,
            vertex_tangents: vertex_data.tangents,
            vertex_uvs: vertex_data.uvs,
            indices: vertex_data.indices,
            material: mat
        };
        out_primitive_array.push(p);
    }
}

#[named]
pub fn _gltf_scenedata(path: &str) -> GLTFSceneData {
    let glb = Gltf::open(path).unwrap();

    //We only load glb's that have one scene
    if glb.scenes().len() != 1 { crash_with_error_dialog("Invalid glTF: glTF must contain exactly one scene."); }
    let scene = glb.scenes().next().unwrap();
    let scene_name = match scene.name() {
        Some(n) => { Some(String::from(n)) }
        None => { None }
    };

    let mut meshes = Vec::with_capacity(glb.meshes().count());
    for node in scene.nodes() {
        let mut primitives = vec![];
        let mut texture_bytes = vec![Vec::new(); glb.textures().count()];
        if let Some(mesh) = node.mesh() {
            if let None = mesh.name() {
                crash_with_error_dialog(&format!("Assets must be named: {}", function_name!()));
            }
            let name = String::from(mesh.name().unwrap());
            
            load_mesh_primitives(&glb, &mesh, &mut primitives, &mut texture_bytes);
        
            let mut children = Vec::with_capacity(node.children().count());
            for child in node.children() {
                children.push(child.index());
            }
    
            let mesh_data = GLTFMeshData {
                name,
                primitives,
                texture_bytes,
                children
            };
            meshes.push(mesh_data);
        }
    }

    GLTFSceneData {
        name: scene_name,
        meshes
    }
}

pub fn gltf_meshdata(path: &str) -> GLTFMeshData {
    let glb = Gltf::open(path).unwrap();
    let mut primitives = vec![];
    let mut texture_bytes = vec![Vec::new(); glb.textures().count()];
    let mut name = None;

    for mesh in glb.meshes() {
        if let None = name {
            if let Some(n) = mesh.name() {
                name = Some(String::from(n));
            }
        }

        load_mesh_primitives(&glb, &mesh, &mut primitives, &mut texture_bytes);
    }

    if let None = name {
        crash_with_error_dialog("Something needs to be named!");
    }
    let name = name.unwrap();

    GLTFMeshData {
        name,
        primitives,
        texture_bytes,
        children: vec![]
    }
}