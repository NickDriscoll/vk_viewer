#![allow(non_snake_case)]
extern crate nalgebra_glm as glm;
extern crate ozy_engine as ozy;
extern crate tinyfiledialogs as tfd;

mod dllr;
mod structs;

use ash::vk;
use ash::vk::Handle;
use imgui::{DrawCmd, FontAtlasRefMut};
use sdl2::event::Event;
use sdl2::mixer;
use sdl2::mixer::Music;
use structs::FreeCam;
use std::fs::{File};
use std::ffi::CStr;
use std::mem::size_of;
use std::os::raw::c_void;
use std::ptr;

use ozy::io::OzyMesh;
use ozy::structs::FrameTimer;

const COMPONENT_MAPPING_DEFAULT: vk::ComponentMapping = vk::ComponentMapping {
    r: vk::ComponentSwizzle::R,
    g: vk::ComponentSwizzle::G,
    b: vk::ComponentSwizzle::B,
    a: vk::ComponentSwizzle::A,
};

const VK_MEMORY_ALLOCATOR: Option<&vk::AllocationCallbacks> = None;

fn crash_with_error_dialog(message: &str) -> ! {
    let message = message.replace("'", "");
    tfd::message_box_ok("Oops...", &message, tfd::MessageBoxIcon::Error);
    panic!("{}", message);
}

fn unwrap_result<T>(res: Result<T, String>) -> T {
    match res {
        Ok(t) => { t }
        Err(e) => {
            crash_with_error_dialog(&e);
        }
    }
}

//Entry point
fn main() {
    //Create the window using SDL
    let sdl_ctxt = sdl2::init().unwrap();
    let mut event_pump = sdl_ctxt.event_pump().unwrap();
    let video_subsystem = sdl_ctxt.video().unwrap();
    let window_size = glm::vec2(1280, 1024);
    let window = video_subsystem.window("Vulkan't", window_size.x, window_size.y).position_centered().vulkan().build().unwrap();

    //Initialize the SDL mixer
    let mut music_volume = 16;
    let _sdl_mixer = mixer::init(mixer::InitFlag::FLAC | mixer::InitFlag::MP3).unwrap();
    mixer::open_audio(mixer::DEFAULT_FREQUENCY, mixer::DEFAULT_FORMAT, 2, 256).unwrap();
    Music::set_volume(music_volume);

    //Load and play bgm
    let bgm = unwrap_result(Music::from_file("./data/music/relaxing_botw.mp3"));
    bgm.play(-1).unwrap();

    //Initialize the Vulkan API
    let vk_entry = ash::Entry::linked();
    let vk_instance = {
        let app_info = vk::ApplicationInfo {
            api_version: vk::make_api_version(0, 1, 3, 0),
            ..Default::default()
        };

        #[cfg(target_os = "windows")]
        let extension_names = [
            ash::extensions::khr::Surface::name().as_ptr(),
            ash::extensions::khr::Win32Surface::name().as_ptr()
        ];

        #[cfg(target_os = "macos")]
        let extension_names = [
            ash::extensions::khr::Surface::name().as_ptr(),
            ash::extensions::mvk::MacOSSurface::name().as_ptr()
        ];

        #[cfg(target_os = "linux")]
        let extension_names = [
            ash::extensions::khr::Surface::name().as_ptr(),
            ash::extensions::khr::XlibSurface::name().as_ptr()
        ];

        let layer_names = unsafe  {[
            CStr::from_bytes_with_nul_unchecked(b"VK_LAYER_KHRONOS_validation\0").as_ptr()
        ]};
        let vk_create_info = vk::InstanceCreateInfo {
            p_application_info: &app_info,
            enabled_extension_count: extension_names.len() as u32,
            pp_enabled_extension_names: &extension_names as *const *const _,
            enabled_layer_count: layer_names.len() as u32,
            pp_enabled_layer_names: &layer_names as *const *const _,
            ..Default::default()
        };

        unsafe { vk_entry.create_instance(&vk_create_info, VK_MEMORY_ALLOCATOR).unwrap() }
    };

    //Use SDL to create the Vulkan surface
    let vk_surface = {
        let raw_surf = window.vulkan_create_surface(vk_instance.handle().as_raw() as usize).unwrap();
        vk::SurfaceKHR::from_raw(raw_surf)
    };
    
    //Create surface extension object
    let vk_ext_surface = ash::extensions::khr::Surface::new(&vk_entry, &vk_instance);

    //Create the Vulkan device
    let vk_physical_device;
    let vk_physical_device_properties;
    let mut vk_queue_family_index = 0;
    let vk_device = unsafe {
        match vk_instance.enumerate_physical_devices() {
            Ok(phys_devices) => {
                let mut phys_device = None;
                let device_types = [vk::PhysicalDeviceType::DISCRETE_GPU, vk::PhysicalDeviceType::INTEGRATED_GPU, vk::PhysicalDeviceType::CPU];
                'gpu_search: for d_type in device_types {
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
                
                let mut indexing_features = vk::PhysicalDeviceDescriptorIndexingFeatures::default();
                let mut physical_device_features = vk::PhysicalDeviceFeatures2 {
                    p_next: &mut indexing_features as *mut _ as *mut c_void,
                    ..Default::default()
                };
                vk_instance.get_physical_device_features2(vk_physical_device, &mut physical_device_features);

                let mut i = 0;
                for qfp in vk_instance.get_physical_device_queue_family_properties(vk_physical_device) {
                    if qfp.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
                        vk_queue_family_index = i;
                        break;
                    }
                    i += 1;
                }

                let priorities = [1.0];
                let queue_create_info = vk::DeviceQueueCreateInfo {
                    queue_family_index: vk_queue_family_index,
                    queue_count: 1,
                    p_queue_priorities: priorities.as_ptr(),
                    ..Default::default()
                };
                
                if !vk_ext_surface.get_physical_device_surface_support(vk_physical_device, vk_queue_family_index, vk_surface).unwrap() {
                    panic!("The physical device queue doesn't support swapchain present!");
                }

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

                vk_instance.create_device(vk_physical_device, &create_info, VK_MEMORY_ALLOCATOR).unwrap()
            }
            Err(e) => {
                crash_with_error_dialog(&format!("Unable to enumerate physical devices: {}", e));
            }
        }
    };

    //Create command buffer
    let vk_command_buffer = unsafe {
        let pool_create_info = vk::CommandPoolCreateInfo {
            queue_family_index: vk_queue_family_index,
            flags: vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
            ..Default::default()
        };

        let command_pool = vk_device.create_command_pool(&pool_create_info, VK_MEMORY_ALLOCATOR).unwrap();

        let command_buffer_alloc_info = vk::CommandBufferAllocateInfo {
            command_pool,
            command_buffer_count: 1,
            level: vk::CommandBufferLevel::PRIMARY,
            ..Default::default()
        };
        vk_device.allocate_command_buffers(&command_buffer_alloc_info).unwrap()[0]
    };

    //Initialize Dear ImGUI
    let mut imgui_context = imgui::Context::create();
    imgui_context.style_mut().use_dark_colors();
    {
        let io = imgui_context.io_mut();
        io.display_size[0] = window_size.x as f32;
        io.display_size[1] = window_size.y as f32;
    }
    
    //Create and upload Dear IMGUI font atlas
    let imgui_font_image = match imgui_context.fonts() {
        FontAtlasRefMut::Owned(atlas) => unsafe {
            let atlas_texture = atlas.build_alpha8_texture();

            let size = (atlas_texture.width * atlas_texture.height) as vk::DeviceSize;
            let buffer_create_info = vk::BufferCreateInfo {
                usage: vk::BufferUsageFlags::TRANSFER_SRC,
                size,
                sharing_mode: vk::SharingMode::EXCLUSIVE,
                ..Default::default()
            };
            let staging_buffer = vk_device.create_buffer(&buffer_create_info, VK_MEMORY_ALLOCATOR).unwrap();            
            let staging_buffer_memory = dllr::allocate_buffer_memory(&vk_instance, vk_physical_device, &vk_device, staging_buffer);    
            vk_device.bind_buffer_memory(staging_buffer, staging_buffer_memory, 0).unwrap();


            let staging_ptr = vk_device.map_memory(staging_buffer_memory, 0, size, vk::MemoryMapFlags::empty()).unwrap();
            ptr::copy_nonoverlapping(atlas_texture.data.as_ptr(), staging_ptr as *mut _, size as usize);
            vk_device.unmap_memory(staging_buffer_memory);
            
            let image_extent = vk::Extent3D {
                width: atlas_texture.width,
                height: atlas_texture.height,
                depth: 1
            };
            let font_create_info = vk::ImageCreateInfo {
                image_type: vk::ImageType::TYPE_2D,
                format: vk::Format::R8_UNORM,
                extent: image_extent,
                mip_levels: 1,
                array_layers: 1,
                samples: vk::SampleCountFlags::TYPE_1,
                tiling: vk::ImageTiling::OPTIMAL,
                usage: vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED,
                sharing_mode: vk::SharingMode::EXCLUSIVE,
                queue_family_index_count: 1,
                p_queue_family_indices: &vk_queue_family_index,
                initial_layout: vk::ImageLayout::UNDEFINED,
                ..Default::default()
            };
            let vk_font_image = vk_device.create_image(&font_create_info, VK_MEMORY_ALLOCATOR).unwrap();
            
            let font_image_memory = dllr::allocate_image_memory(&vk_instance, vk_physical_device, &vk_device, vk_font_image);
    
            vk_device.bind_image_memory(vk_font_image, font_image_memory, 0).unwrap();

            vk_device.begin_command_buffer(vk_command_buffer, &vk::CommandBufferBeginInfo::default()).unwrap();

            let subresource_range = vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1
            };
            let image_memory_barrier = vk::ImageMemoryBarrier {
                src_access_mask: vk::AccessFlags::empty(),
                dst_access_mask: vk::AccessFlags::TRANSFER_WRITE,
                old_layout: vk::ImageLayout::UNDEFINED,
                new_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                image: vk_font_image,
                subresource_range,
                ..Default::default()
            };
            vk_device.cmd_pipeline_barrier(vk_command_buffer, vk::PipelineStageFlags::TOP_OF_PIPE, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[], &[], &[image_memory_barrier]);

            let subresource_layers = vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1

            };
            let copy_region = vk::BufferImageCopy {
                buffer_offset: 0,
                buffer_row_length: 0,
                buffer_image_height: 0,
                image_extent,
                image_offset: vk::Offset3D::default(),
                image_subresource: subresource_layers
            };
            vk_device.cmd_copy_buffer_to_image(vk_command_buffer, staging_buffer, vk_font_image, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &[copy_region]);
            
            let subresource_range = vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1
            };
            let image_memory_barrier = vk::ImageMemoryBarrier {
                src_access_mask: vk::AccessFlags::TRANSFER_WRITE,
                dst_access_mask: vk::AccessFlags::SHADER_READ,
                old_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                new_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                image: vk_font_image,
                subresource_range,
                ..Default::default()
            };
            vk_device.cmd_pipeline_barrier(vk_command_buffer, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::FRAGMENT_SHADER, vk::DependencyFlags::empty(), &[], &[], &[image_memory_barrier]);


            vk_device.end_command_buffer(vk_command_buffer).unwrap();

            
            let submit_info = vk::SubmitInfo {
                command_buffer_count: 1,
                p_command_buffers: &vk_command_buffer,
                ..Default::default()
            };

            let fence = vk_device.create_fence(&vk::FenceCreateInfo::default(), VK_MEMORY_ALLOCATOR).unwrap();
            let queue = vk_device.get_device_queue(vk_queue_family_index, 0);
            vk_device.queue_submit(queue, &[submit_info], fence).unwrap();
            vk_device.wait_for_fences(&[fence], true, vk::DeviceSize::MAX).unwrap();
            vk_device.destroy_fence(fence, VK_MEMORY_ALLOCATOR);
            vk_device.destroy_buffer(staging_buffer, VK_MEMORY_ALLOCATOR);

            atlas.tex_id = imgui::TextureId::new(0);    //Giving Dear Imgui a reference to the font atlas GPU texture
            atlas.clear_tex_data();                         //Free atlas memory CPU-side

            vk_font_image
        }
        FontAtlasRefMut::Shared(_) => {
            panic!("Not dealing with this case.");
        }
    };

    //Create swapchain extension object
    let vk_ext_swapchain = ash::extensions::khr::Swapchain::new(&vk_instance, &vk_device);

    //Create the main swapchain for window present
    let vk_swapchain_image_format;
    let vk_swapchain_extent;
    let vk_swapchain = unsafe {
        let present_mode = vk_ext_surface.get_physical_device_surface_present_modes(vk_physical_device, vk_surface).unwrap()[0];
        let surf_capabilities = vk_ext_surface.get_physical_device_surface_capabilities(vk_physical_device, vk_surface).unwrap();
        let surf_formats = vk_ext_surface.get_physical_device_surface_formats(vk_physical_device, vk_surface).unwrap();

        //Search for an SRGB swapchain format
        let mut surf_format = vk::SurfaceFormatKHR::default();
        for sformat in surf_formats.iter() {
            if sformat.format == vk::Format::B8G8R8A8_SRGB {
                surf_format = *sformat;
                break;
            }
            surf_format = vk::SurfaceFormatKHR::default();
        }

        vk_swapchain_image_format = surf_format.format;
        vk_swapchain_extent = surf_capabilities.current_extent;
        let create_info = vk::SwapchainCreateInfoKHR {
            surface: vk_surface,
            min_image_count: surf_capabilities.min_image_count,
            image_format: vk_swapchain_image_format,
            image_color_space: surf_format.color_space,
            image_extent: surf_capabilities.current_extent,
            image_array_layers: 1,
            image_usage: vk::ImageUsageFlags::COLOR_ATTACHMENT,
            image_sharing_mode: vk::SharingMode::EXCLUSIVE,
            queue_family_index_count: 1,
            p_queue_family_indices: [vk_queue_family_index].as_ptr(),
            pre_transform: surf_capabilities.current_transform,
            composite_alpha: vk::CompositeAlphaFlagsKHR::OPAQUE,
            present_mode,
            ..Default::default()
        };

        let sc = vk_ext_swapchain.create_swapchain(&create_info, VK_MEMORY_ALLOCATOR).unwrap();
        sc
    };
    
    let vk_swapchain_image_views = unsafe {
        let vk_swapchain_images = vk_ext_swapchain.get_swapchain_images(vk_swapchain).unwrap();

        let mut image_views = Vec::with_capacity(vk_swapchain_images.len());
        for i in 0..vk_swapchain_images.len() {
            let image_subresource_range = vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1
            };
            let view_info = vk::ImageViewCreateInfo {
                image: vk_swapchain_images[i],
                format: vk_swapchain_image_format,
                view_type: vk::ImageViewType::TYPE_2D,
                components: COMPONENT_MAPPING_DEFAULT,
                subresource_range: image_subresource_range,
                ..Default::default()
            };

            image_views.push(vk_device.create_image_view(&view_info, VK_MEMORY_ALLOCATOR).unwrap());
        }

        image_views
    };

    let vk_depth_format;
    let vk_depth_image = unsafe {
        let surf_capabilities = vk_ext_surface.get_physical_device_surface_capabilities(vk_physical_device, vk_surface).unwrap();
        let extent = vk::Extent3D {
            width: surf_capabilities.current_extent.width,
            height: surf_capabilities.current_extent.height,
            depth: 1
        };

        vk_depth_format = vk::Format::D24_UNORM_S8_UINT;
        let create_info = vk::ImageCreateInfo {
            queue_family_index_count: 1,
            p_queue_family_indices: [vk_queue_family_index].as_ptr(),
            flags: vk::ImageCreateFlags::empty(),
            image_type: vk::ImageType::TYPE_2D,
            format: vk_depth_format,
            extent,
            mip_levels: 1,
            array_layers: 1,
            samples: vk::SampleCountFlags::TYPE_1,
            usage: vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            ..Default::default()
        };

        let depth_image = vk_device.create_image(&create_info, VK_MEMORY_ALLOCATOR).unwrap();
        let depth_memory = dllr::allocate_image_memory(&vk_instance, vk_physical_device, &vk_device, depth_image);

        //Bind the depth image to its memory
        vk_device.bind_image_memory(depth_image, depth_memory, 0).unwrap();

        depth_image
    };

    let vk_depth_image_view = unsafe {
        let image_subresource_range = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::DEPTH,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1
        };
        let view_info = vk::ImageViewCreateInfo {
            image: vk_depth_image,
            format: vk_depth_format,
            view_type: vk::ImageViewType::TYPE_2D,
            components: COMPONENT_MAPPING_DEFAULT,
            subresource_range: image_subresource_range,
            ..Default::default()
        };

        vk_device.create_image_view(&view_info, VK_MEMORY_ALLOCATOR).unwrap()
    };

    let global_transform_slots = 1024 * 1024;
    let vk_uniform_buffer_size;
    let vk_transform_storage_buffer;
    let vk_scene_storage_buffer_ptr = unsafe {
        let mut buffer_size = (size_of::<glm::TMat4<f32>>() * global_transform_slots) as vk::DeviceSize;
        let alignment = vk_physical_device_properties.limits.min_uniform_buffer_offset_alignment;
        if alignment > 0 {
            buffer_size = (buffer_size + (alignment - 1)) & !(alignment - 1);   //Alignment is 2^N where N is a whole number
        }
        vk_uniform_buffer_size = buffer_size;
        println!("Transform buffer is {} bytes", vk_uniform_buffer_size);

        let buffer_create_info = vk::BufferCreateInfo {
            usage: vk::BufferUsageFlags::STORAGE_BUFFER,
            size: vk_uniform_buffer_size,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            ..Default::default()
        };

        vk_transform_storage_buffer = vk_device.create_buffer(&buffer_create_info, VK_MEMORY_ALLOCATOR).unwrap();
        
        let transform_buffer_memory = dllr::allocate_buffer_memory(&vk_instance, vk_physical_device, &vk_device, vk_transform_storage_buffer);

        vk_device.bind_buffer_memory(vk_transform_storage_buffer, transform_buffer_memory, 0).unwrap();

        let uniform_ptr = vk_device.map_memory(transform_buffer_memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty()).unwrap();
        uniform_ptr
    };

    let vk_descriptor_set_layout;
    let vk_pipeline_layout = unsafe {
        let storage_binding = vk::DescriptorSetLayoutBinding {
            binding: 0,
            descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: 1,
            stage_flags: vk::ShaderStageFlags::VERTEX,
            ..Default::default()
        };

        let texture_binding = vk::DescriptorSetLayoutBinding {
            binding: 1,
            descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            descriptor_count: 1,
            stage_flags: vk::ShaderStageFlags::FRAGMENT,
            ..Default::default()
        };
        
        let bindings = [storage_binding, texture_binding];
        
        let descriptor_layout = vk::DescriptorSetLayoutCreateInfo {
            binding_count: bindings.len() as u32,
            p_bindings: bindings.as_ptr(),
            ..Default::default()
        };

        vk_descriptor_set_layout = vk_device.create_descriptor_set_layout(&descriptor_layout, VK_MEMORY_ALLOCATOR).unwrap();

        let push_constant_range = vk::PushConstantRange {
            stage_flags: vk::ShaderStageFlags::FRAGMENT,
            offset: 0,
            size: size_of::<u32>() as u32
        };
        let pipeline_layout_createinfo = vk::PipelineLayoutCreateInfo {
            push_constant_range_count: 1,
            p_push_constant_ranges: &push_constant_range,
            set_layout_count: 1,
            p_set_layouts: &vk_descriptor_set_layout,
            ..Default::default()
        };
        
        vk_device.create_pipeline_layout(&pipeline_layout_createinfo, VK_MEMORY_ALLOCATOR).unwrap()
    };
    
    //Set up descriptors
    let vk_descriptor_sets = unsafe {
        let storage_pool_size = vk::DescriptorPoolSize {
            ty: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: 1
        };
        let sampler_pool_size = vk::DescriptorPoolSize {
            ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            descriptor_count: 1,
        };

        let pool_sizes = [storage_pool_size, sampler_pool_size];
        let descriptor_pool_info = vk::DescriptorPoolCreateInfo {
            max_sets: 1,
            pool_size_count: pool_sizes.len() as u32,
            p_pool_sizes: pool_sizes.as_ptr(),
            ..Default::default()
        };
        let descriptor_pool = vk_device.create_descriptor_pool(&descriptor_pool_info, VK_MEMORY_ALLOCATOR).unwrap();

        let vk_alloc_info = vk::DescriptorSetAllocateInfo {
            descriptor_pool,
            descriptor_set_count: 1,
            p_set_layouts: &vk_descriptor_set_layout,
            ..Default::default()
        };
        let vk_descriptor_sets = vk_device.allocate_descriptor_sets(&vk_alloc_info).unwrap();

        let buffer_infos = [
            vk::DescriptorBufferInfo {
                buffer: vk_transform_storage_buffer,
                offset: 0,
                range: (global_transform_slots * size_of::<glm::TMat4<f32>>()) as vk::DeviceSize
            }
        ];

        let storage_write = vk::WriteDescriptorSet {
            dst_set: vk_descriptor_sets[0],
            descriptor_count: 1,
            descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
            p_buffer_info: buffer_infos.as_ptr(),
            dst_array_element: 0,
            dst_binding: 0,
            ..Default::default()
        };

        let sampler_subresource_range = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1
        };
        let sampler_view_info = vk::ImageViewCreateInfo {
            image: imgui_font_image,
            format: vk::Format::R8_UNORM,
            view_type: vk::ImageViewType::TYPE_2D,
            components: COMPONENT_MAPPING_DEFAULT,
            subresource_range: sampler_subresource_range,
            ..Default::default()
        };
        let sampler_view = vk_device.create_image_view(&sampler_view_info, VK_MEMORY_ALLOCATOR).unwrap();
        let sampler_info = vk::SamplerCreateInfo {
            min_filter: vk::Filter::NEAREST,
            mag_filter: vk::Filter::NEAREST,
            mipmap_mode: vk::SamplerMipmapMode::NEAREST,
            address_mode_u: vk::SamplerAddressMode::MIRRORED_REPEAT,
            address_mode_v: vk::SamplerAddressMode::MIRRORED_REPEAT,
            address_mode_w: vk::SamplerAddressMode::MIRRORED_REPEAT,
            mip_lod_bias: 0.0,
            anisotropy_enable: vk::FALSE,
            compare_enable: vk::FALSE,
            min_lod: 0.0,
            max_lod: vk::LOD_CLAMP_NONE,
            border_color: vk::BorderColor::FLOAT_OPAQUE_BLACK,
            unnormalized_coordinates: vk::FALSE,
            ..Default::default()
        };
        let sampler = vk_device.create_sampler(&sampler_info, VK_MEMORY_ALLOCATOR).unwrap();
        let image_info = vk::DescriptorImageInfo {
            sampler,
            image_view: sampler_view,
            image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
        };
        let sampler_write = vk::WriteDescriptorSet {
            dst_set: vk_descriptor_sets[0],
            descriptor_count: 1,
            descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            p_image_info: &image_info,
            dst_array_element: 0,
            dst_binding: 1,
            ..Default::default()
        };

        let writes = [storage_write, sampler_write];
        vk_device.update_descriptor_sets(&writes, &[]);

        vk_descriptor_sets
    };

    let vk_render_pass = unsafe {
        let color_attachment_description = vk::AttachmentDescription {
            format: vk_swapchain_image_format,
            samples: vk::SampleCountFlags::TYPE_1,
            load_op: vk::AttachmentLoadOp::CLEAR,
            store_op: vk::AttachmentStoreOp::STORE,
            stencil_load_op: vk::AttachmentLoadOp::DONT_CARE,
            stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
            initial_layout: vk::ImageLayout::UNDEFINED,
            final_layout: vk::ImageLayout::PRESENT_SRC_KHR,
            ..Default::default()
        };

        let depth_attachment_description = vk::AttachmentDescription {
            format: vk_depth_format,
            samples: vk::SampleCountFlags::TYPE_1,
            load_op: vk::AttachmentLoadOp::CLEAR,
            store_op: vk::AttachmentStoreOp::DONT_CARE,
            stencil_load_op: vk::AttachmentLoadOp::DONT_CARE,
            stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
            initial_layout: vk::ImageLayout::UNDEFINED,
            final_layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            ..Default::default()
        };

        let color_attachment_reference = vk::AttachmentReference {
            attachment: 0,
            layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
        };
        let depth_attachment_reference = vk::AttachmentReference {
            attachment: 1,
            layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL
        };

        let subpass = vk::SubpassDescription {
            pipeline_bind_point: vk::PipelineBindPoint::GRAPHICS,
            color_attachment_count: 1,
            p_color_attachments: &color_attachment_reference,
            p_depth_stencil_attachment: &depth_attachment_reference,
            ..Default::default()
        };

        let subpass_dependency = vk::SubpassDependency {
            src_subpass: vk::SUBPASS_EXTERNAL,
            dst_subpass: 0,
            src_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            dst_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            src_access_mask: vk::AccessFlags::NONE_KHR,
            dst_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            ..Default::default()
        };

        let attachments = [color_attachment_description, depth_attachment_description];
        let renderpass_info = vk::RenderPassCreateInfo {
            attachment_count: attachments.len() as u32,
            p_attachments: attachments.as_ptr(),
            subpass_count: 1,
            p_subpasses: &subpass,
            dependency_count: 1,
            p_dependencies: &subpass_dependency,
            ..Default::default()
        };
        vk_device.create_render_pass(&renderpass_info, VK_MEMORY_ALLOCATOR).unwrap()
    };

    //Load shaders
    let vk_shader_stages = unsafe {
        let v = dllr::load_shader_stage(&vk_device, vk::ShaderStageFlags::VERTEX, "./shaders/main_vert.spv");
        let f = dllr::load_shader_stage(&vk_device, vk::ShaderStageFlags::FRAGMENT, "./shaders/main_frag.spv");
        [v, f]
    };

    let imgui_shader_stages = unsafe {
        let v = dllr::load_shader_stage(&vk_device, vk::ShaderStageFlags::VERTEX, "./shaders/imgui_vert.spv");
        let f = dllr::load_shader_stage(&vk_device, vk::ShaderStageFlags::FRAGMENT, "./shaders/imgui_frag.spv");
        [v, f]
    };

    //Create framebuffers
    let vk_swapchain_framebuffers = unsafe {
        let mut attachments = [vk::ImageView::default(), vk_depth_image_view];
        let fb_info = vk::FramebufferCreateInfo {
            render_pass: vk_render_pass,
            attachment_count: attachments.len() as u32,
            p_attachments: attachments.as_ptr(),
            width: vk_swapchain_extent.width,
            height: vk_swapchain_extent.height,
            layers: 1,
            ..Default::default()
        };

        let mut fbs = Vec::with_capacity(vk_swapchain_image_views.len());
        for view in vk_swapchain_image_views {
            attachments[0] = view;
            fbs.push(vk_device.create_framebuffer(&fb_info, VK_MEMORY_ALLOCATOR).unwrap())
        }

        fbs
    };

    //Create graphics pipelines
    let [vk_3D_graphics_pipeline, imgui_graphics_pipeline] = unsafe {
        let dynamic_state_enables = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_state = vk::PipelineDynamicStateCreateInfo {
            p_dynamic_states: dynamic_state_enables.as_ptr(),
            dynamic_state_count: dynamic_state_enables.len() as u32,
            ..Default::default()
        };

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

        let color_blend_pipeline_state = vk::PipelineColorBlendStateCreateInfo {
            attachment_count: 1,
            p_attachments: &color_blend_attachment_state,
            logic_op_enable: vk::FALSE,
            logic_op: vk::LogicOp::NO_OP,
            blend_constants: [0.0; 4],
            ..Default::default()
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

        let vert_binding = vk::VertexInputBindingDescription {
            binding: 0,
            stride: 14 * size_of::<f32>() as u32,
            input_rate: vk::VertexInputRate::VERTEX
        };

        let position_attribute = vk::VertexInputAttributeDescription {
            location: 0,
            binding: 0,
            format: vk::Format::R32G32B32_SFLOAT,
            offset: 0
        };

        let tangent_attribute = vk::VertexInputAttributeDescription {
            location: 1,
            binding: 0,
            format: vk::Format::R32G32B32_SFLOAT,
            offset: 3 * size_of::<f32>() as u32
        };

        let bitangent_attribute = vk::VertexInputAttributeDescription {
            location: 2,
            binding: 0,
            format: vk::Format::R32G32B32_SFLOAT,
            offset: 6 * size_of::<f32>() as u32
        };

        let normal_attribute = vk::VertexInputAttributeDescription {
            location: 3,
            binding: 0,
            format: vk::Format::R32G32B32_SFLOAT,
            offset: 9 * size_of::<f32>() as u32
        };

        let uv_attribute = vk::VertexInputAttributeDescription {
            location: 4,
            binding: 0,
            format: vk::Format::R32G32_SFLOAT,
            offset: 12 * size_of::<f32>() as u32
        };

        let bindings = [vert_binding];
        let attrs = [position_attribute, tangent_attribute, bitangent_attribute, normal_attribute, uv_attribute];
        let vertex_input_state = vk::PipelineVertexInputStateCreateInfo {
            vertex_binding_description_count: bindings.len() as u32,
            p_vertex_binding_descriptions: bindings.as_ptr(),
            vertex_attribute_description_count: attrs.len() as u32,
            p_vertex_attribute_descriptions: attrs.as_ptr(),
            ..Default::default()
        };

        let graphics_pipeline_info = vk::GraphicsPipelineCreateInfo {
            layout: vk_pipeline_layout,
            p_vertex_input_state: &vertex_input_state,
            p_input_assembly_state: &input_assembly_state,
            p_rasterization_state: &rasterization_state,
            p_color_blend_state: &color_blend_pipeline_state,
            p_multisample_state: &multisample_state,
            p_dynamic_state: &dynamic_state,
            p_viewport_state: &viewport_state,
            p_depth_stencil_state: &depth_stencil_state,
            p_stages: vk_shader_stages.as_ptr(),
            stage_count: vk_shader_stages.len() as u32,
            render_pass: vk_render_pass,
            ..Default::default()
        };

        let imgui_rasterization_state = vk::PipelineRasterizationStateCreateInfo {
            polygon_mode: vk::PolygonMode::FILL,
            cull_mode: vk::CullModeFlags::NONE,
            depth_clamp_enable: vk::FALSE,
            rasterizer_discard_enable: vk::FALSE,
            depth_bias_enable: vk::TRUE,
            line_width: 1.0,
            ..Default::default()
        };

        let imgui_depth_stencil_state = vk::PipelineDepthStencilStateCreateInfo {
            depth_test_enable: vk::FALSE,
            depth_write_enable: vk::FALSE,
            depth_bounds_test_enable: vk::FALSE,
            stencil_test_enable: vk::FALSE,
            ..Default::default()
        };

        let im_vert_binding = vk::VertexInputBindingDescription {
            binding: 0,
            stride: 8 * size_of::<f32>() as u32,
            input_rate: vk::VertexInputRate::VERTEX
        };

        let im_position_attribute = vk::VertexInputAttributeDescription {
            location: 0,
            binding: 0,
            format: vk::Format::R32G32_SFLOAT,
            offset: 0
        };

        let im_uv_attribute = vk::VertexInputAttributeDescription {
            location: 1,
            binding: 0,
            format: vk::Format::R32G32_SFLOAT,
            offset: 2 * size_of::<f32>() as u32
        };

        let im_color_attribute = vk::VertexInputAttributeDescription {
            location: 2,
            binding: 0,
            format: vk::Format::R32G32B32A32_SFLOAT,
            offset: 4 * size_of::<f32>() as u32
        };

        let im_bindings = [im_vert_binding];
        let im_attrs = [im_position_attribute, im_uv_attribute, im_color_attribute];
        let im_vertex_input_state = vk::PipelineVertexInputStateCreateInfo {
            vertex_binding_description_count: im_bindings.len() as u32,
            p_vertex_binding_descriptions: im_bindings.as_ptr(),
            vertex_attribute_description_count: im_attrs.len() as u32,
            p_vertex_attribute_descriptions: im_attrs.as_ptr(),
            ..Default::default()
        };

        let imgui_pipeline_info = vk::GraphicsPipelineCreateInfo {
            layout: vk_pipeline_layout,
            p_vertex_input_state: &im_vertex_input_state,
            p_input_assembly_state: &input_assembly_state,
            p_rasterization_state: &imgui_rasterization_state,
            p_color_blend_state: &color_blend_pipeline_state,
            p_multisample_state: &multisample_state,
            p_dynamic_state: &dynamic_state,
            p_viewport_state: &viewport_state,
            p_depth_stencil_state: &imgui_depth_stencil_state,
            p_stages: imgui_shader_stages.as_ptr(),
            stage_count: imgui_shader_stages.len() as u32,
            render_pass: vk_render_pass,
            ..Default::default()
        };

        let pipeline_infos = [graphics_pipeline_info, imgui_pipeline_info];
        let pipelines = vk_device.create_graphics_pipelines(vk::PipelineCache::null(), &pipeline_infos, VK_MEMORY_ALLOCATOR).unwrap();
        [pipelines[0], pipelines[1]]
    };

    let g_plane_width = 64;
    let g_plane_height = 64;
    let g_plane_vertices = ozy::prims::plane_vertex_buffer(g_plane_width, g_plane_height, 5.0);
    let g_plane_indices = ozy::prims::plane_index_buffer(g_plane_width, g_plane_height);

    //Load UV sphere OzyMesh
    let uv_sphere = OzyMesh::load("./data/models/sphere.ozy").unwrap();
    let uv_sphere_indices: Vec<u32> = uv_sphere.vertex_array.indices.iter().map(|&n|{n as u32}).collect();

    let scene_vertex_buffers = [g_plane_vertices.as_slice(), uv_sphere.vertex_array.vertices.as_slice()];
    let scene_index_buffers = [g_plane_indices.as_slice(), uv_sphere_indices.as_slice()];

    //Allocate and distribute memory to buffer objects
    let g_plane_geometry;
    let sphere_geometry;
    unsafe {
        let mut scene_geo_buffer_size = 0;
        let scene_geo_buffer = {
            //Buffer creation
            for (&v_buffer, &i_buffer) in scene_vertex_buffers.iter().zip(scene_index_buffers.iter()) {
                scene_geo_buffer_size += v_buffer.len() * size_of::<f32>();
                scene_geo_buffer_size += i_buffer.len() * size_of::<u32>();
            }
            
            let buffer_create_info = vk::BufferCreateInfo {
                usage: vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::INDEX_BUFFER,
                size: scene_geo_buffer_size as vk::DeviceSize,
                sharing_mode: vk::SharingMode::EXCLUSIVE,
                ..Default::default()
            };
            let buffer = vk_device.create_buffer(&buffer_create_info, VK_MEMORY_ALLOCATOR).unwrap();
            buffer
        };

        let buffer_memory = dllr::allocate_buffer_memory(&vk_instance, vk_physical_device, &vk_device, scene_geo_buffer);

        //Bind buffer
        vk_device.bind_buffer_memory(scene_geo_buffer, buffer_memory, 0).unwrap();

        //Map buffer to host memory
        let buffer_ptr = vk_device.map_memory(
            buffer_memory,
            0,
            vk::WHOLE_SIZE,
            vk::MemoryMapFlags::empty()
        ).unwrap();

        //Create virtual bump allocator
        let mut scene_geo_allocator = dllr::VirtualBumpAllocator::new(scene_geo_buffer, buffer_ptr, scene_geo_buffer_size.try_into().unwrap());

        g_plane_geometry = scene_geo_allocator.allocate_geometry(&g_plane_vertices, &g_plane_indices).unwrap();
        sphere_geometry = scene_geo_allocator.allocate_geometry(&uv_sphere.vertex_array.vertices, &uv_sphere_indices).unwrap();

        vk_device.unmap_memory(buffer_memory);
    }

    let mut imgui_geo_allocator = unsafe {
        let imgui_buffer_size = 1024 * 64;
        let buffer_create_info = vk::BufferCreateInfo {
            usage: vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::INDEX_BUFFER,
            size: imgui_buffer_size as vk::DeviceSize,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            ..Default::default()
        };
        let buffer = vk_device.create_buffer(&buffer_create_info, VK_MEMORY_ALLOCATOR).unwrap();
        let buffer_memory = dllr::allocate_buffer_memory(&vk_instance, vk_physical_device, &vk_device, buffer);
        vk_device.bind_buffer_memory(buffer, buffer_memory, 0).unwrap();

        let ptr = vk_device.map_memory(buffer_memory, 0, imgui_buffer_size, vk::MemoryMapFlags::empty()).unwrap();

        dllr::VirtualBumpAllocator::new(buffer, ptr, imgui_buffer_size)
    };

    let vk_render_area = {        
        let offset = vk::Offset2D {
            x: 0,
            y: 0
        };
        vk::Rect2D {
            offset,
            extent: vk_swapchain_extent
        }
    };

    let vk_color_clear = {
        let color = vk::ClearColorValue {
            float32: [0.0, 0.0, 0.0, 1.0]
        };
        vk::ClearValue {
            color
        }
    };

    let vk_depth_stencil_clear = {
        let value = vk::ClearDepthStencilValue {
            depth: 1.0,
            stencil: 0
        };
        vk::ClearValue {
            depth_stencil: value
        }
    };

    let vk_clear_values = [vk_color_clear, vk_depth_stencil_clear];

    //Create semaphore used to wait on swapchain image
    let vk_swapchain_semaphore = unsafe { vk_device.create_semaphore(&vk::SemaphoreCreateInfo::default(), VK_MEMORY_ALLOCATOR).unwrap() };

    //State for freecam controls
    let mut camera = FreeCam::new(glm::vec3(0.0f32, -10.0, 5.0));

    let projection_matrix = glm::perspective(window_size.x as f32 / window_size.y as f32, glm::half_pi(), 0.5, 50.0);

    //Relative to GL clip space, Vulkan has negative Y and half Z.
    let projection_matrix = glm::mat4(
        1.0, 0.0, 0.0, 0.0,
        0.0, -1.0, 0.0, 0.0,
        0.0, 0.0, 0.5, 0.0,
        0.0, 0.0, 0.5, 1.0,
    ) * projection_matrix;

    let mut timer = FrameTimer::new();      //Struct for doing basic framerate independence

    let vk_submission_fence = unsafe { vk_device.create_fence(&vk::FenceCreateInfo::default(), VK_MEMORY_ALLOCATOR).unwrap() };
    
    let mut sphere_width = 8 as u32;
    let mut sphere_height = 8 as u32;
    let mut sphere_spacing = 5.0;
    let mut sphere_amplitude = 3.0;
    let mut sphere_z_offset = 2.0;

    //Main application loop
    'running: loop {
        timer.update(); //Update frame timer

        //Abstracted input variables
        let mut movement_vector: glm::TVec3<f32> = glm::zero();
        let mut orientation_vector: glm::TVec2<f32> = glm::zero();

        //Pump event queue
        let framerate;
        {
            use sdl2::keyboard::{Scancode};
            use sdl2::mouse::MouseButton;

            let imgui_io = imgui_context.io_mut();
            imgui_io.delta_time = timer.delta_time;
            for event in event_pump.poll_iter() {
                match event {
                    Event::Quit{..} => { break 'running; }
                    Event::MouseButtonDown { mouse_btn, .. } => {
                        match mouse_btn {
                            MouseButton::Left => { imgui_io.mouse_down[0] = true; }
                            MouseButton::Right => { imgui_io.mouse_down[1] = true; }
                            _ => {}
                        }
                    }
                    Event::MouseButtonUp { mouse_btn, ..} => {
                        match mouse_btn {
                            MouseButton::Left => { imgui_io.mouse_down[0] = false; }
                            MouseButton::Right => {
                                imgui_io.mouse_down[1] = false;
                                camera.cursor_captured = !camera.cursor_captured;
                                let mouse_util = sdl_ctxt.mouse();
                                mouse_util.set_relative_mouse_mode(camera.cursor_captured);
                                if !camera.cursor_captured {
                                    mouse_util.warp_mouse_in_window(&window, window_size.x as i32 / 2, window_size.y as i32 / 2);
                                }
                            }
                            _ => {}
                        }
                    }
                    Event::MouseMotion { xrel, yrel, .. } => {
                        if camera.cursor_captured {
                            const DAMPENING: f32 = 0.5 / 720.0;
                            orientation_vector += glm::vec2(DAMPENING * xrel as f32, DAMPENING * yrel as f32);
                        }
                    }
                    Event::MouseWheel { y, .. } => {
                        imgui_io.mouse_wheel = y as f32;
                    }
                    _ => {}
                }
            }
            let keyboard_state = event_pump.keyboard_state();
            let mouse_state = event_pump.mouse_state();
            imgui_io.mouse_pos[0] = mouse_state.x() as f32;
            imgui_io.mouse_pos[1] = mouse_state.y() as f32;

            let mut movement_multiplier = 1.0f32;
            if keyboard_state.is_scancode_pressed(Scancode::LShift) {
                movement_multiplier = 15.0;
            }
            if keyboard_state.is_scancode_pressed(Scancode::W) {
                movement_vector += movement_multiplier * glm::vec3(0.0, 0.0, -1.0);
            }
            if keyboard_state.is_scancode_pressed(Scancode::A) {
                movement_vector += movement_multiplier * glm::vec3(-1.0, 0.0, 0.0);
            }
            if keyboard_state.is_scancode_pressed(Scancode::S) {
                movement_vector += movement_multiplier * glm::vec3(0.0, 0.0, 1.0);
            }
            if keyboard_state.is_scancode_pressed(Scancode::D) {
                movement_vector += movement_multiplier * glm::vec3(1.0, 0.0, 0.0);
            }
            if keyboard_state.is_scancode_pressed(Scancode::Q) {
                movement_vector += movement_multiplier * glm::vec3(0.0, -1.0, 0.0);
            }
            if keyboard_state.is_scancode_pressed(Scancode::E) {
                movement_vector += movement_multiplier * glm::vec3(0.0, 1.0, 0.0);
            }

            framerate = imgui_io.framerate;
        }

        //Update
        let imgui_ui = imgui_context.frame();
        imgui_ui.text(format!("Rendering at {:.0} FPS ({:.2} ms frametime)", framerate, 1000.0 / framerate));
        if imgui_ui.button_with_size("Exit", [0.0, 32.0]) {
            break 'running;
        }
        if imgui_ui.button_with_size("Really long button with really long text", [0.0, 32.0]) {
            tfd::message_box_yes_no("The question", "What do you think?", tfd::MessageBoxIcon::Info, tfd::YesNo::Yes);
        }
        
        //Camera orientation based on user input
        camera.orientation += orientation_vector;
        camera.orientation.y = camera.orientation.y.clamp(-glm::half_pi::<f32>(), glm::half_pi::<f32>());
        let view_matrix = camera.make_view_matrix();
        const CAMERA_SPEED: f32 = 3.0;
        let delta_pos = CAMERA_SPEED * glm::affine_inverse(view_matrix) * glm::vec3_to_vec4(&movement_vector) * timer.delta_time;
        camera.position += glm::vec4_to_vec3(&delta_pos);
        let view_projection = projection_matrix * view_matrix;

        let mut transform_ptr = vk_scene_storage_buffer_ptr as *mut f32;
        
        //Update static scene data
        unsafe {
            let clip_from_screen = glm::mat4(
                2.0 / window_size.x as f32, 0.0, 0.0, -1.0,
                0.0, 2.0 / window_size.y as f32, 0.0, -1.0,
                0.0, 0.0, 1.0, 0.0,
                0.0, 0.0, 0.0, 1.0
            );
            ptr::copy_nonoverlapping(clip_from_screen.as_ptr(), transform_ptr, size_of::<glm::TMat4<f32>>());
            transform_ptr = transform_ptr.offset(16);

            let mvp = view_projection * glm::scaling(&glm::vec3(10.0, 10.0, 0.0));
            ptr::copy_nonoverlapping(mvp.as_ptr(), transform_ptr, size_of::<glm::TMat4<f32>>());
            transform_ptr = transform_ptr.offset(16);
        };

        imgui::Slider::new("Sphere width", 1, 150).build(&imgui_ui, &mut sphere_width);
        imgui::Slider::new("Sphere height", 1, 150).build(&imgui_ui, &mut sphere_height);
        imgui::Slider::new("Sphere spacing", 0.0, 20.0).build(&imgui_ui, &mut sphere_spacing);
        imgui::Slider::new("Sphere amplitude", 0.0, 20.0).build(&imgui_ui, &mut sphere_amplitude);
        imgui::Slider::new("Sphere Z offset", 0.0, 20.0).build(&imgui_ui, &mut sphere_z_offset);
        if imgui::Slider::new("Music volume", 0, 128).build(&imgui_ui, &mut music_volume) { Music::set_volume(music_volume); }

        let sphere_count = sphere_width as usize * sphere_height as usize;
        imgui_ui.text(format!("Drawing {} spheres every frame", sphere_count));
        let mut sphere_transforms = vec![0.0; 16 * sphere_count];
        for i in 0..sphere_width {
            for j in 0..sphere_height {
                let sphere_matrix = glm::translation(&glm::vec3(
                    i as f32 * sphere_spacing,
                    j as f32 * sphere_spacing,
                    sphere_z_offset + sphere_amplitude * f32::sin(timer.elapsed_time * (i + 7) as f32) + 5.0)
                ) * glm::rotation(3.0 * timer.elapsed_time, &glm::vec3(0.0, 0.0, 1.0));                        
                let mvp = view_projection * sphere_matrix;

                let trans_offset = i * 16 * sphere_height + j * 16;
                for k in 0..16 {
                    sphere_transforms[(trans_offset + k) as usize] = mvp[k as usize];
                }
            }
        }
        unsafe { ptr::copy_nonoverlapping(sphere_transforms.as_ptr(), transform_ptr, 16 * sphere_count)};

        //Done specifying Dear ImGUI ui for this frame
        let imgui_draw_data = imgui_ui.render();
        
        //Dear ImGUI geometry buffer creation
        let (imgui_geometries, imgui_cmd_lists) = {
            let mut geos = Vec::with_capacity(16);
            let mut cmds = Vec::with_capacity(16);
            imgui_geo_allocator.clear();

            if imgui_draw_data.total_vtx_count > 0 {
                for list in imgui_draw_data.draw_lists() {
                    let vert_size = 8;  //Size in floats
                    let vtx_buffer = list.vtx_buffer();
                    let mut verts = vec![0.0; vtx_buffer.len() * vert_size];

                    let mut current_vertex = 0;
                    for vtx in vtx_buffer.iter() {
                        let idx = current_vertex * vert_size;
                        verts[idx] = vtx.pos[0];
                        verts[idx + 1] = vtx.pos[1];
                        verts[idx + 2] = vtx.uv[0];
                        verts[idx + 3] = vtx.uv[1];
                        verts[idx + 4] = vtx.col[0] as f32 / 255.0;
                        verts[idx + 5] = vtx.col[1] as f32 / 255.0;
                        verts[idx + 6] = vtx.col[2] as f32 / 255.0;
                        verts[idx + 7] = vtx.col[3] as f32 / 255.0;

                        current_vertex += 1;
                    }

                    let idx_buffer = list.idx_buffer();
                    let mut inds = vec![0u32; idx_buffer.len()];
                    for i in 0..idx_buffer.len() {
                        inds[i] = idx_buffer[i] as u32;
                    }

                    let g = imgui_geo_allocator.allocate_geometry(&verts, &inds).unwrap();
                    geos.push(g);

                    let mut cmd_list = Vec::with_capacity(list.commands().count());
                    for command in list.commands() { cmd_list.push(command); }
                    cmds.push(cmd_list);
                }
            }
            (geos, cmds)
        };

        //Draw
        unsafe {
            let begin_info = vk::CommandBufferBeginInfo {
                p_inheritance_info: ptr::null(),
                ..Default::default()
            };
            vk_device.begin_command_buffer(vk_command_buffer, &begin_info).unwrap();
            
            //Set the viewport for this frame
            let viewport = vk::Viewport {
                x: 0.0,
                y: 0.0,
                width: vk_swapchain_extent.width as f32,
                height: vk_swapchain_extent.height as f32,
                min_depth: 0.0,
                max_depth: 1.0
            };
            vk_device.cmd_set_viewport(vk_command_buffer, 0, &[viewport]);

            //Set scissor rect to be same as render area
            vk_device.cmd_set_scissor(vk_command_buffer, 0, &[vk_render_area]);

            let current_framebuffer_index = vk_ext_swapchain.acquire_next_image(vk_swapchain, vk::DeviceSize::MAX, vk_swapchain_semaphore, vk::Fence::null()).unwrap().0 as usize;
            let rp_begin_info = vk::RenderPassBeginInfo {
                render_pass: vk_render_pass,
                framebuffer: vk_swapchain_framebuffers[current_framebuffer_index],
                render_area: vk_render_area,
                clear_value_count: vk_clear_values.len() as u32,
                p_clear_values: vk_clear_values.as_ptr(),
                ..Default::default()
            };
            vk_device.cmd_begin_render_pass(vk_command_buffer, &rp_begin_info, vk::SubpassContents::INLINE);

            //Bind main rendering pipeline to GRAPHICS pipeline bind point
            vk_device.cmd_bind_pipeline(vk_command_buffer, vk::PipelineBindPoint::GRAPHICS, vk_3D_graphics_pipeline);

            //Once per frame descriptor binding
            vk_device.cmd_bind_descriptor_sets(vk_command_buffer, vk::PipelineBindPoint::GRAPHICS, vk_pipeline_layout, 0, &vk_descriptor_sets, &[]);

            //Bind plane's render data
            vk_device.cmd_bind_vertex_buffers(vk_command_buffer, 0, &[g_plane_geometry.vertex_buffer.backing_buffer()], &[g_plane_geometry.vertex_buffer.offset() as u64]);
            vk_device.cmd_bind_index_buffer(vk_command_buffer, g_plane_geometry.index_buffer.backing_buffer(), (g_plane_geometry.index_buffer.offset()) as vk::DeviceSize, vk::IndexType::UINT32);
            vk_device.cmd_draw_indexed(vk_command_buffer, g_plane_geometry.index_count, 1, 0, 0, 0);

            //Bind sphere's render data
            vk_device.cmd_bind_vertex_buffers(vk_command_buffer, 0, &[sphere_geometry.vertex_buffer.backing_buffer()], &[sphere_geometry.vertex_buffer.offset() as u64]);
            vk_device.cmd_bind_index_buffer(vk_command_buffer, sphere_geometry.index_buffer.backing_buffer(), (sphere_geometry.index_buffer.offset()) as vk::DeviceSize, vk::IndexType::UINT32);
            vk_device.cmd_draw_indexed(vk_command_buffer, sphere_geometry.index_count, sphere_count as u32, 0, 0, 1);

            //Record Dear ImGUI drawing commands
            vk_device.cmd_bind_pipeline(vk_command_buffer, vk::PipelineBindPoint::GRAPHICS, imgui_graphics_pipeline);
            for i in 0..imgui_cmd_lists.len() {
                let cmd_list = &imgui_cmd_lists[i];
                for cmd in cmd_list {
                    match cmd {
                        DrawCmd::Elements {count, cmd_params} => {
                            let i_offset = cmd_params.idx_offset;
                            let v_offset = cmd_params.vtx_offset;
                            let v_buffer = imgui_geometries[i].vertex_buffer;
                            let i_buffer = imgui_geometries[i].index_buffer;
                            let tex_id = cmd_params.texture_id.id() as u32;

                            let ext_x = cmd_params.clip_rect[2] - cmd_params.clip_rect[0];
                            let ext_y = cmd_params.clip_rect[3] - cmd_params.clip_rect[1];
                            let scissor_rect = {        
                                let offset = vk::Offset2D {
                                    x: cmd_params.clip_rect[0] as i32,
                                    y: cmd_params.clip_rect[1] as i32
                                };
                                let extent = vk::Extent2D {
                                    width: ext_x as u32,
                                    height: ext_y as u32
                                };
                                vk::Rect2D {
                                    offset,
                                    extent
                                }
                            };
                            vk_device.cmd_set_scissor(vk_command_buffer, 0, &[scissor_rect]);
                            
                            vk_device.cmd_push_constants(vk_command_buffer, vk_pipeline_layout, vk::ShaderStageFlags::FRAGMENT, 0, &tex_id.to_le_bytes());
                            vk_device.cmd_bind_vertex_buffers(vk_command_buffer, 0, &[v_buffer.backing_buffer()], &[v_buffer.offset()]);
                            vk_device.cmd_bind_index_buffer(vk_command_buffer, i_buffer.backing_buffer(), i_buffer.offset(), vk::IndexType::UINT32);
                            vk_device.cmd_draw_indexed(vk_command_buffer, *count as u32, 1, i_offset as u32, v_offset as i32, 0);
                        }
                        DrawCmd::ResetRenderState => { println!("DrawCmd::ResetRenderState."); }
                        DrawCmd::RawCallback {..} => { println!("DrawCmd::RawCallback."); }
                    }
                }
            }

            vk_device.cmd_end_render_pass(vk_command_buffer);

            vk_device.end_command_buffer(vk_command_buffer).unwrap();

            let pipeline_stage_flags = vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT;
            let submit_info = vk::SubmitInfo {
                wait_semaphore_count: 1,
                p_wait_semaphores: [vk_swapchain_semaphore].as_ptr(),
                p_wait_dst_stage_mask: &pipeline_stage_flags,
                command_buffer_count: 1,
                p_command_buffers: &vk_command_buffer,
                ..Default::default()
            };

            let queue = vk_device.get_device_queue(vk_queue_family_index, 0);
            vk_device.queue_submit(queue, &[submit_info], vk_submission_fence).unwrap();

            vk_device.wait_for_fences(&[vk_submission_fence], true, vk::DeviceSize::MAX).unwrap();
            vk_device.reset_fences(&[vk_submission_fence]).unwrap();

            let present_info = vk::PresentInfoKHR {
                swapchain_count: 1,
                p_swapchains: &vk_swapchain,
                p_image_indices: &(current_framebuffer_index as u32),
                ..Default::default()
            };
            vk_ext_swapchain.queue_present(queue, &present_info).unwrap();
        }
    }
}
