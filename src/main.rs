#![allow(non_snake_case)]
extern crate nalgebra_glm as glm;
extern crate ozy_engine as ozy;
extern crate tinyfiledialogs as tfd;

mod vkutil;
mod structs;

use ash::vk;
use ash::vk::Handle;
use imgui::{DrawCmd, FontAtlasRefMut};
use sdl2::event::Event;
use sdl2::controller::GameController;
use sdl2::mixer;
use sdl2::mixer::Music;
use structs::FreeCam;
use std::fmt::Display;
use std::fs::{File};
use std::ffi::CStr;
use std::mem::size_of;
use std::ptr;

use ozy::io::OzyMesh;
use ozy::structs::{FrameTimer, OptionVec};

const COMPONENT_MAPPING_DEFAULT: vk::ComponentMapping = vk::ComponentMapping {
    r: vk::ComponentSwizzle::R,
    g: vk::ComponentSwizzle::G,
    b: vk::ComponentSwizzle::B,
    a: vk::ComponentSwizzle::A,
};

const VK_MEMORY_ALLOCATOR: Option<&vk::AllocationCallbacks> = None;

fn crash_with_error_dialog(message: &str) -> ! {
    tfd::message_box_ok("Oops...", &message.replace("'", ""), tfd::MessageBoxIcon::Error);
    panic!("{}", message);
}

fn unwrap_result<T, E: Display>(res: Result<T, E>) -> T {
    match res {
        Ok(t) => { t }
        Err(e) => {
            crash_with_error_dialog(&format!("{}", e));
        }
    }
}

//Entry point
fn main() {
    //Create the window using SDL
    let sdl_ctxt = unwrap_result(sdl2::init());
    let mut event_pump = unwrap_result(sdl_ctxt.event_pump());
    let video_subsystem = unwrap_result(sdl_ctxt.video());
    let controller_subsystem = unwrap_result(sdl_ctxt.game_controller());
    let mut window_size = glm::vec2(1920, 1080);
    let window = video_subsystem.window("Vulkan't", window_size.x, window_size.y).position_centered().resizable().vulkan().build().unwrap();

    //Initialize the SDL mixer
    let mut music_volume = 16;
    let _sdl_mixer = mixer::init(mixer::InitFlag::FLAC | mixer::InitFlag::MP3).unwrap();
    mixer::open_audio(mixer::DEFAULT_FREQUENCY, mixer::DEFAULT_FORMAT, 2, 256).unwrap();
    Music::set_volume(music_volume);

    //Initialize Dear ImGUI
    let mut imgui_context = imgui::Context::create();
    {
        imgui_context.style_mut().use_dark_colors();
        let io = imgui_context.io_mut();
        io.display_size[0] = window_size.x as f32;
        io.display_size[1] = window_size.y as f32;
    }

    //Initialize the Vulkan API
    let vk = vkutil::VulkanAPI::initialize(&window);

    //Create command buffer
    let vk_command_buffer = unsafe {
        let pool_create_info = vk::CommandPoolCreateInfo {
            queue_family_index: vk.queue_family_index,
            flags: vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
            ..Default::default()
        };

        let command_pool = vk.device.create_command_pool(&pool_create_info, VK_MEMORY_ALLOCATOR).unwrap();

        let command_buffer_alloc_info = vk::CommandBufferAllocateInfo {
            command_pool,
            command_buffer_count: 1,
            level: vk::CommandBufferLevel::PRIMARY,
            ..Default::default()
        };
        vk.device.allocate_command_buffers(&command_buffer_alloc_info).unwrap()[0]
    };

    //Create texture samplers
    let (material_sampler, font_sampler) = unsafe {
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
        let mat = vk.device.create_sampler(&sampler_info, VK_MEMORY_ALLOCATOR).unwrap();
        
        let sampler_info = vk::SamplerCreateInfo {
            min_filter: vk::Filter::NEAREST,
            mag_filter: vk::Filter::NEAREST,
            mipmap_mode: vk::SamplerMipmapMode::NEAREST,
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
        let font = vk.device.create_sampler(&sampler_info, VK_MEMORY_ALLOCATOR).unwrap();
        
        (mat, font)
    };

    //Maintain free list for texture allocation
    let global_texture_slots = 1024;
    let mut global_texture_free_list = OptionVec::with_capacity(global_texture_slots);
    let mut global_texture_update;
    let default_texture_sampler;

    //Load grass billboard texture
    let grass_billboard_global_index = unsafe {
        let image = vkutil::load_bc7_texture(
            &vk,
            vk_command_buffer,
            "./data/textures/billboard_grass.dds"
        );

        let sampler_subresource_range = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1
        };
        let grass_view_info = vk::ImageViewCreateInfo {
            image: image,
            format: vk::Format::BC7_SRGB_BLOCK,
            view_type: vk::ImageViewType::TYPE_2D,
            components: COMPONENT_MAPPING_DEFAULT,
            subresource_range: sampler_subresource_range,
            ..Default::default()
        };
        let view = vk.device.create_image_view(&grass_view_info, VK_MEMORY_ALLOCATOR).unwrap();

        let descriptor_info = vk::DescriptorImageInfo {
            sampler: material_sampler,
            image_view: view,
            image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
        };
        let index = global_texture_free_list.insert(descriptor_info);
        global_texture_update = true;

        index as u32
    };

    //Load steel plate texture
    let steel_plate_global_index = unsafe {
        let image = vkutil::load_bc7_texture(
            &vk,
            vk_command_buffer,
            "./data/textures/steel_plate/albedo.dds"
        );

        let sampler_subresource_range = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 10,
            base_array_layer: 0,
            layer_count: 1
        };
        let grass_view_info = vk::ImageViewCreateInfo {
            image: image,
            format: vk::Format::BC7_SRGB_BLOCK,
            view_type: vk::ImageViewType::TYPE_2D,
            components: COMPONENT_MAPPING_DEFAULT,
            subresource_range: sampler_subresource_range,
            ..Default::default()
        };
        let view = vk.device.create_image_view(&grass_view_info, VK_MEMORY_ALLOCATOR).unwrap();

        let descriptor_info = vk::DescriptorImageInfo {
            sampler: material_sampler,
            image_view: view,
            image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
        };
        let index = global_texture_free_list.insert(descriptor_info);
        global_texture_update = true;

        index as u32
    };

    //Create and upload Dear IMGUI font atlas
    let imgui_font_global_index = match imgui_context.fonts() {
        FontAtlasRefMut::Owned(atlas) => unsafe {
            let atlas_texture = atlas.build_alpha8_texture();

            let size = (atlas_texture.width * atlas_texture.height) as vk::DeviceSize;
            let buffer_create_info = vk::BufferCreateInfo {
                usage: vk::BufferUsageFlags::TRANSFER_SRC,
                size,
                sharing_mode: vk::SharingMode::EXCLUSIVE,
                ..Default::default()
            };
            let staging_buffer = vk.device.create_buffer(&buffer_create_info, VK_MEMORY_ALLOCATOR).unwrap();            
            let staging_buffer_memory = vkutil::allocate_buffer_memory(&vk, staging_buffer);    
            vk.device.bind_buffer_memory(staging_buffer, staging_buffer_memory, 0).unwrap();

            let staging_ptr = vk.device.map_memory(staging_buffer_memory, 0, size, vk::MemoryMapFlags::empty()).unwrap();
            ptr::copy_nonoverlapping(atlas_texture.data.as_ptr(), staging_ptr as *mut _, size as usize);
            vk.device.unmap_memory(staging_buffer_memory);
            
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
                p_queue_family_indices: &vk.queue_family_index,
                initial_layout: vk::ImageLayout::UNDEFINED,
                ..Default::default()
            };
            let vk_font_image = vk.device.create_image(&font_create_info, VK_MEMORY_ALLOCATOR).unwrap();
            
            let font_image_memory = vkutil::allocate_image_memory(&vk, vk_font_image);
    
            vk.device.bind_image_memory(vk_font_image, font_image_memory, 0).unwrap();

            vk.device.begin_command_buffer(vk_command_buffer, &vk::CommandBufferBeginInfo::default()).unwrap();

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
            vk.device.cmd_pipeline_barrier(vk_command_buffer, vk::PipelineStageFlags::TOP_OF_PIPE, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[], &[], &[image_memory_barrier]);

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
            vk.device.cmd_copy_buffer_to_image(vk_command_buffer, staging_buffer, vk_font_image, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &[copy_region]);
            
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
            vk.device.cmd_pipeline_barrier(vk_command_buffer, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::FRAGMENT_SHADER, vk::DependencyFlags::empty(), &[], &[], &[image_memory_barrier]);


            vk.device.end_command_buffer(vk_command_buffer).unwrap();

            
            let submit_info = vk::SubmitInfo {
                command_buffer_count: 1,
                p_command_buffers: &vk_command_buffer,
                ..Default::default()
            };

            let fence = vk.device.create_fence(&vk::FenceCreateInfo::default(), VK_MEMORY_ALLOCATOR).unwrap();
            let queue = vk.device.get_device_queue(vk.queue_family_index, 0);
            vk.device.queue_submit(queue, &[submit_info], fence).unwrap();
            vk.device.wait_for_fences(&[fence], true, vk::DeviceSize::MAX).unwrap();
            vk.device.destroy_fence(fence, VK_MEMORY_ALLOCATOR);
            vk.device.destroy_buffer(staging_buffer, VK_MEMORY_ALLOCATOR);
            atlas.clear_tex_data();  //Free atlas memory CPU-side

            //Then create the image view
            let sampler_subresource_range = vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1
            };
            let font_view_info = vk::ImageViewCreateInfo {
                image: vk_font_image,
                format: vk::Format::R8_UNORM,
                view_type: vk::ImageViewType::TYPE_2D,
                components: COMPONENT_MAPPING_DEFAULT,
                subresource_range: sampler_subresource_range,
                ..Default::default()
            };
            let font_view = vk.device.create_image_view(&font_view_info, VK_MEMORY_ALLOCATOR).unwrap();
            
            let image_info = vk::DescriptorImageInfo {
                sampler: font_sampler,
                image_view: font_view,
                image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
            };
            let sampler_index = global_texture_free_list.insert(image_info);
            global_texture_update = true;
            
            default_texture_sampler = image_info;
            atlas.tex_id = imgui::TextureId::new(sampler_index);    //Giving Dear Imgui a reference to the font atlas GPU texture
            sampler_index as u32
        }
        FontAtlasRefMut::Shared(_) => {
            panic!("Not dealing with this case.");
        }
    };

    //Create swapchain extension object
    let vk_ext_swapchain = ash::extensions::khr::Swapchain::new(&vk.instance, &vk.device);

    //Search for an SRGB swapchain format    
    let surf_formats = unsafe { vk.ext_surface.get_physical_device_surface_formats(vk.physical_device, vk.surface).unwrap() };
    let mut vk_surface_format = vk::SurfaceFormatKHR::default();
    for sformat in surf_formats.iter() {
        if sformat.format == vk::Format::B8G8R8A8_SRGB {
            vk_surface_format = *sformat;
            break;
        }
        vk_surface_format = vk::SurfaceFormatKHR::default();
    }
    
    let vk_render_pass = unsafe {
        let color_attachment_description = vk::AttachmentDescription {
            format: vk_surface_format.format,
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
            format: vk::Format::D24_UNORM_S8_UINT,
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
        vk.device.create_render_pass(&renderpass_info, VK_MEMORY_ALLOCATOR).unwrap()
    };

    //Create the main swapchain for window present
    let mut vk_display = vkutil::Display::initialize_swapchain(&vk, &vk_ext_swapchain, vk_render_pass);

    let global_transform_slots = 1024 * 1024;
    let vk_uniform_buffer_size;
    let vk_transform_storage_buffer;
    let vk_scene_storage_buffer_ptr = unsafe {
        let mut buffer_size = (size_of::<glm::TMat4<f32>>() * global_transform_slots) as vk::DeviceSize;
        let alignment = vk.physical_device_properties.limits.min_uniform_buffer_offset_alignment;
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

        vk_transform_storage_buffer = vk.device.create_buffer(&buffer_create_info, VK_MEMORY_ALLOCATOR).unwrap();
        
        let transform_buffer_memory = vkutil::allocate_buffer_memory(&vk, vk_transform_storage_buffer);

        vk.device.bind_buffer_memory(vk_transform_storage_buffer, transform_buffer_memory, 0).unwrap();

        let uniform_ptr = vk.device.map_memory(transform_buffer_memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty()).unwrap();
        uniform_ptr
    };

    let vk_descriptor_set_layout;
    let push_constant_shader_stage_flags = vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT;
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
            descriptor_count: global_texture_slots as u32,
            stage_flags: vk::ShaderStageFlags::FRAGMENT,
            ..Default::default()
        };
        
        let bindings = [storage_binding, texture_binding];
        
        let descriptor_layout = vk::DescriptorSetLayoutCreateInfo {
            binding_count: bindings.len() as u32,
            p_bindings: bindings.as_ptr(),
            ..Default::default()
        };

        vk_descriptor_set_layout = vk.device.create_descriptor_set_layout(&descriptor_layout, VK_MEMORY_ALLOCATOR).unwrap();

        let push_constant_range = vk::PushConstantRange {
            stage_flags: push_constant_shader_stage_flags,
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
        
        vk.device.create_pipeline_layout(&pipeline_layout_createinfo, VK_MEMORY_ALLOCATOR).unwrap()
    };
    
    //Set up descriptors
    let vk_descriptor_sets = unsafe {
        let storage_pool_size = vk::DescriptorPoolSize {
            ty: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: 1
        };
        let sampler_pool_size = vk::DescriptorPoolSize {
            ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            descriptor_count: global_texture_slots as u32,
        };

        let pool_sizes = [storage_pool_size, sampler_pool_size];
        let descriptor_pool_info = vk::DescriptorPoolCreateInfo {
            max_sets: 1,
            pool_size_count: pool_sizes.len() as u32,
            p_pool_sizes: pool_sizes.as_ptr(),
            ..Default::default()
        };
        let descriptor_pool = vk.device.create_descriptor_pool(&descriptor_pool_info, VK_MEMORY_ALLOCATOR).unwrap();

        let vk_alloc_info = vk::DescriptorSetAllocateInfo {
            descriptor_pool,
            descriptor_set_count: 1,
            p_set_layouts: &vk_descriptor_set_layout,
            ..Default::default()
        };
        vk.device.allocate_descriptor_sets(&vk_alloc_info).unwrap()
    };

    //Write initial values to descriptors
    unsafe {
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
        vk.device.update_descriptor_sets(&[storage_write], &[]);
    }

    

    //Load shaders
    let vk_shader_stages = unsafe {
        let v = vkutil::load_shader_stage(&vk.device, vk::ShaderStageFlags::VERTEX, "./shaders/main_vert.spv");
        let f = vkutil::load_shader_stage(&vk.device, vk::ShaderStageFlags::FRAGMENT, "./shaders/main_frag.spv");
        [v, f]
    };

    let imgui_shader_stages = unsafe {
        let v = vkutil::load_shader_stage(&vk.device, vk::ShaderStageFlags::VERTEX, "./shaders/imgui_vert.spv");
        let f = vkutil::load_shader_stage(&vk.device, vk::ShaderStageFlags::FRAGMENT, "./shaders/imgui_frag.spv");
        [v, f]
    };

    //Create graphics pipelines
    let [vk_3D_graphics_pipeline, imgui_graphics_pipeline, vk_wireframe_graphics_pipeline] = unsafe {
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

        let wire_raster_state = vk::PipelineRasterizationStateCreateInfo {
            polygon_mode: vk::PolygonMode::LINE,
            ..rasterization_state
        };
        let wireframe_pipeline_info = vk::GraphicsPipelineCreateInfo {
            p_rasterization_state: &wire_raster_state,
            ..graphics_pipeline_info
        };

        let pipeline_infos = [graphics_pipeline_info, imgui_pipeline_info, wireframe_pipeline_info];
        let pipelines = vk.device.create_graphics_pipelines(vk::PipelineCache::null(), &pipeline_infos, VK_MEMORY_ALLOCATOR).unwrap();
        [pipelines[0], pipelines[1], pipelines[2]]
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
            let buffer = vk.device.create_buffer(&buffer_create_info, VK_MEMORY_ALLOCATOR).unwrap();
            buffer
        };

        let buffer_memory = vkutil::allocate_buffer_memory(&vk, scene_geo_buffer);

        //Bind buffer
        vk.device.bind_buffer_memory(scene_geo_buffer, buffer_memory, 0).unwrap();

        //Map buffer to host memory
        let buffer_ptr = vk.device.map_memory(
            buffer_memory,
            0,
            vk::WHOLE_SIZE,
            vk::MemoryMapFlags::empty()
        ).unwrap();

        //Create virtual bump allocator
        let mut scene_geo_allocator = vkutil::VirtualBumpAllocator::new(scene_geo_buffer, buffer_ptr, scene_geo_buffer_size.try_into().unwrap());

        g_plane_geometry = scene_geo_allocator.allocate_geometry(&g_plane_vertices, &g_plane_indices).unwrap();
        sphere_geometry = scene_geo_allocator.allocate_geometry(&uv_sphere.vertex_array.vertices, &uv_sphere_indices).unwrap();

        vk.device.unmap_memory(buffer_memory);
    }

    let mut imgui_geo_allocator = unsafe {
        let imgui_buffer_size = 1024 * 64;
        let buffer_create_info = vk::BufferCreateInfo {
            usage: vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::INDEX_BUFFER,
            size: imgui_buffer_size as vk::DeviceSize,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            ..Default::default()
        };
        let buffer = vk.device.create_buffer(&buffer_create_info, VK_MEMORY_ALLOCATOR).unwrap();
        let buffer_memory = vkutil::allocate_buffer_memory(&vk, buffer);
        vk.device.bind_buffer_memory(buffer, buffer_memory, 0).unwrap();

        let ptr = vk.device.map_memory(buffer_memory, 0, imgui_buffer_size, vk::MemoryMapFlags::empty()).unwrap();

        vkutil::VirtualBumpAllocator::new(buffer, ptr, imgui_buffer_size)
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
    let vk_swapchain_semaphore = unsafe { vk.device.create_semaphore(&vk::SemaphoreCreateInfo::default(), VK_MEMORY_ALLOCATOR).unwrap() };

    //State for freecam controls
    let mut camera = FreeCam::new(glm::vec3(0.0f32, -10.0, 5.0));

    let mut timer = FrameTimer::new();      //Struct for doing basic framerate independence

    let vk_submission_fence = unsafe { vk.device.create_fence(&vk::FenceCreateInfo::default(), VK_MEMORY_ALLOCATOR).unwrap() };
    
    let mut sphere_width = 10 as u32;
    let mut sphere_height = 8 as u32;
    let mut sphere_spacing = 5.0;
    let mut sphere_amplitude = 3.0;
    let mut sphere_z_offset = 2.0;
    let mut sphere_rotation = 3.0;
    
    //Load and play bgm
    let bgm = unwrap_result(Music::from_file("./data/music/relaxing_botw.mp3"));
    bgm.play(-1).unwrap();

    let mut wireframe = false;
    let mut main_pipeline = vk_3D_graphics_pipeline;

    let mut game_controllers = [None, None, None, None];

    //Main application loop
    'running: loop {
        timer.update(); //Update frame timer

        //Abstracted input variables
        let mut movement_multiplier = 1.0f32;
        let mut movement_vector: glm::TVec3<f32> = glm::zero();
        let mut orientation_vector: glm::TVec2<f32> = glm::zero();

        let framerate;
        {
            use sdl2::event::WindowEvent;
            use sdl2::keyboard::{Scancode};
            use sdl2::mouse::MouseButton;

            //Sync controller array with how many controllers are actually connected
            for i in 0..game_controllers.len() {
                match &mut game_controllers[i] {
                    None => {
                        if i < unwrap_result(controller_subsystem.num_joysticks()) as usize {
                            let controller = unwrap_result(controller_subsystem.open(i as u32));
                            game_controllers[i] = Some(controller);
                        }
                    }
                    Some(controller) => {
                        if !controller.attached() {
                            game_controllers[i] = None;
                        }
                    }
                }
            }

            let imgui_io = imgui_context.io_mut();
            imgui_io.delta_time = timer.delta_time;
            
            //Pump event queue
            for event in event_pump.poll_iter() {
                match event {
                    Event::Quit{..} => { break 'running; }
                    Event::Window { win_event, .. } => {
                        match win_event {
                            WindowEvent::Resized(x, y) => unsafe {
                                //Free the now-invalid swapchain data
                                for framebuffer in vk_display.swapchain_framebuffers {
                                    vk.device.destroy_framebuffer(framebuffer, VK_MEMORY_ALLOCATOR);
                                }
                                for view in vk_display.swapchain_image_views {
                                    vk.device.destroy_image_view(view, VK_MEMORY_ALLOCATOR);
                                }
                                vk.device.destroy_image_view(vk_display.depth_image_view, VK_MEMORY_ALLOCATOR);
                                vk_ext_swapchain.destroy_swapchain(vk_display.swapchain, VK_MEMORY_ALLOCATOR);

                                //Recreate swapchain managing struct
                                vk_display = vkutil::Display::initialize_swapchain(&vk, &vk_ext_swapchain, vk_render_pass);

                                window_size = glm::vec2(vk_display.extent.width, vk_display.extent.height);
                                imgui_io.display_size[0] = window_size.x as f32;
                                imgui_io.display_size[1] = window_size.y as f32;
                            }
                            _ => {}
                        }
                    }
                    Event::MouseButtonUp { mouse_btn, ..} => {
                        match mouse_btn {
                            MouseButton::Right => {
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
                            const DAMPENING: f32 = 0.25 / 360.0;
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
            imgui_io.mouse_down = [mouse_state.left(), mouse_state.right(), mouse_state.middle(), false, false];
            imgui_io.mouse_pos[0] = mouse_state.x() as f32;
            imgui_io.mouse_pos[1] = mouse_state.y() as f32;

            if let Some(controller) = &mut game_controllers[0] {
                use sdl2::controller::{Axis, Button};
                if controller.button(Button::A) {
                    println!("Pressing A");
                    unwrap_result(controller.set_rumble(0xFFFF, 0xFFFF, 30));
                }

                let left_trigger = controller.axis(Axis::TriggerLeft) as f32 / i16::MAX as f32;
                movement_multiplier = 9.0 * left_trigger + 1.0;

                const JOYSTICK_DEADZONE: f32 = 0.15;
                let left_joy_vector = {
                    let x = controller.axis(Axis::LeftX) as f32 / i16::MAX as f32;
                    let y = controller.axis(Axis::LeftY) as f32 / i16::MAX as f32;
                    let mut res = glm::vec3(x, -y, 0.0);
                    if glm::length(&res) < JOYSTICK_DEADZONE {
                        res = glm::zero();
                    }
                    res
                };
                let right_joy_vector = {
                    let x = controller.axis(Axis::RightX) as f32 / i16::MAX as f32;
                    let y = controller.axis(Axis::RightY) as f32 / i16::MAX as f32;
                    let mut res = glm::vec2(x, -y);
                    if glm::length(&res) < JOYSTICK_DEADZONE {
                        res = glm::zero();
                    }
                    res
                };

                movement_vector += 5.0 * &left_joy_vector;
                orientation_vector += 4.0 * timer.delta_time * glm::vec2(right_joy_vector.x, -right_joy_vector.y);
            }

            if keyboard_state.is_scancode_pressed(Scancode::LShift) {
                movement_multiplier *= 15.0;
            }
            if keyboard_state.is_scancode_pressed(Scancode::LCtrl) {
                movement_multiplier *= 0.25;
            }
            if keyboard_state.is_scancode_pressed(Scancode::W) {
                movement_vector += glm::vec3(0.0, 1.0, 0.0);
            }
            if keyboard_state.is_scancode_pressed(Scancode::A) {
                movement_vector += glm::vec3(-1.0, 0.0, 0.0);
            }
            if keyboard_state.is_scancode_pressed(Scancode::S) {
                movement_vector += glm::vec3(0.0, -1.0, 0.0);
            }
            if keyboard_state.is_scancode_pressed(Scancode::D) {
                movement_vector += glm::vec3(1.0, 0.0, 0.0);
            }
            if keyboard_state.is_scancode_pressed(Scancode::Q) {
                movement_vector += glm::vec3(0.0, 0.0, -1.0);
            }
            if keyboard_state.is_scancode_pressed(Scancode::E) {
                movement_vector += glm::vec3(0.0, 0.0, 1.0);
            }

            framerate = imgui_io.framerate;
        }

        //Update
        movement_vector *= movement_multiplier;

        let imgui_ui = imgui_context.frame();
        imgui_ui.text(format!("Rendering at {:.0} FPS ({:.2} ms frametime)", framerate, 1000.0 / framerate));

        let (message, color) =  if let Some(_) = &game_controllers[0] {
            ("Controller is connected.", [0.0, 1.0, 0.0, 1.0])
        } else {
            ("Controller is not connected.", [1.0, 0.0, 0.0, 1.0])
        };

        let color_token = imgui_ui.push_style_color(imgui::StyleColor::Text, color);
        imgui_ui.text(message);
        color_token.pop();

        if imgui_ui.button_with_size("Really long button with really long text", [0.0, 32.0]) {
            tfd::message_box_yes_no("The question", "What do you think?", tfd::MessageBoxIcon::Info, tfd::YesNo::Yes);
        }

        //Camera orientation based on user input
        camera.orientation += orientation_vector;
        camera.orientation.y = camera.orientation.y.clamp(-glm::half_pi::<f32>(), glm::half_pi::<f32>());
        let view_matrix = camera.make_view_matrix();

        const CAMERA_SPEED: f32 = 3.0;
        let view_movement_vector = glm::mat4(
            1.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, -1.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 1.0
        ) * glm::vec3_to_vec4(&movement_vector);
        let view_movement_vector = glm::vec4_to_vec3(&view_movement_vector);
        let delta_pos = CAMERA_SPEED * glm::affine_inverse(view_matrix) * glm::vec3_to_vec4(&view_movement_vector) * timer.delta_time;
        camera.position += glm::vec4_to_vec3(&delta_pos);

        let projection_matrix = glm::perspective(window_size.x as f32 / window_size.y as f32, glm::half_pi(), 0.2, 50.0);

        //Relative to GL clip space, Vulkan has negative Y and half Z.
        let projection_matrix = glm::mat4(
            1.0, 0.0, 0.0, 0.0,
            0.0, -1.0, 0.0, 0.0,
            0.0, 0.0, 0.5, 0.0,
            0.0, 0.0, 0.5, 1.0,
        ) * projection_matrix;

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

            let mvp = view_projection * glm::scaling(&glm::vec3(10.0, 10.0, 1.0));
            ptr::copy_nonoverlapping(mvp.as_ptr(), transform_ptr, size_of::<glm::TMat4<f32>>());
            transform_ptr = transform_ptr.offset(16);
        };

        imgui::Slider::new("Sphere width", 1, 150).build(&imgui_ui, &mut sphere_width);
        imgui::Slider::new("Sphere height", 1, 150).build(&imgui_ui, &mut sphere_height);
        imgui::Slider::new("Sphere spacing", 0.0, 20.0).build(&imgui_ui, &mut sphere_spacing);
        imgui::Slider::new("Sphere amplitude", 0.0, 20.0).build(&imgui_ui, &mut sphere_amplitude);
        imgui::Slider::new("Sphere rotation speed", 0.0, 20.0).build(&imgui_ui, &mut sphere_rotation);
        imgui::Slider::new("Sphere Z offset", 0.0, 20.0).build(&imgui_ui, &mut sphere_z_offset);
        if imgui::Slider::new("Music volume", 0, 128).build(&imgui_ui, &mut music_volume) { Music::set_volume(music_volume); }
        if imgui_ui.checkbox("Wireframe view", &mut wireframe) {
            if !wireframe {
                main_pipeline = vk_3D_graphics_pipeline;
            } else {
                main_pipeline = vk_wireframe_graphics_pipeline;
            }
        }

        let sphere_count = sphere_width as usize * sphere_height as usize;
        imgui_ui.text(format!("Drawing {} spheres every frame", sphere_count));
        if imgui_ui.button_with_size("Exit", [0.0, 32.0]) {
            break 'running;
        }

        let mut sphere_transforms = vec![0.0; 16 * sphere_count];
        for i in 0..sphere_width {
            for j in 0..sphere_height {
                let sphere_matrix = glm::translation(&glm::vec3(
                    i as f32 * sphere_spacing,
                    j as f32 * sphere_spacing,
                    sphere_z_offset + sphere_amplitude * f32::sin(timer.elapsed_time * (i + 7) as f32) + 5.0)
                ) * glm::rotation(sphere_rotation * timer.elapsed_time, &glm::vec3(0.0, 0.0, 1.0));                        
                let mvp = view_projection * sphere_matrix;

                let trans_offset = i * 16 * sphere_height + j * 16;
                for k in 0..16 {
                    sphere_transforms[(trans_offset + k) as usize] = mvp[k as usize];
                }
            }
        }
        unsafe { ptr::copy_nonoverlapping(sphere_transforms.as_ptr(), transform_ptr, 16 * sphere_count)};

        //Pre-render phase

        //Update bindless texture sampler descriptors
        if global_texture_update {
            global_texture_update = false;

            let mut image_infos = vec![default_texture_sampler; global_texture_slots];
            for i in 0..global_texture_free_list.len() {
                if let Some(info) = global_texture_free_list[i] {
                    image_infos[i] = info;
                }
            }

            let sampler_write = vk::WriteDescriptorSet {
                dst_set: vk_descriptor_sets[0],
                descriptor_count: global_texture_slots as u32,
                descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                p_image_info: image_infos.as_ptr(),
                dst_array_element: 0,
                dst_binding: 1,
                ..Default::default()
            };
            unsafe { vk.device.update_descriptor_sets(&[sampler_write], &[]); }
        }

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
                        verts[idx] =     vtx.pos[0];
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
            vk.device.begin_command_buffer(vk_command_buffer, &begin_info).unwrap();
            
            //Set the viewport for this frame
            let viewport = vk::Viewport {
                x: 0.0,
                y: 0.0,
                width: vk_display.extent.width as f32,
                height: vk_display.extent.height as f32,
                min_depth: 0.0,
                max_depth: 1.0
            };
            vk.device.cmd_set_viewport(vk_command_buffer, 0, &[viewport]);

            //Set scissor rect to be same as render area
            let vk_render_area = {
                let offset = vk::Offset2D {
                    x: 0,
                    y: 0
                };
                vk::Rect2D {
                    offset,
                    extent: vk_display.extent
                }
            };
            vk.device.cmd_set_scissor(vk_command_buffer, 0, &[vk_render_area]);

            let current_framebuffer_index = vk_ext_swapchain.acquire_next_image(vk_display.swapchain, vk::DeviceSize::MAX, vk_swapchain_semaphore, vk::Fence::null()).unwrap().0 as usize;
            let rp_begin_info = vk::RenderPassBeginInfo {
                render_pass: vk_render_pass,
                framebuffer: vk_display.swapchain_framebuffers[current_framebuffer_index],
                render_area: vk_render_area,
                clear_value_count: vk_clear_values.len() as u32,
                p_clear_values: vk_clear_values.as_ptr(),
                ..Default::default()
            };
            vk.device.cmd_begin_render_pass(vk_command_buffer, &rp_begin_info, vk::SubpassContents::INLINE);

            //Bind main rendering pipeline to GRAPHICS pipeline bind point
            vk.device.cmd_bind_pipeline(vk_command_buffer, vk::PipelineBindPoint::GRAPHICS, main_pipeline);

            //Once per frame descriptor binding
            vk.device.cmd_bind_descriptor_sets(vk_command_buffer, vk::PipelineBindPoint::GRAPHICS, vk_pipeline_layout, 0, &vk_descriptor_sets, &[]);

            //Bind plane's render data
            vk.device.cmd_push_constants(vk_command_buffer, vk_pipeline_layout, push_constant_shader_stage_flags, 0, &steel_plate_global_index.to_le_bytes());
            vk.device.cmd_bind_vertex_buffers(vk_command_buffer, 0, &[g_plane_geometry.vertex_buffer.backing_buffer()], &[g_plane_geometry.vertex_buffer.offset() as u64]);
            vk.device.cmd_bind_index_buffer(vk_command_buffer, g_plane_geometry.index_buffer.backing_buffer(), (g_plane_geometry.index_buffer.offset()) as vk::DeviceSize, vk::IndexType::UINT32);
            vk.device.cmd_draw_indexed(vk_command_buffer, g_plane_geometry.index_count, 1, 0, 0, 0);

            //Bind sphere's render data
            vk.device.cmd_push_constants(vk_command_buffer, vk_pipeline_layout, push_constant_shader_stage_flags, 0, &grass_billboard_global_index.to_le_bytes());
            vk.device.cmd_bind_vertex_buffers(vk_command_buffer, 0, &[sphere_geometry.vertex_buffer.backing_buffer()], &[sphere_geometry.vertex_buffer.offset() as u64]);
            vk.device.cmd_bind_index_buffer(vk_command_buffer, sphere_geometry.index_buffer.backing_buffer(), (sphere_geometry.index_buffer.offset()) as vk::DeviceSize, vk::IndexType::UINT32);
            vk.device.cmd_draw_indexed(vk_command_buffer, sphere_geometry.index_count, sphere_count as u32, 0, 0, 1);

            //Record Dear ImGUI drawing commands
            vk.device.cmd_bind_pipeline(vk_command_buffer, vk::PipelineBindPoint::GRAPHICS, imgui_graphics_pipeline);
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
                            vk.device.cmd_set_scissor(vk_command_buffer, 0, &[scissor_rect]);
                            
                            vk.device.cmd_push_constants(vk_command_buffer, vk_pipeline_layout, push_constant_shader_stage_flags, 0, &tex_id.to_le_bytes());
                            vk.device.cmd_bind_vertex_buffers(vk_command_buffer, 0, &[v_buffer.backing_buffer()], &[v_buffer.offset()]);
                            vk.device.cmd_bind_index_buffer(vk_command_buffer, i_buffer.backing_buffer(), i_buffer.offset(), vk::IndexType::UINT32);
                            vk.device.cmd_draw_indexed(vk_command_buffer, *count as u32, 1, i_offset as u32, v_offset as i32, 0);
                        }
                        DrawCmd::ResetRenderState => { println!("DrawCmd::ResetRenderState."); }
                        DrawCmd::RawCallback {..} => { println!("DrawCmd::RawCallback."); }
                    }
                }
            }

            vk.device.cmd_end_render_pass(vk_command_buffer);

            vk.device.end_command_buffer(vk_command_buffer).unwrap();

            let pipeline_stage_flags = vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT;
            let submit_info = vk::SubmitInfo {
                wait_semaphore_count: 1,
                p_wait_semaphores: [vk_swapchain_semaphore].as_ptr(),
                p_wait_dst_stage_mask: &pipeline_stage_flags,
                command_buffer_count: 1,
                p_command_buffers: &vk_command_buffer,
                ..Default::default()
            };

            let queue = vk.device.get_device_queue(vk.queue_family_index, 0);
            vk.device.queue_submit(queue, &[submit_info], vk_submission_fence).unwrap();

            vk.device.wait_for_fences(&[vk_submission_fence], true, vk::DeviceSize::MAX).unwrap();
            vk.device.reset_fences(&[vk_submission_fence]).unwrap();

            let present_info = vk::PresentInfoKHR {
                swapchain_count: 1,
                p_swapchains: &vk_display.swapchain,
                p_image_indices: &(current_framebuffer_index as u32),
                ..Default::default()
            };
            unwrap_result(vk_ext_swapchain.queue_present(queue, &present_info));
        }
    }
}
