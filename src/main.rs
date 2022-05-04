#![allow(non_snake_case)]
extern crate nalgebra_glm as glm;
extern crate ozy_engine as ozy;
extern crate tinyfiledialogs as tfd;

#[macro_use]
mod vkutil;
mod structs;

use ash::vk;
use imgui::{DrawCmd, FontAtlasRefMut};
use sdl2::controller::GameController;
use sdl2::event::Event;
use sdl2::mixer;
use sdl2::mixer::Music;
use std::fmt::Display;
use std::fs::{File};
use std::ffi::CStr;
use std::mem::size_of;
use std::ptr;
use std::time::SystemTime;

use ozy::io::OzyMesh;
use ozy::structs::{FrameTimer, OptionVec};

use vkutil::{ColorSpace, FreeList, Material, VirtualBuffer, VirtualImage, VulkanAPI};
use structs::{FreeCam, NoiseParameters, TerrainSpec};

fn crash_with_error_dialog(message: &str) -> ! {
    tfd::message_box_ok("Oops...", &message.replace("'", ""), tfd::MessageBoxIcon::Error);
    panic!("{}", message);
}

fn unwrap_result<T, E: Display>(res: Result<T, E>, msg: &str) -> T {
    match res {
        Ok(t) => { t }
        Err(e) => {
            crash_with_error_dialog(&format!("{}\n{}", msg, e));
        }
    }
}

fn push_matrix_to_vec(vec: &mut Vec<f32>, matrix: &[f32]) {
    for k in 0..16 {
        vec.push(matrix[k]);
    }
}

fn time_from_epoch_ms() -> u128 {
    SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis()
}

fn regenerate_terrain(spec: &mut TerrainSpec, terrain_geometry: &vkutil::VirtualGeometry, fixed_seed: bool) {
    if !fixed_seed {
        spec.seed = time_from_epoch_ms();
    }
    let plane_vertices = spec.generate_vertices();
    terrain_geometry.vertex_buffer.upload_buffer(&plane_vertices);
}

fn load_global_bc7(vk: &VulkanAPI, global_textures: &mut FreeList<vk::DescriptorImageInfo>, sampler: vk::Sampler, command_buffer: vk::CommandBuffer, path: &str, color_space: vkutil::ColorSpace) -> u32 {
    unsafe {
        let vim = VirtualImage::from_bc7(&vk, command_buffer, path, color_space);

        let descriptor_info = vk::DescriptorImageInfo {
            sampler: sampler,
            image_view: vim.vk_view,
            image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
        };
        let index = global_textures.insert(descriptor_info);

        index as u32
    }
}

//Entry point
fn main() {
    //Create the window using SDL
    let sdl_context = unwrap_result(sdl2::init(), "Error initializing SDL");
    let mut event_pump = unwrap_result(sdl_context.event_pump(), "Error initializing SDL event pump");
    let video_subsystem = unwrap_result(sdl_context.video(), "Error initializing SDL video subsystem");
    let controller_subsystem = unwrap_result(sdl_context.game_controller(), "Error initializing SDL controller subsystem");
    let mut window_size = glm::vec2(1280, 720);
    let window = unwrap_result(video_subsystem.window("Vulkan't", window_size.x, window_size.y).position_centered().resizable().vulkan().build(), "Error creating window");
    
    //Initialize the SDL mixer
    let mut music_volume = 0;
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

        let command_pool = vk.device.create_command_pool(&pool_create_info, vkutil::MEMORY_ALLOCATOR).unwrap();

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

    //Maintain free list for texture allocation
    let mut global_textures = FreeList::with_capacity(1024);

    let default_texture_sampler;

    let default_normal_index = unsafe {

    };

    //Global texture loading
    let cartoon_grass_global_index = load_global_bc7(&vk, &mut global_textures, material_sampler, vk_command_buffer, "./data/textures/grass/color.dds", ColorSpace::SRGB);
    let grass_color_global_index = load_global_bc7(&vk, &mut global_textures, material_sampler, vk_command_buffer, "./data/textures/whispy_grass/color.dds", ColorSpace::SRGB);
    let grass_normal_global_index = load_global_bc7(&vk, &mut global_textures, material_sampler, vk_command_buffer, "./data/textures/whispy_grass/normal.dds", ColorSpace::LINEAR);
    let rock_color_global_index = load_global_bc7(&vk, &mut global_textures, material_sampler, vk_command_buffer, "./data/textures/rocky_ground/color.dds", ColorSpace::SRGB);
    let rock_normal_global_index = load_global_bc7(&vk, &mut global_textures, material_sampler, vk_command_buffer, "./data/textures/rocky_ground/normal.dds", ColorSpace::LINEAR);
    let steel_plate_global_index = load_global_bc7(&vk, &mut global_textures, material_sampler, vk_command_buffer, "./data/textures/steel_plate/color.dds", ColorSpace::SRGB);
    let dragon_color_global_index = load_global_bc7(&vk, &mut global_textures, material_sampler, vk_command_buffer, "./data/textures/dragon/color.dds", ColorSpace::SRGB);
    let dragon_normal_global_index = load_global_bc7(&vk, &mut global_textures, material_sampler, vk_command_buffer, "./data/textures/dragon/normal.dds", ColorSpace::LINEAR);

    //Load environment textures
    let atmosphere_tex_indices = {
        let sunzenith_index = load_global_bc7(&vk, &mut global_textures, material_sampler, vk_command_buffer, "./data/textures/sunzenith_gradient.dds", ColorSpace::SRGB);
        let viewzenith_index = load_global_bc7(&vk, &mut global_textures, material_sampler, vk_command_buffer, "./data/textures/viewzenith_gradient.dds", ColorSpace::SRGB);
        let sunview_index = load_global_bc7(&vk, &mut global_textures, material_sampler, vk_command_buffer, "./data/textures/sunview_gradient.dds", ColorSpace::SRGB);
        [sunzenith_index.to_le_bytes(), viewzenith_index.to_le_bytes(), sunview_index.to_le_bytes()].concat()
    };

    //Create and upload Dear IMGUI font atlas
    let imgui_font_global_index = match imgui_context.fonts() {
        FontAtlasRefMut::Owned(atlas) => unsafe {
            let atlas_texture = atlas.build_alpha8_texture();
            
            let atlas_format = vk::Format::R8_UNORM;
            let image_extent = vk::Extent3D {
                width: atlas_texture.width,
                height: atlas_texture.height,
                depth: 1
            };
            let font_create_info = vk::ImageCreateInfo {
                image_type: vk::ImageType::TYPE_2D,
                format: atlas_format,
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
            let vk_font_image = vk.device.create_image(&font_create_info, vkutil::MEMORY_ALLOCATOR).unwrap();

            let vim = vkutil::VirtualImage {
                vk_image: vk_font_image,
                vk_view: vk::ImageView::default(),
                width: atlas_texture.width,
                height: atlas_texture.height,
                mip_count: 1
            };
            vkutil::upload_image(&vk, vk_command_buffer, &vim, atlas_texture.data);

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
                format: atlas_format,
                view_type: vk::ImageViewType::TYPE_2D,
                components: vkutil::COMPONENT_MAPPING_DEFAULT,
                subresource_range: sampler_subresource_range,
                ..Default::default()
            };
            let font_view = vk.device.create_image_view(&font_view_info, vkutil::MEMORY_ALLOCATOR).unwrap();
            
            let image_info = vk::DescriptorImageInfo {
                sampler: font_sampler,
                image_view: font_view,
                image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
            };
            let sampler_index = global_textures.insert(image_info);
            
            default_texture_sampler = image_info;
            atlas.tex_id = imgui::TextureId::new(sampler_index);    //Giving Dear Imgui a reference to the font atlas GPU texture
            sampler_index as u32
        }
        FontAtlasRefMut::Shared(_) => {
            panic!("Not dealing with this case.");
        }
    };

    //Create free list for materials
    let mut global_materials = FreeList::with_capacity(256);

    let terrain_grass_matidx = global_materials.insert(Material {
        color_idx: grass_color_global_index,
        normal_idx: grass_normal_global_index
    }) as u32;

    let terrain_rock_matidx = global_materials.insert(Material {
        color_idx: rock_color_global_index,
        normal_idx: rock_normal_global_index
    }) as u32;

    let dragon_matidx = global_materials.insert(Material {
        color_idx: dragon_color_global_index,
        normal_idx: dragon_normal_global_index
    }) as u32;

    let totoro_matidx = global_materials.insert(Material {
        color_idx: imgui_font_global_index,
        normal_idx: rock_normal_global_index
    }) as u32;

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
        vk.device.create_render_pass(&renderpass_info, vkutil::MEMORY_ALLOCATOR).unwrap()
    };

    //Create the main swapchain for window present
    let mut vk_display = vkutil::Display::initialize_swapchain(&vk, &vk_ext_swapchain, vk_render_pass);

    //Allocate buffer for frame-constant uniforms
    let uniform_buffer_size = (5 * size_of::<glm::TMat4<f32>>() + size_of::<glm::TVec4<f32>>()) as vk::DeviceSize;
    let frame_uniform_buffer = VirtualBuffer::new(&vk, uniform_buffer_size, vk::BufferUsageFlags::UNIFORM_BUFFER);
    
    //Allocate buffer for object transforms
    let global_transform_slots = 4 * 1024 * 1024;
    let buffer_size = (size_of::<glm::TMat4<f32>>() * global_transform_slots) as vk::DeviceSize;
    let transform_storage_buffer = VirtualBuffer::new(&vk, buffer_size, vk::BufferUsageFlags::STORAGE_BUFFER);

    //Allocate material buffer
    let global_material_slots = 1024;
    let material_size = 2 * size_of::<u32>() as u64;
    let material_storage_buffer = VirtualBuffer::new(&vk, material_size * global_material_slots, vk::BufferUsageFlags::STORAGE_BUFFER);

    let mut host_transform_buffer = Vec::with_capacity(16 * global_transform_slots);
    
    //Set up descriptors
    let vk_descriptor_set_layout;
    let vk_descriptor_sets = unsafe {
        let per_frame_type = vk::DescriptorType::UNIFORM_BUFFER;
        let uniform_binding = vk::DescriptorSetLayoutBinding {
            binding: 0,
            descriptor_type: per_frame_type,
            descriptor_count: 1,
            stage_flags: vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
            ..Default::default()
        };
        let uniform_pool_size = vk::DescriptorPoolSize {
            ty: per_frame_type,
            descriptor_count: 1
        };

        let texture_binding = vk::DescriptorSetLayoutBinding {
            binding: 1,
            descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            descriptor_count: global_textures.size() as u32,
            stage_flags: vk::ShaderStageFlags::FRAGMENT,
            ..Default::default()
        };
        let sampler_pool_size = vk::DescriptorPoolSize {
            ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            descriptor_count: global_textures.size() as u32,
        };

        let transforms_binding = vk::DescriptorSetLayoutBinding {
            binding: 2,
            descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: 1,
            stage_flags: vk::ShaderStageFlags::VERTEX,
            ..Default::default()
        };
        let transforms_pool_size = vk::DescriptorPoolSize {
            ty: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: 1
        };

        let materials_binding = vk::DescriptorSetLayoutBinding {
            binding: 3,
            descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: 1,
            stage_flags: vk::ShaderStageFlags::FRAGMENT,
            ..Default::default()
        };
        let materials_pool_size = vk::DescriptorPoolSize {
            ty: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: 1
        };

        let bindings = [uniform_binding, texture_binding, transforms_binding, materials_binding];
        let pool_sizes = [uniform_pool_size, sampler_pool_size, transforms_pool_size, materials_pool_size];

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

        vk_descriptor_set_layout = vk.device.create_descriptor_set_layout(&descriptor_layout, vkutil::MEMORY_ALLOCATOR).unwrap();

        let vk_alloc_info = vk::DescriptorSetAllocateInfo {
            descriptor_pool,
            descriptor_set_count: total_set_count,
            p_set_layouts: &vk_descriptor_set_layout,
            ..Default::default()
        };
        vk.device.allocate_descriptor_sets(&vk_alloc_info).unwrap()
    };

    //Write initial values to buffer descriptors
    unsafe {
        let uniform_infos = [
            vk::DescriptorBufferInfo {
                buffer: frame_uniform_buffer.backing_buffer(),
                offset: 0,
                range: uniform_buffer_size
            }
        ];
        let storage_info = vk::DescriptorBufferInfo {
            buffer: transform_storage_buffer.backing_buffer(),
            offset: 0,
            range: (global_transform_slots * size_of::<glm::TMat4<f32>>()) as vk::DeviceSize
        };
        let materials_info = vk::DescriptorBufferInfo {
            buffer: material_storage_buffer.backing_buffer(),
            offset: 0,
            range: material_storage_buffer.length()
        };

        let uniform_write = vk::WriteDescriptorSet {
            dst_set: vk_descriptor_sets[0],
            descriptor_count: uniform_infos.len() as u32,
            descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
            p_buffer_info: uniform_infos.as_ptr(),
            dst_array_element: 0,
            dst_binding: 0,
            ..Default::default()
        };
        let storage_write = vk::WriteDescriptorSet {
            dst_set: vk_descriptor_sets[0],
            descriptor_count: 1,
            descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
            p_buffer_info: &storage_info,
            dst_array_element: 0,
            dst_binding: 2,
            ..Default::default()
        };
        let material_write = vk::WriteDescriptorSet {
            dst_set: vk_descriptor_sets[0],
            descriptor_count: 1,
            descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
            p_buffer_info: &materials_info,
            dst_array_element: 0,
            dst_binding: 3,
            ..Default::default()
        };

        vk.device.update_descriptor_sets(&[uniform_write, storage_write, material_write], &[]);
    }

    let push_constant_shader_stage_flags = vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT;
    let vk_pipeline_layout = unsafe {
        let push_constant_range = vk::PushConstantRange {
            stage_flags: push_constant_shader_stage_flags,
            offset: 0,
            size: 3 * size_of::<u32>() as u32
        };
        let pipeline_layout_createinfo = vk::PipelineLayoutCreateInfo {
            push_constant_range_count: 1,
            p_push_constant_ranges: &push_constant_range,
            set_layout_count: 1,
            p_set_layouts: &vk_descriptor_set_layout,
            ..Default::default()
        };
        
        vk.device.create_pipeline_layout(&pipeline_layout_createinfo, vkutil::MEMORY_ALLOCATOR).unwrap()
    };

    //Create pipelines
    let [vk_3D_graphics_pipeline, terrain_pipeline, atmosphere_pipeline, imgui_graphics_pipeline, vk_wireframe_graphics_pipeline] = unsafe {
        //Load shaders
        let main_shader_stages = {
            let v = vkutil::load_shader_stage(&vk.device, vk::ShaderStageFlags::VERTEX, "./data/shaders/main.vs.spv");
            let f = vkutil::load_shader_stage(&vk.device, vk::ShaderStageFlags::FRAGMENT, "./data/shaders/main.fs.spv");
            [v, f]
        };
        
        let terrain_shader_stages = {
            let v = vkutil::load_shader_stage(&vk.device, vk::ShaderStageFlags::VERTEX, "./data/shaders/main.vs.spv");
            let f = vkutil::load_shader_stage(&vk.device, vk::ShaderStageFlags::FRAGMENT, "./data/shaders/terrain.fs.spv");
            [v, f]
        };
        
        let atm_shader_stages = {
            let v = vkutil::load_shader_stage(&vk.device, vk::ShaderStageFlags::VERTEX, "./data/shaders/atmosphere.vs.spv");
            let f = vkutil::load_shader_stage(&vk.device, vk::ShaderStageFlags::FRAGMENT, "./data/shaders/atmosphere.fs.spv");
            [v, f]
        };

        let im_shader_stages = {
            let v = vkutil::load_shader_stage(&vk.device, vk::ShaderStageFlags::VERTEX, "./data/shaders/imgui.vs.spv");
            let f = vkutil::load_shader_stage(&vk.device, vk::ShaderStageFlags::FRAGMENT, "./data/shaders/imgui.fs.spv");
            [v, f]
        };

        let pipeline_creator = vkutil::PipelineCreator::init(vk_pipeline_layout);

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
        let main_vertex_config = vkutil::VertexInputConfiguration {
            binding_descriptions: &bindings,
            attribute_descriptions: &attrs
        };
        let mut main_create_info = vkutil::VirtualPipelineCreateInfo::new(vk_render_pass, main_vertex_config, &main_shader_stages);
        let main_pipeline = pipeline_creator.create_pipeline(&vk, &main_create_info);

        main_create_info.rasterization_state = Some(vk::PipelineRasterizationStateCreateInfo {
            polygon_mode: vk::PolygonMode::LINE,
            ..pipeline_creator.default_rasterization_state
        });
        let wire_pipeline = pipeline_creator.create_pipeline(&vk, &main_create_info);
        main_create_info.rasterization_state = None;

        main_create_info.shader_stages = &terrain_shader_stages;
        let ter_pipeline = pipeline_creator.create_pipeline(&vk, &main_create_info);

        let atmosphere_vert_binding = vk::VertexInputBindingDescription {
            binding: 0,
            stride: 3 * size_of::<f32>() as u32,
            input_rate: vk::VertexInputRate::VERTEX
        };

        let atmosphere_position_attribute = vk::VertexInputAttributeDescription {
            location: 0,
            binding: 0,
            format: vk::Format::R32G32B32_SFLOAT,
            offset: 0
        };

        let bindings = [atmosphere_vert_binding];
        let attrs = [atmosphere_position_attribute];
        let atm_vertex_config = vkutil::VertexInputConfiguration {
            binding_descriptions: &bindings,
            attribute_descriptions: &attrs
        };
        let atm_create_info = vkutil::VirtualPipelineCreateInfo::new(vk_render_pass, atm_vertex_config, &atm_shader_stages);
        let atm_pipeline = pipeline_creator.create_pipeline(&vk, &atm_create_info);

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
        let im_vertex_config = vkutil::VertexInputConfiguration {
            binding_descriptions: &im_bindings,
            attribute_descriptions: &im_attrs
        };
        let mut im_create_info = vkutil::VirtualPipelineCreateInfo::new(vk_render_pass, im_vertex_config, &im_shader_stages);

        let im_depthstencil = vk::PipelineDepthStencilStateCreateInfo {
            depth_test_enable: vk::FALSE,
            depth_write_enable: vk::FALSE,
            ..pipeline_creator.default_depthstencil_state
        };
        let im_rasterization_state = vk::PipelineRasterizationStateCreateInfo {
            cull_mode: vk::CullModeFlags::NONE,
            ..pipeline_creator.default_rasterization_state
        };
        im_create_info.depthstencil_state = Some(im_depthstencil);
        im_create_info.rasterization_state = Some(im_rasterization_state);
        let im_pipeline = pipeline_creator.create_pipeline(&vk, &im_create_info);

        [main_pipeline, ter_pipeline, atm_pipeline, im_pipeline, wire_pipeline]
    };

    let terrain_width_height = 256;
    let mut terrain_fixed_seed = false;
    let mut terrain_interactive_generation = false;
    let mut terrain = TerrainSpec {
        vertex_width: terrain_width_height,
        vertex_height: terrain_width_height,
        noise_parameters: vec![
            NoiseParameters { amplitude: 2.0, frequency: 0.15 },
            NoiseParameters { amplitude: 2.0, frequency: 0.20 },
            NoiseParameters { amplitude: 0.25, frequency: 1.5 },
            NoiseParameters { amplitude: 0.0125, frequency: 8.0 },
        ],
        amplitude: 2.0,
        exponent: 2.2,
        seed: time_from_epoch_ms()
    };

    let plane_vertices = terrain.generate_vertices();
    let plane_indices = ozy::prims::plane_index_buffer(terrain_width_height, terrain_width_height);

    //Load totoro model
    let totoro_mesh = OzyMesh::load("./data/models/totoro.ozy").unwrap();

    //Load dragon model
    let dragon_mesh = OzyMesh::load("./data/models/dragon.ozy").unwrap();

    let terrain_geometry;
    let totoro_geometry;
    let dragon_geometry;
    let atmosphere_geometry;
    unsafe {
        let scene_geo_buffer_size = 2 * vkutil::DEFAULT_ALLOCATION_SIZE;
        let scene_geo_buffer = {
            //Buffer creation
            let buffer_create_info = vk::BufferCreateInfo {
                usage: vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::INDEX_BUFFER,
                size: scene_geo_buffer_size as vk::DeviceSize,
                sharing_mode: vk::SharingMode::EXCLUSIVE,
                ..Default::default()
            };
            let buffer = vk.device.create_buffer(&buffer_create_info, vkutil::MEMORY_ALLOCATOR).unwrap();
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

        terrain_geometry = scene_geo_allocator.allocate_geometry(&plane_vertices, &plane_indices).unwrap();
        totoro_geometry = scene_geo_allocator.allocate_geometry(&totoro_mesh.vertex_array.vertices, &totoro_mesh.vertex_array.indices).unwrap();
        dragon_geometry = scene_geo_allocator.allocate_geometry(&dragon_mesh.vertex_array.vertices, &dragon_mesh.vertex_array.indices).unwrap();
        atmosphere_geometry = scene_geo_allocator.allocate_geometry(&ozy::prims::skybox_cube_vertex_buffer(), &ozy::prims::skybox_cube_index_buffer()).unwrap();

        vk.device.unmap_memory(buffer_memory);
    }

    let mut imgui_geo_allocator = unsafe {
        let imgui_buffer_size = vkutil::DEFAULT_ALLOCATION_SIZE;
        let buffer_create_info = vk::BufferCreateInfo {
            usage: vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::INDEX_BUFFER,
            size: imgui_buffer_size as vk::DeviceSize,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            ..Default::default()
        };
        let buffer = vk.device.create_buffer(&buffer_create_info, vkutil::MEMORY_ALLOCATOR).unwrap();
        let buffer_memory = vkutil::allocate_buffer_memory(&vk, buffer);
        vk.device.bind_buffer_memory(buffer, buffer_memory, 0).unwrap();

        let ptr = vk.device.map_memory(buffer_memory, 0, imgui_buffer_size, vk::MemoryMapFlags::empty()).unwrap();

        vkutil::VirtualBumpAllocator::new(buffer, ptr, imgui_buffer_size)
    };

    //Create semaphore used to wait on swapchain image
    let vk_swapchain_semaphore = unsafe { vk.device.create_semaphore(&vk::SemaphoreCreateInfo::default(), vkutil::MEMORY_ALLOCATOR).unwrap() };
    let vk_rendercomplete_semaphore = unsafe { vk.device.create_semaphore(&vk::SemaphoreCreateInfo::default(), vkutil::MEMORY_ALLOCATOR).unwrap() };

    //State for freecam controls
    let mut free_camera = FreeCam::new(glm::vec3(0.0f32, -30.0, 15.0));

    let mut totoro_lookat_pos = glm::normalize(&glm::vec3(-1.0f32, 0.0, 1.75));
    let totoro_lookat_dist = 10.0;

    let mut timer = FrameTimer::new();      //Struct for doing basic framerate independence
    
    let mut sun_speed = 0.003;
    let mut sun_pitch = 0.118;
    let mut sun_yaw = 0.783;
    
    //Load and play bgm
    let bgm = unwrap_result(Music::from_file("./data/music/relaxing_botw.mp3"), "Error loading bgm");
    bgm.play(-1).unwrap();

    let mut do_imgui = true;
    let mut do_terrain_window = false;
    let mut do_freecam = true;
    
    let mut wireframe = false;
    let mut main_pipeline = vk_3D_graphics_pipeline;

    let mut game_controllers = [None, None, None, None];

    //Draw related lists that reset every frame
    let mut vk_draw_calls = Vec::with_capacity(64);

    let vk_submission_fence = unsafe {
        let create_info = vk::FenceCreateInfo {
            flags: vk::FenceCreateFlags::SIGNALED,
            ..Default::default()
        };
        vk.device.create_fence(&create_info, vkutil::MEMORY_ALLOCATOR).unwrap()
    };

    //Main application loop
    'running: loop {
        timer.update(); //Update frame timer
        vk_draw_calls.clear();
        host_transform_buffer.clear();

        //Abstracted input variables
        let mut movement_multiplier = 5.0f32;
        let mut movement_vector: glm::TVec3<f32> = glm::zero();
        let mut camera_orientation_delta: glm::TVec2<f32> = glm::zero();

        //Input
        let framerate;
        {
            use sdl2::controller::Button;
            use sdl2::event::WindowEvent;
            use sdl2::keyboard::{Scancode};
            use sdl2::mouse::MouseButton;

            //Sync controller array with how many controllers are actually connected
            for i in 0..game_controllers.len() {
                match &mut game_controllers[i] {
                    None => {
                        if i < unwrap_result(controller_subsystem.num_joysticks(), "Error getting number of controllers") as usize {
                            let controller = unwrap_result(controller_subsystem.open(i as u32), "Error opening controller");
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
                            WindowEvent::Resized(_, _) => unsafe {
                                vk.device.wait_for_fences(&[vk_submission_fence], true, vk::DeviceSize::MAX).unwrap();

                                //Free the now-invalid swapchain data
                                for framebuffer in vk_display.swapchain_framebuffers {
                                    vk.device.destroy_framebuffer(framebuffer, vkutil::MEMORY_ALLOCATOR);
                                }
                                for view in vk_display.swapchain_image_views {
                                    vk.device.destroy_image_view(view, vkutil::MEMORY_ALLOCATOR);
                                }
                                vk.device.destroy_image_view(vk_display.depth_image_view, vkutil::MEMORY_ALLOCATOR);
                                vk_ext_swapchain.destroy_swapchain(vk_display.swapchain, vkutil::MEMORY_ALLOCATOR);

                                //Recreate swapchain and associated data
                                vk_display = vkutil::Display::initialize_swapchain(&vk, &vk_ext_swapchain, vk_render_pass);

                                window_size = glm::vec2(vk_display.extent.width, vk_display.extent.height);
                                imgui_io.display_size[0] = window_size.x as f32;
                                imgui_io.display_size[1] = window_size.y as f32;
                            }
                            _ => {}
                        }
                    }
                    Event::KeyDown { scancode: Some(Scancode::Escape), repeat: false, .. } => {
                        do_imgui = !do_imgui;
                    }
                    Event::MouseButtonUp { mouse_btn, ..} => {
                        match mouse_btn {
                            MouseButton::Right => {
                                free_camera.cursor_captured = !free_camera.cursor_captured;
                                let mouse_util = sdl_context.mouse();
                                mouse_util.set_relative_mouse_mode(free_camera.cursor_captured);
                                if !free_camera.cursor_captured {
                                    mouse_util.warp_mouse_in_window(&window, window_size.x as i32 / 2, window_size.y as i32 / 2);
                                }
                            }
                            _ => {}
                        }
                    }
                    Event::MouseMotion { xrel, yrel, .. } => {
                        if free_camera.cursor_captured {
                            const DAMPENING: f32 = 0.25 / 360.0;
                            camera_orientation_delta += glm::vec2(DAMPENING * xrel as f32, DAMPENING * yrel as f32);
                        }
                    }
                    Event::MouseWheel { x, y, .. } => {
                        imgui_io.mouse_wheel_h = x as f32;
                        imgui_io.mouse_wheel = y as f32;
                    }
                    _ => {  }
                }
            }

            let keyboard_state = event_pump.keyboard_state();
            let mouse_state = event_pump.mouse_state();
            imgui_io.mouse_down = [mouse_state.left(), mouse_state.right(), mouse_state.middle(), mouse_state.x1(), mouse_state.x2()];
            imgui_io.mouse_pos[0] = mouse_state.x() as f32;
            imgui_io.mouse_pos[1] = mouse_state.y() as f32;

            if let Some(controller) = &mut game_controllers[0] {
                use sdl2::controller::{Axis};

                fn get_normalized_axis(controller: &GameController, axis: Axis) -> f32 {
                    controller.axis(axis) as f32 / i16::MAX as f32
                }

                if controller.button(Button::LeftShoulder) {
                    movement_vector += glm::vec3(0.0, 0.0, -1.0);                    
                }

                if controller.button(Button::RightShoulder) {
                    movement_vector += glm::vec3(0.0, 0.0, 1.0);                    
                }

                if controller.button(Button::Y) {
                    regenerate_terrain(&mut terrain, &terrain_geometry, terrain_fixed_seed);
                    if let Err(e) = controller.set_rumble(0xFFFF, 0xFFFF, 50) {
                        println!("{}", e);
                    }
                }

                let left_trigger = get_normalized_axis(&controller, Axis::TriggerLeft);
                movement_multiplier *= 4.0 * left_trigger + 1.0;

                const JOYSTICK_DEADZONE: f32 = 0.15;
                let left_joy_vector = {
                    let x = get_normalized_axis(&controller, Axis::LeftX);
                    let y = get_normalized_axis(&controller, Axis::LeftY);
                    let mut res = glm::vec3(x, -y, 0.0);
                    if glm::length(&res) < JOYSTICK_DEADZONE {
                        res = glm::zero();
                    }
                    res
                };
                let right_joy_vector = {
                    let x = get_normalized_axis(&controller, Axis::RightX);
                    let y = get_normalized_axis(&controller, Axis::RightY);
                    let mut res = glm::vec2(x, -y);
                    if glm::length(&res) < JOYSTICK_DEADZONE {
                        res = glm::zero();
                    }
                    res
                };

                movement_vector += &left_joy_vector;
                camera_orientation_delta += 4.0 * timer.delta_time * glm::vec2(right_joy_vector.x, -right_joy_vector.y);
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
        let imgui_ui = imgui_context.frame();   //Transition Dear ImGUI into recording state
        if do_imgui && do_terrain_window {
            if let Some(token) = imgui::Window::new("Terrain builder").begin(&imgui_ui) { 
                let mut parameters_changed = false;

                let mut num_deleted = 0;
                for i in 0..terrain.noise_parameters.len() {
                    let idx = i - num_deleted;
                    imgui_ui.text(format!("Noise sample {}", i));

                    parameters_changed |= imgui::Slider::new(format!("Amplitude##{}", i), 0.0, 2.0).build(&imgui_ui, &mut terrain.noise_parameters[idx].amplitude);
                    parameters_changed |= imgui::Slider::new(format!("Frequency##{}", i), 0.0, 5.0).build(&imgui_ui, &mut terrain.noise_parameters[idx].frequency);

                    if imgui_ui.button_with_size(format!("Remove this layer##{}", i), [0.0, 32.0]) {
                        terrain.noise_parameters.remove(idx);
                        num_deleted += 1;
                        parameters_changed = true;
                    }

                    imgui_ui.separator();
                }
                
                if imgui_ui.button_with_size("Add noise layer", [0.0, 32.0]) {
                    terrain.noise_parameters.push(NoiseParameters::default());
                }

                imgui_ui.text("Global amplitude and exponent:");
                parameters_changed |= imgui::Slider::new("Amplitude", 0.0, 8.0).build(&imgui_ui, &mut terrain.amplitude);
                parameters_changed |= imgui::Slider::new("Exponent", 1.0, 5.0).build(&imgui_ui, &mut terrain.exponent);
                imgui_ui.separator();

                imgui_ui.text(format!("Last seed used: 0x{:X}", terrain.seed));
                imgui_ui.checkbox("Use fixed seed", &mut terrain_fixed_seed);
                imgui_ui.checkbox("Interactive mode", &mut terrain_interactive_generation);
                if imgui_ui.button_with_size("Regenerate", [0.0, 32.0]) {
                    regenerate_terrain(&mut terrain, &terrain_geometry, terrain_fixed_seed);
                }

                if terrain_interactive_generation && parameters_changed {
                    regenerate_terrain(&mut terrain, &terrain_geometry, terrain_fixed_seed);
                }

                if imgui_ui.button_with_size("Close", [0.0, 32.0]) { do_terrain_window = false; }

                token.end();
            }
        }

        movement_vector *= movement_multiplier;

        let dc = vkutil::VirtualDrawCall::new(&terrain_geometry, terrain_pipeline, [terrain_grass_matidx, 0, 0], 1, host_transform_buffer.len() as u32 / 16);
        vk_draw_calls.push(dc);        
        let plane_model_matrix = glm::scaling(&glm::vec3(30.0, 30.0, 30.0));
        push_matrix_to_vec(&mut host_transform_buffer, plane_model_matrix.as_slice());
        
        let dc = vkutil::VirtualDrawCall::new(&totoro_geometry, main_pipeline, [totoro_matidx, 0, 0], 1, host_transform_buffer.len() as u32 / 16);
        vk_draw_calls.push(dc);        
        let model_matrix = glm::rotation(timer.elapsed_time, &glm::vec3(0.0, 0.0, 1.0)) * ozy::routines::uniform_scale(3.0);
        push_matrix_to_vec(&mut host_transform_buffer, model_matrix.as_slice());

        let view_matrix = if do_freecam {
            //Camera orientation based on user input
            free_camera.orientation += camera_orientation_delta;
            free_camera.orientation.y = free_camera.orientation.y.clamp(-glm::half_pi::<f32>(), glm::half_pi::<f32>());
            free_camera.make_view_matrix()
        } else {
            let tot_lookat = model_matrix * glm::vec4(model_matrix[12], model_matrix[13], model_matrix[14] + 0.75, 1.0);
            let new_pos = glm::rotation(camera_orientation_delta.x, &glm::vec3(0.0, 0.0, 1.0)) * glm::vec3_to_vec4(&totoro_lookat_pos);
            let new_pos =  glm::rotation(camera_orientation_delta.y, &glm::vec3(1.0, 0.0, 0.0)) * new_pos;
            totoro_lookat_pos = glm::vec4_to_vec3(&new_pos);
            glm::look_at(&(totoro_lookat_dist * totoro_lookat_pos), &glm::vec4_to_vec3(&tot_lookat), &glm::vec3(0.0, 0.0, 1.0))
        };

        if do_freecam {
            const FREECAM_SPEED: f32 = 3.0;
            let view_movement_vector = glm::mat4(
                1.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0,
                0.0, -1.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 1.0
            ) * glm::vec3_to_vec4(&movement_vector);
            let view_movement_vector = glm::vec4_to_vec3(&view_movement_vector);
            let delta_pos = FREECAM_SPEED * glm::affine_inverse(view_matrix) * glm::vec3_to_vec4(&view_movement_vector) * timer.delta_time;
            free_camera.position += glm::vec4_to_vec3(&delta_pos);
        }

        let imgui_window_token = if do_imgui {
            imgui::Window::new("Main control panel (press ESC to hide)").menu_bar(true).begin(&imgui_ui)
        } else {
            None
        };

        if let Some(_) = imgui_window_token {
            if let Some(mb) = imgui_ui.begin_menu_bar() {
                if let Some(mt) = imgui_ui.begin_menu("Environment") {
                    if imgui::MenuItem::new("Terrain builder").build(&imgui_ui) {
                        do_terrain_window = true;
                    }
                    mt.end();
                }
                mb.end();
            }

            imgui_ui.text(format!("Rendering at {:.0} FPS ({:.2} ms frametime)", framerate, 1000.0 / framerate));
            let (message, color) =  if game_controllers[0].is_some() {
                ("Controller is connected.", [0.0, 1.0, 0.0, 1.0])
            } else {
                ("Controller is not connected.", [1.0, 0.0, 0.0, 1.0])
            };
            let color_token = imgui_ui.push_style_color(imgui::StyleColor::Text, color);
            imgui_ui.text(message);
            color_token.pop();
            imgui::Slider::new("Sun speed", 0.0, 1.0).build(&imgui_ui, &mut sun_speed);
            imgui::Slider::new("Sun pitch", 0.0, glm::two_pi::<f32>()).build(&imgui_ui, &mut sun_pitch);
            imgui::Slider::new("Sun yaw", 0.0, glm::two_pi::<f32>()).build(&imgui_ui, &mut sun_yaw);
        }

        let dc = vkutil::VirtualDrawCall::new(&dragon_geometry, main_pipeline, [dragon_matidx, 0, 0], 1, host_transform_buffer.len() as u32 / 16);
        vk_draw_calls.push(dc);

        let model_matrix = glm::translation(&glm::vec3(-200.0, 300.0, 2.0 * f32::sin(timer.elapsed_time) + 11.0)) * glm::rotation(glm::quarter_pi::<f32>(), &glm::vec3(0.0, 0.0, 1.0));
        push_matrix_to_vec(&mut host_transform_buffer, model_matrix.as_slice());
        
        //Update sun's position
        sun_pitch += sun_speed * timer.delta_time;
        if sun_pitch > glm::two_pi() {
            sun_pitch -= glm::two_pi::<f32>();
        }

        if let Some(t) = imgui_window_token {
            if imgui::Slider::new("Music volume", 0, 128).build(&imgui_ui, &mut music_volume) { Music::set_volume(music_volume); }
            if imgui_ui.checkbox("Wireframe view", &mut wireframe) {
                if !wireframe {
                    main_pipeline = vk_3D_graphics_pipeline;
                } else {
                    main_pipeline = vk_wireframe_graphics_pipeline;
                }
            }
            imgui_ui.checkbox("Freecam", &mut do_freecam);

            imgui_ui.text(format!("Freecam is at ({}, {}, {})", free_camera.position.x, free_camera.position.y, free_camera.position.z));
            if imgui_ui.button_with_size("Exit", [0.0, 32.0]) {
                break 'running;
            }

            t.end();
        }
        
        //Pre-render phase

        unsafe {
            vk.device.wait_for_fences(&[vk_submission_fence], true, vk::DeviceSize::MAX).unwrap();
            vk.device.reset_fences(&[vk_submission_fence]).unwrap();
        }

        //Update bindless texture sampler descriptors
        if global_textures.updated {
            global_textures.updated = false;

            let mut image_infos = vec![default_texture_sampler; global_textures.size() as usize];
            for i in 0..global_textures.len() {
                if let Some(info) = global_textures[i] {
                    image_infos[i] = info;
                }
            }

            let sampler_write = vk::WriteDescriptorSet {
                dst_set: vk_descriptor_sets[0],
                descriptor_count: global_textures.size() as u32,
                descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                p_image_info: image_infos.as_ptr(),
                dst_array_element: 0,
                dst_binding: 1,
                ..Default::default()
            };
            unsafe { vk.device.update_descriptor_sets(&[sampler_write], &[]); }
        }

        if global_materials.updated {
            global_materials.updated = false;

            let mut upload_mats = Vec::with_capacity(global_materials.len());
            for i in 0..global_materials.len() {
                if let Some(mat) = &global_materials[i] {
                    upload_mats.push(mat.color_idx);
                    upload_mats.push(mat.normal_idx);
                }
            }

            let material_ptr = material_storage_buffer.ptr() as *mut u32;
            unsafe { ptr::copy_nonoverlapping(upload_mats.as_ptr(), material_ptr, upload_mats.len())} ;
        }
        
        //Update uniform/storage buffers
        unsafe {
            //Update static scene data
            let clip_from_screen = glm::mat4(
                2.0 / window_size.x as f32, 0.0, 0.0, -1.0,
                0.0, 2.0 / window_size.y as f32, 0.0, -1.0,
                0.0, 0.0, 1.0, 0.0,
                0.0, 0.0, 0.0, 1.0
            );

            let projection_matrix = glm::perspective(window_size.x as f32 / window_size.y as f32, glm::half_pi(), 0.2, 50.0);
    
            //Relative to OpenGL clip space, Vulkan has negative Y and half Z.
            let projection_matrix = glm::mat4(
                1.0, 0.0, 0.0, 0.0,
                0.0, -1.0, 0.0, 0.0,
                0.0, 0.0, 0.5, 0.0,
                0.0, 0.0, 0.5, 1.0,
            ) * projection_matrix;
    
            let view_projection = projection_matrix * view_matrix;
            
            //Compute sun direction from pitch and yaw
            let sun_direction = glm::vec4_to_vec3(&(
                glm::rotation(sun_yaw, &glm::vec3(0.0, 0.0, 1.0)) *
                glm::rotation(sun_pitch, &glm::vec3(0.0, 1.0, 0.0)) *
                glm::vec4(-1.0, 0.0, 0.0, 0.0)
            ));
            
            //Compute the view-projection matrix for the skybox (the conversion functions are just there to nullify the translation component of the view matrix)
            //The skybox vertices should be rotated along with the camera, but they shouldn't be translated in order to maintain the illusion
            //that the sky is infinitely far away
            let skybox_view_projection = projection_matrix * glm::mat3_to_mat4(&glm::mat4_to_mat3(&view_matrix));

            let uniform_buffer = [
                clip_from_screen.as_slice(),
                view_projection.as_slice(),
                projection_matrix.as_slice(),
                view_matrix.as_slice(),
                skybox_view_projection.as_slice(),
                sun_direction.as_slice()
            ].concat();

            let uniform_ptr = frame_uniform_buffer.ptr() as *mut f32;
            ptr::copy_nonoverlapping(uniform_buffer.as_ptr() as *mut _, uniform_ptr, uniform_buffer.len() * size_of::<f32>());

            //Update model matrix storage buffer
            let transform_ptr = transform_storage_buffer.ptr() as *mut f32;
            ptr::copy_nonoverlapping(host_transform_buffer.as_ptr(), transform_ptr, host_transform_buffer.len());
        };

        //Dear ImGUI virtual geo allocations
        let (imgui_geometries, imgui_cmd_lists) = {
            let mut geos = Vec::with_capacity(16);
            let mut cmds = Vec::with_capacity(16);
            imgui_geo_allocator.clear();

            let imgui_draw_data = imgui_ui.render();
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
            //Begin acquiring swapchain. This is called as early as possible in order to minimize time waiting
            let current_framebuffer_index = vk_ext_swapchain.acquire_next_image(vk_display.swapchain, vk::DeviceSize::MAX, vk_swapchain_semaphore, vk::Fence::null()).unwrap().0 as usize;

            //Put command buffer in recording state
            vk.device.begin_command_buffer(vk_command_buffer, &vk::CommandBufferBeginInfo::default()).unwrap();
            
            //Set the viewport for this frame
            let viewport = vk::Viewport {
                x: 0.0,
                y: 0.0,
                width: (vk_display.extent.width) as f32,
                height: (vk_display.extent.height) as f32,
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

            let scissor_area = vk::Rect2D {
                offset: vk::Offset2D::default(),
                extent: vk::Extent2D {
                    width: window_size.x,
                    height: window_size.y
                }
            };
            vk.device.cmd_set_scissor(vk_command_buffer, 0, &[scissor_area]);

            let vk_clear_values = [vkutil::COLOR_CLEAR, vkutil::DEPTH_STENCIL_CLEAR];
            let rp_begin_info = vk::RenderPassBeginInfo {
                render_pass: vk_render_pass,
                framebuffer: vk_display.swapchain_framebuffers[current_framebuffer_index],
                render_area: vk_render_area,
                clear_value_count: vk_clear_values.len() as u32,
                p_clear_values: vk_clear_values.as_ptr(),
                ..Default::default()
            };
            vk.device.cmd_begin_render_pass(vk_command_buffer, &rp_begin_info, vk::SubpassContents::INLINE);

            //Once-per-frame bindless descriptor binding
            vk.device.cmd_bind_descriptor_sets(vk_command_buffer, vk::PipelineBindPoint::GRAPHICS, vk_pipeline_layout, 0, &vk_descriptor_sets, &[]);

            //Iterate through draw calls
            let mut last_bound_pipeline = vk::Pipeline::default();
            for virtual_draw in vk_draw_calls.iter() {
                if virtual_draw.pipeline != last_bound_pipeline {
                    vk.device.cmd_bind_pipeline(vk_command_buffer, vk::PipelineBindPoint::GRAPHICS, virtual_draw.pipeline);
                    last_bound_pipeline = virtual_draw.pipeline;
                }
                vk.device.cmd_push_constants(vk_command_buffer, vk_pipeline_layout, push_constant_shader_stage_flags, 0, &virtual_draw.push_constants);
                vk.device.cmd_bind_vertex_buffers(vk_command_buffer, 0, &[virtual_draw.geometry.vertex_buffer.backing_buffer()], &[virtual_draw.geometry.vertex_buffer.offset() as u64]);
                vk.device.cmd_bind_index_buffer(vk_command_buffer, virtual_draw.geometry.index_buffer.backing_buffer(), (virtual_draw.geometry.index_buffer.offset()) as vk::DeviceSize, vk::IndexType::UINT32);
                vk.device.cmd_draw_indexed(vk_command_buffer, virtual_draw.geometry.index_count, virtual_draw.instance_count, 0, 0, virtual_draw.first_instance);
            }

            //Record atmosphere rendering commands
            vk.device.cmd_bind_pipeline(vk_command_buffer, vk::PipelineBindPoint::GRAPHICS, atmosphere_pipeline);
            vk.device.cmd_push_constants(vk_command_buffer, vk_pipeline_layout, push_constant_shader_stage_flags, 0, atmosphere_tex_indices.as_slice());
            vk.device.cmd_bind_vertex_buffers(vk_command_buffer, 0, &[atmosphere_geometry.vertex_buffer.backing_buffer()], &[atmosphere_geometry.vertex_buffer.offset() as u64]);
            vk.device.cmd_bind_index_buffer(vk_command_buffer, atmosphere_geometry.index_buffer.backing_buffer(), (atmosphere_geometry.index_buffer.offset()) as vk::DeviceSize, vk::IndexType::UINT32);
            vk.device.cmd_draw_indexed(vk_command_buffer, atmosphere_geometry.index_count, 1, 0, 0, 0);

            //Record Dear ImGUI drawing commands
            let mut prev_tex_id = u32::MAX;
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
                            
                            let tex_id = cmd_params.texture_id.id() as u32;
                            if tex_id != prev_tex_id {
                                prev_tex_id = tex_id;
                                vk.device.cmd_push_constants(vk_command_buffer, vk_pipeline_layout, push_constant_shader_stage_flags, 0, &tex_id.to_le_bytes());
                            }

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
                p_wait_semaphores: &vk_swapchain_semaphore,
                p_wait_dst_stage_mask: &pipeline_stage_flags,
                signal_semaphore_count: 1,
                p_signal_semaphores: &vk_rendercomplete_semaphore,
                command_buffer_count: 1,
                p_command_buffers: &vk_command_buffer,
                ..Default::default()
            };

            let queue = vk.device.get_device_queue(vk.queue_family_index, 0);
            vk.device.queue_submit(queue, &[submit_info], vk_submission_fence).unwrap();

            let present_info = vk::PresentInfoKHR {
                swapchain_count: 1,
                p_swapchains: &vk_display.swapchain,
                p_image_indices: &(current_framebuffer_index as u32),
                wait_semaphore_count: 1,
                p_wait_semaphores: &vk_rendercomplete_semaphore,
                ..Default::default()
            };
            if let Err(e) = vk_ext_swapchain.queue_present(queue, &present_info) {
                println!("{}", e);
            }
        }
    }
}
