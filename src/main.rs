#![allow(non_snake_case)]

//Alias some library names
extern crate ispc_texcomp as ispc;
extern crate nalgebra_glm as glm;
extern crate ozy_engine as ozy;
extern crate tinyfiledialogs as tfd;

mod gltfutil;
mod gui;
mod input;
mod physics;
mod render;
mod routines;
mod structs;

#[macro_use]
mod vkutil;

use ash::vk::{self, BufferImageCopy};
use gltfutil::GLTFPrimitive;
use gpu_allocator::MemoryLocation;
use imgui::{FontAtlasRefMut};
use ozy::io::{DDSHeader, DDSHeader_DXT10, DDS_PixelFormat};
use rapier3d::prelude::*;
use routines::struct_to_bytes;
use sdl2::event::Event;
use sdl2::mixer;
use sdl2::mixer::Music;
use slotmap::DenseSlotMap;
use std::collections::HashMap;
use std::fmt::Display;
use std::fs::{File, OpenOptions};
use std::ffi::{CStr, c_void};
use std::io::{Read, Write};
use std::mem::size_of;
use std::ptr;
use std::time::SystemTime;

use ozy::structs::{FrameTimer, OptionVec};

use input::UserInput;
use vkutil::{FreeList, GPUBuffer, GPUImage, VulkanAPI, DeferredImage};
use physics::PhysicsEngine;
use structs::{Camera, TerrainSpec, PhysicsProp};
use render::{Primitive, Renderer, Material, CascadedShadowMap, ShadowType, SunLight};

use crate::routines::*;
use crate::gltfutil::GLTFMeshData;
use crate::gui::DevGui;
use crate::structs::StaticProp;

//Entry point
fn main() {
    //Create the window using SDL
    let sdl_context = unwrap_result(sdl2::init(), "Error initializing SDL");
    let video_subsystem = unwrap_result(sdl_context.video(), "Error initializing SDL video subsystem");
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
    let mut vk = vkutil::VulkanAPI::init();

    //Initialize the physics engine
    let mut physics_engine = PhysicsEngine::new();

    let shadow_pass = unsafe {
        let depth_description = vk::AttachmentDescription {
            format: vk::Format::D32_SFLOAT,
            samples: vk::SampleCountFlags::TYPE_1,
            load_op: vk::AttachmentLoadOp::CLEAR,
            store_op: vk::AttachmentStoreOp::STORE,
            initial_layout: vk::ImageLayout::UNDEFINED,
            final_layout: vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL,
            ..Default::default()
        };

        let depth_attachment_reference = vk::AttachmentReference {
            attachment: 0,
            layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL
        };

        let subpass = vk::SubpassDescription {
            pipeline_bind_point: vk::PipelineBindPoint::GRAPHICS,
            color_attachment_count: 0,
            p_color_attachments: ptr::null(),
            p_depth_stencil_attachment: &depth_attachment_reference,
            ..Default::default()
        };

        //Multiview info
        let mask = (1 << CascadedShadowMap::CASCADE_COUNT) - 1;
        let multiview_info = vk::RenderPassMultiviewCreateInfo {
            subpass_count: 1,
            p_view_masks: &mask,
            correlation_mask_count: 1,
            p_correlation_masks: &mask,
            ..Default::default()
        };

        let attachments = [depth_description];
        let renderpass_info = vk::RenderPassCreateInfo {
            p_next: &multiview_info as *const _ as *const c_void,
            attachment_count: attachments.len() as u32,
            p_attachments: attachments.as_ptr(),
            subpass_count: 1,
            p_subpasses: &subpass,
            ..Default::default()
        };
        vk.device.create_render_pass(&renderpass_info, vkutil::MEMORY_ALLOCATOR).unwrap()
    };

    let hdr_forward_pass = unsafe {
        let color_attachment_description = vk::AttachmentDescription {
            format: vk::Format::R32G32B32A32_SFLOAT,
            samples: vk::SampleCountFlags::TYPE_1,
            load_op: vk::AttachmentLoadOp::CLEAR,
            store_op: vk::AttachmentStoreOp::STORE,
            stencil_load_op: vk::AttachmentLoadOp::DONT_CARE,
            stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
            initial_layout: vk::ImageLayout::UNDEFINED,
            final_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            ..Default::default()
        };

        let depth_attachment_description = vk::AttachmentDescription {
            format: vk::Format::D32_SFLOAT,
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

        let attachments = [color_attachment_description, depth_attachment_description];
        let renderpass_info = vk::RenderPassCreateInfo {
            attachment_count: attachments.len() as u32,
            p_attachments: attachments.as_ptr(),
            subpass_count: 1,
            p_subpasses: &subpass,
            ..Default::default()
        };
        vk.device.create_render_pass(&renderpass_info, vkutil::MEMORY_ALLOCATOR).unwrap()
    };

    let swapchain_pass = unsafe {
        let color_attachment_description = vk::AttachmentDescription {
            format: vk::Format::B8G8R8A8_SRGB,
            samples: vk::SampleCountFlags::TYPE_1,
            load_op: vk::AttachmentLoadOp::CLEAR,
            store_op: vk::AttachmentStoreOp::STORE,
            stencil_load_op: vk::AttachmentLoadOp::DONT_CARE,
            stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
            initial_layout: vk::ImageLayout::UNDEFINED,
            final_layout: vk::ImageLayout::PRESENT_SRC_KHR,
            ..Default::default()
        };

        let color_attachment_reference = vk::AttachmentReference {
            attachment: 0,
            layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
        };

        let subpass = vk::SubpassDescription {
            pipeline_bind_point: vk::PipelineBindPoint::GRAPHICS,
            color_attachment_count: 1,
            p_color_attachments: &color_attachment_reference,
            ..Default::default()
        };

        let attachments = [color_attachment_description];
        let renderpass_info = vk::RenderPassCreateInfo {
            attachment_count: attachments.len() as u32,
            p_attachments: attachments.as_ptr(),
            subpass_count: 1,
            p_subpasses: &subpass,
            ..Default::default()
        };
        vk.device.create_render_pass(&renderpass_info, vkutil::MEMORY_ALLOCATOR).unwrap()
    };

    //Initialize the renderer
    let mut renderer = Renderer::init(&mut vk, &window, swapchain_pass, hdr_forward_pass);

    //Create and upload Dear IMGUI font atlas
    match imgui_context.fonts() {
        FontAtlasRefMut::Owned(atlas) => unsafe {
            let atlas_texture = atlas.build_alpha8_texture();
            let atlas_format = vk::Format::R8_UNORM;
            let atlas_layout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
            let descriptor_info = vkutil::upload_raw_image(&mut vk, renderer.point_sampler, atlas_format, atlas_layout, atlas_texture.width, atlas_texture.height, atlas_texture.data);
            let index = renderer.global_textures.insert(descriptor_info);
            renderer.default_texture_idx = index as u32;
            
            atlas.clear_tex_data();  //Free atlas memory CPU-side
            atlas.tex_id = imgui::TextureId::new(index);    //Giving Dear Imgui a reference to the font atlas GPU texture
            index as u32
        }
        FontAtlasRefMut::Shared(_) => {
            panic!("Not dealing with this case.");
        }
    };

    let push_constant_stage_flags = vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT;
    let pipeline_layout = unsafe {
        let push_constant_range = vk::PushConstantRange {
            stage_flags: push_constant_stage_flags,
            offset: 0,
            size: 20
        };
        let pipeline_layout_createinfo = vk::PipelineLayoutCreateInfo {
            push_constant_range_count: 1,
            p_push_constant_ranges: &push_constant_range,
            set_layout_count: 1,
            p_set_layouts: &renderer.descriptor_set_layout,
            ..Default::default()
        };
        
        vk.device.create_pipeline_layout(&pipeline_layout_createinfo, vkutil::MEMORY_ALLOCATOR).unwrap()
    };

    let sun_shadow_map = CascadedShadowMap::new(
        &mut vk,
        &mut renderer,
        2048,
        &glm::perspective_fov_rh_zo(glm::half_pi::<f32>(), window_size.x as f32, window_size.y as f32, 0.1, 1000.0),
        shadow_pass
    );
    renderer.uniform_data.sun_shadowmap_idx = sun_shadow_map.texture_index() as u32;
    renderer.main_sun = Some(
        SunLight {
            pitch: 0.118,
            yaw: 0.783,
            pitch_speed: 0.003,
            yaw_speed: 0.0,
            intensity: 2.5,
            shadow_map: sun_shadow_map
        }
    );

    //Create pipelines
    let [vk_3D_graphics_pipeline, terrain_pipeline, atmosphere_pipeline, shadow_pipeline, postfx_pipeline] = unsafe {
        //Load shaders
        let main_shader_stages = {
            let v = vkutil::load_shader_stage(&vk.device, vk::ShaderStageFlags::VERTEX, "./data/shaders/vertex_main.spv");
            let f = vkutil::load_shader_stage(&vk.device, vk::ShaderStageFlags::FRAGMENT, "./data/shaders/pbr_metallic_roughness.spv");
            vec![v, f]
        };
        
        let terrain_shader_stages = {
            let v = vkutil::load_shader_stage(&vk.device, vk::ShaderStageFlags::VERTEX, "./data/shaders/vertex_main.spv");
            let f = vkutil::load_shader_stage(&vk.device, vk::ShaderStageFlags::FRAGMENT, "./data/shaders/terrain.spv");
            vec![v, f]
        };
        
        let atm_shader_stages = {
            let v = vkutil::load_shader_stage(&vk.device, vk::ShaderStageFlags::VERTEX, "./data/shaders/atmosphere_vert.spv");
            let f = vkutil::load_shader_stage(&vk.device, vk::ShaderStageFlags::FRAGMENT, "./data/shaders/atmosphere_frag.spv");
            vec![v, f]
        };

        let s_shader_stages = {
            let v = vkutil::load_shader_stage(&vk.device, vk::ShaderStageFlags::VERTEX, "./data/shaders/shadow_vert.spv");
            let f = vkutil::load_shader_stage(&vk.device, vk::ShaderStageFlags::FRAGMENT, "./data/shaders/shadow_frag.spv");
            vec![v, f]
        };

        let postfx_shader_stages = {
            let v = vkutil::load_shader_stage(&vk.device, vk::ShaderStageFlags::VERTEX, "./data/shaders/postfx_vert.spv");
            let f = vkutil::load_shader_stage(&vk.device, vk::ShaderStageFlags::FRAGMENT, "./data/shaders/postfx_frag.spv");
            vec![v, f]
        };

        let main_info = vkutil::GraphicsPipelineBuilder::init(hdr_forward_pass, pipeline_layout)
                        .set_shader_stages(main_shader_stages).build_info();
        let terrain_info = vkutil::GraphicsPipelineBuilder::init(hdr_forward_pass, pipeline_layout)
                            .set_shader_stages(terrain_shader_stages).build_info();
        let atm_info = vkutil::GraphicsPipelineBuilder::init(hdr_forward_pass, pipeline_layout)
                            .set_shader_stages(atm_shader_stages).build_info();
        let shadow_info = vkutil::GraphicsPipelineBuilder::init(shadow_pass, pipeline_layout)
                            .set_shader_stages(s_shader_stages).set_cull_mode(vk::CullModeFlags::NONE).build_info();
        let postfx_info = vkutil::GraphicsPipelineBuilder::init(swapchain_pass, pipeline_layout)
                            .set_shader_stages(postfx_shader_stages).build_info();
                            
    
        let infos = [main_info, terrain_info, atm_info, shadow_info, postfx_info];
        let pipelines = vkutil::GraphicsPipelineBuilder::create_pipelines(&mut vk, &infos);

        [
            pipelines[0],
            pipelines[1],
            pipelines[2],
            pipelines[3],
            pipelines[4]
        ]
    };

    let mut timescale_factor = 1.0;
    let terrain_generation_scale = 20.0;

    //Define terrain
    let mut terrain = TerrainSpec {
        vertex_width: 256,
        vertex_height: 256,
        amplitude: 2.0,
        exponent: 2.2,
        seed: unix_epoch_ms(),
        ..Default::default()
    };

    let terrain_vertices = terrain.generate_vertices(terrain_generation_scale);
    let terrain_indices = ozy::prims::plane_index_buffer(terrain.vertex_width, terrain.vertex_height);

    let mut terrain_collider_handle = physics_engine.make_terrain_collider(&terrain_vertices.positions, terrain.vertex_width, terrain.vertex_height);
    
    //Loading terrain textures in a deferred way
    let mut deferred_images = Vec::with_capacity(64);
    let terrain_image_paths = [
        "./data/textures/whispy_grass/color.png",
        "./data/textures/whispy_grass/normal.png",
        "./data/textures/whispy_grass/ao_roughness_metallic.png",
        "./data/textures/rocky_ground/color.png",
        "./data/textures/rocky_ground/normal.png",
        "./data/textures/rocky_ground/ao_roughness_metallic.png"
    ];
    let mut terrain_image_indices = [0; 6];
    for i in 0..terrain_image_paths.len() {
        let image = GPUImage::from_png_file_deferred(&mut vk, terrain_image_paths[i]);
        terrain_image_indices[i] = vkutil::make_global_texture_descriptor(
            &mut renderer.global_textures,
            renderer.material_sampler,
            image.final_image.view
        );
        deferred_images.push(image);
    }
    let [grass_color_index, grass_normal_index, grass_arm_index, rock_color_index, rock_normal_index, rock_arm_index] = terrain_image_indices;

    //Compress grass color texture
    unsafe {
        use ozy::io::{D3D10_RESOURCE_DIMENSION, DXGI_FORMAT, compute_pitch_bc};

        let mut file = unwrap_result(File::open("./data/textures/whispy_grass/color.png"), &format!("Error opening png {}", "./data/textures/whispy_grass/color.png"));
        let mut png_bytes = vec![0u8; file.metadata().unwrap().len().try_into().unwrap()];
        file.read_exact(&mut png_bytes).unwrap();
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
        let mip_levels = (f32::floor(f32::log2(u32::max(width, height) as f32))) as u32;
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
            usage: vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            queue_family_index_count: 1,
            p_queue_family_indices: &vk.queue_family_index,
            initial_layout: vk::ImageLayout::UNDEFINED,
            ..Default::default()
        };
        
        let gpu_image_layout = vk::ImageLayout::TRANSFER_SRC_OPTIMAL;
        let def_image = upload_image_deferred(&mut vk, &image_create_info, gpu_image_layout, &bytes);
        let def_image = &DeferredImage::synchronize(&mut vk, vec![def_image])[0];

        //Get the GPU image back into system RAM
        let finished_image_reqs = vk.device.get_image_memory_requirements(def_image.final_image.image);
        let readback_buffer = GPUBuffer::allocate(&mut vk, finished_image_reqs.size, finished_image_reqs.alignment, vk::BufferUsageFlags::TRANSFER_DST, MemoryLocation::CpuToGpu);
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
            current_offset += (w * h) as u64;
        }
        vk.device.cmd_copy_image_to_buffer(command_buffer, def_image.final_image.image, gpu_image_layout, readback_buffer.backing_buffer(), &regions);

        vk.device.end_command_buffer(command_buffer).unwrap();
        vk.command_buffer_indices.remove(cb_idx);

        let surface = ispc::RgbaSurface {
            data: &bytes,
            width,
            height,
            stride: 4 * width
        };

        let settings = ispc::bc7::opaque_basic_settings();
        let bc7_bytes = ispc::bc7::compress_blocks(&settings, &surface);
        let dds_pixelformat = DDS_PixelFormat {
            rgb_bitcount,
            flags: DDS_PixelFormat::DDPF_FOURCC,
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
            mipmap_count: 1,
            spf: dds_pixelformat,
            dx10_header,
            ..Default::default()
        };

        let mut out_file = OpenOptions::new().write(true).create(true).open("./data/textures/whispy_grass/color_compressed.dds").unwrap();
        out_file.write(struct_to_bytes(&dds_header)).unwrap();
        out_file.write(&bc7_bytes).unwrap();
    }

    //let grass_color_global_index = vkutil::load_global_bc7(&mut vk, &mut renderer.global_textures, renderer.material_sampler, "./data/textures/whispy_grass/color.dds", ColorSpace::SRGB);
    //let grass_normal_global_index = vkutil::load_global_bc7(&mut vk, &mut renderer.global_textures, renderer.material_sampler, "./data/textures/whispy_grass/normal.dds", ColorSpace::LINEAR);
    //let grass_metalrough_global_index = vkutil::load_global_bc7(&mut vk, &mut renderer.global_textures, renderer.material_sampler, "./data/textures/whispy_grass/metallic_roughness.dds", ColorSpace::LINEAR);
    //let rock_color_global_index = vkutil::load_global_bc7(&mut vk, &mut renderer.global_textures, renderer.material_sampler, "./data/textures/rocky_ground/color.dds", ColorSpace::SRGB);
    //let rock_normal_global_index = vkutil::load_global_bc7(&mut vk, &mut renderer.global_textures, renderer.material_sampler, "./data/textures/rocky_ground/normal.dds", ColorSpace::LINEAR);
    //let rock_metalrough_global_index = vkutil::load_global_bc7(&mut vk, &mut renderer.global_textures, renderer.material_sampler, "./data/textures/rocky_ground/metallic_roughness.dds", ColorSpace::LINEAR);
    
    let terrain_grass_matidx = renderer.global_materials.insert(
        Material {
            pipeline: terrain_pipeline,
            base_color:  [1.0; 4],
            base_roughness: 1.0,
            color_idx: grass_color_index,
            normal_idx: grass_normal_index,
            metal_roughness_idx: grass_arm_index,
            emissive_idx: renderer.default_emissive_idx
        }
    ) as u32;
    let terrain_rock_matidx = renderer.global_materials.insert(
        Material {
            pipeline: terrain_pipeline,
            base_color:  [1.0; 4],
            base_roughness: 1.0,
            color_idx: rock_color_index,
            normal_idx: rock_normal_index,
            metal_roughness_idx: rock_arm_index,
            emissive_idx: renderer.default_emissive_idx
        }
    ) as u32;
    
    //Upload terrain geometry
    //let terrain_model_idx = 0;
    let terrain_model_idx = {
        let terrain_offsets = upload_vertex_attributes(&mut vk, &mut renderer, &terrain_vertices);
        drop(terrain_vertices);
        let index_buffer = routines::make_index_buffer(&mut vk, &terrain_indices);
        renderer.register_primitive(Primitive {
            shadow_type: ShadowType::OpaqueCaster,
            index_buffer,
            index_count: terrain_indices.len().try_into().unwrap(),
            position_offset: terrain_offsets.position_offset,
            tangent_offset: terrain_offsets.tangent_offset,
            normal_offset: terrain_offsets.normal_offset,
            uv_offset: terrain_offsets.uv_offset,
            material_idx: terrain_grass_matidx,
        })
    };

    let mut lookat_dist = 7.5;
    let mut lookat_pos = lookat_dist * glm::normalize(&glm::vec3(-1.0f32, 0.0, 1.75));

    //let test_scene_data = gltfutil::gltf_scenedata("./data/models/sponza2k.glb");
    //println!("Meshes in file: {}", test_scene_data.meshes.len());

    //Load totoro as glb
    let totoro_data = gltfutil::gltf_meshdata("./data/models/totoro_backup.glb");

    //Register each primitive with the renderer
    let totoro_model_indices = upload_gltf_primitives(&mut vk, &mut renderer, &totoro_data, vk_3D_graphics_pipeline);

    //Make totoro collider
    let mut totoro_list = OptionVec::new();
    let main_totoro_idx = {
        let rigid_body = RigidBodyBuilder::dynamic()
        .translation(glm::vec3(0.0, 0.0, 20.0))
        .ccd_enabled(true)
        .build();
        let collider = ColliderBuilder::ball(2.1).restitution(2.5).build();
        let rigid_body_handle = physics_engine.rigid_body_set.insert(rigid_body);
        let collider_handle = physics_engine.collider_set.insert_with_parent(collider, rigid_body_handle, &mut physics_engine.rigid_body_set);
        totoro_list.insert(PhysicsProp {
            rigid_body_handle,
            collider_handle
        })
    };

    let mut static_props = DenseSlotMap::<_, StaticProp>::new();

    //Create semaphore used to wait on swapchain image
    let vk_swapchain_semaphore = unsafe { vk.device.create_semaphore(&vk::SemaphoreCreateInfo::default(), vkutil::MEMORY_ALLOCATOR).unwrap() };

    //State for freecam controls
    let mut camera = Camera::new(glm::vec3(0.0f32, -30.0, 15.0));
    let mut last_view_from_world = glm::identity();
    let mut do_freecam = false;

    let mut timer = FrameTimer::new();      //Struct for doing basic framerate independence

    renderer.uniform_data.sun_luminance = glm::vec4(1.0, 1.0, 1.0, 0.0);
    renderer.uniform_data.ambient_factor = 0.1;
    renderer.uniform_data.stars_threshold = 8.0;
    renderer.uniform_data.stars_exposure = 200.0;
    renderer.uniform_data.fog_density = 2.8;
    
    //Load and play bgm
    let bgm = unwrap_result(Music::from_file("./data/music/relaxing_botw.mp3"), "Error loading bgm");
    bgm.play(-1).unwrap();

    let mut dev_gui = DevGui::new(&mut vk, swapchain_pass, pipeline_layout);

    let mut input_system = input::InputSystem::init(&sdl_context);

    //Synchronize with deferred asset loading
    DeferredImage::synchronize(&mut vk, deferred_images);

    //Main application loop
    'running: loop {
        timer.update(); //Update frame timer
        let scaled_delta_time = if timer.delta_time > 1.0 / 30.0 {
            timescale_factor / 30.0
        } else {
            timer.delta_time * timescale_factor
        };

        //Reset renderer
        renderer.reset();

        //Input sampling
        let imgui_io = imgui_context.io_mut();
        let input_output = match input_system.do_thing(&timer, imgui_io) {
            UserInput::Output(o) => { o }
            UserInput::ExitProgram => { break 'running; }
        };

        //Handling of some input results before update
        if input_output.gui_toggle {
            dev_gui.do_gui = !dev_gui.do_gui
        }
        if input_output.regen_terrain {
            regenerate_terrain(
                &mut vk,
                &mut renderer,
                &mut physics_engine,
                &mut terrain_collider_handle,
                terrain_model_idx,
                &mut terrain,
                terrain_generation_scale
            );
        }
        if input_output.reset_totoro {
            reset_totoro(&mut physics_engine, &totoro_list[main_totoro_idx]);
        }

        if input_output.spawn_totoro_prop {
            let view_mat = camera.make_view_matrix();
            let dir = glm::vec3(0.0, 0.0, -1.0);
            let shoot_dir = glm::vec4_to_vec3(&(glm::affine_inverse(view_mat) * glm::vec3_to_vec4(&dir)));
            let init_pos = camera.position + 5.0 * shoot_dir;
            let totoro_prop = {
                let rigid_body = RigidBodyBuilder::dynamic()
                .translation(init_pos)
                .linvel(shoot_dir * 40.0)
                .ccd_enabled(true)
                .build();
                let collider = ColliderBuilder::ball(2.25).restitution(0.9).build();
                let rigid_body_handle = physics_engine.rigid_body_set.insert(rigid_body);
                let collider_handle = physics_engine.collider_set.insert_with_parent(collider, rigid_body_handle, &mut physics_engine.rigid_body_set);
                PhysicsProp {
                    rigid_body_handle,
                    collider_handle
                }
            };
            totoro_list.insert(totoro_prop);
        }

        //Handle needing to resize the window
        unsafe {
            if input_output.resize_window {
                //Window resizing requires us to "flush the pipeline" as it were. wahh
                vk.device.wait_for_fences(&renderer.in_flight_fences(), true, vk::DeviceSize::MAX).unwrap();

                //Free the now-invalid swapchain data
                for framebuffer in renderer.window_manager.swapchain_framebuffers {
                    vk.device.destroy_framebuffer(framebuffer, vkutil::MEMORY_ALLOCATOR);
                }
                for view in renderer.window_manager.swapchain_image_views {
                    vk.device.destroy_image_view(view, vkutil::MEMORY_ALLOCATOR);
                }
                vk.ext_swapchain.destroy_swapchain(renderer.window_manager.swapchain, vkutil::MEMORY_ALLOCATOR);

                //Recreate swapchain and associated data
                renderer.window_manager = render::WindowManager::init(&mut vk, &window, swapchain_pass);

                //Recreate internal rendering buffers
                let extent = vk::Extent3D {
                    width: renderer.window_manager.extent.width,
                    height: renderer.window_manager.extent.height,
                    depth: 1
                };
                renderer.resize_hdr_framebuffers(&mut vk, extent, hdr_forward_pass);

                window_size = glm::vec2(renderer.window_manager.extent.width, renderer.window_manager.extent.height);
                imgui_io.display_size[0] = window_size.x as f32;
                imgui_io.display_size[1] = window_size.y as f32;
            }
        }

        {
            let mouse_util = sdl_context.mouse();
            mouse_util.set_relative_mouse_mode(input_system.cursor_captured);
            if input_system.cursor_captured {
                mouse_util.warp_mouse_in_window(&window, window_size.x as i32 / 2, window_size.y as i32 / 2);
            }
        }

        // --- Sim Update ---

        let imgui_ui = imgui_context.frame();   //Transition Dear ImGUI into recording state

        //Terrain generation window
        if dev_gui.do_terrain_window(&imgui_ui, &mut terrain) {
            regenerate_terrain(
                &mut vk,
                &mut renderer,
                &mut physics_engine,
                &mut terrain_collider_handle,
                terrain_model_idx,
                &mut terrain,
                terrain_generation_scale
            );
        }

        if dev_gui.do_gui {
            if let Some(t) = imgui::Window::new("Main control panel (press ESC to hide)").menu_bar(true).begin(&imgui_ui) {
                if let Some(mb) = imgui_ui.begin_menu_bar() {
                    if let Some(mt) = imgui_ui.begin_menu("File") {
                        if imgui::MenuItem::new("New").build(&imgui_ui) {}
                        if imgui::MenuItem::new("Load").build(&imgui_ui) {}
                        if imgui::MenuItem::new("Save").build(&imgui_ui) {}
                        if imgui::MenuItem::new("Quit").build(&imgui_ui) { break 'running; }
                        mt.end();
                    }
                    if let Some(mt) = imgui_ui.begin_menu("Debug") {
                        if imgui::MenuItem::new("Active material list").build(&imgui_ui) { dev_gui.do_mat_list = true; }
                        mt.end();
                    }
                    if let Some(mt) = imgui_ui.begin_menu("Environment") {
                        if imgui::MenuItem::new("Props window").build(&imgui_ui) { dev_gui.do_props_window = true; }
                        if imgui::MenuItem::new("Terrain generator").build(&imgui_ui) { dev_gui.do_terrain_window = true; }
                        mt.end();
                    }
                    mb.end();
                }
    
                imgui_ui.text(format!("Rendering at {:.0} FPS ({:.2} ms frametime, frame {})", input_output.framerate, 1000.0 / input_output.framerate, timer.frame_count));
                
                let (message, color) =  match input_system.controllers[0] {
                    Some(_) => { ("Controller is connected.", [0.0, 1.0, 0.0, 1.0]) }
                    None => { ("Controller is not connected.", [1.0, 0.0, 0.0, 1.0]) }
                };
                let color_token = imgui_ui.push_style_color(imgui::StyleColor::Text, color);
                imgui_ui.text(message);
                color_token.pop();
    
                if let Some(sun) = &mut renderer.main_sun {
                    imgui::Slider::new("Sun pitch speed", 0.0, 1.0).build(&imgui_ui, &mut sun.pitch_speed);
                    imgui::Slider::new("Sun pitch", 0.0, glm::two_pi::<f32>()).build(&imgui_ui, &mut sun.pitch);
                    imgui::Slider::new("Sun yaw speed", -1.0, 1.0).build(&imgui_ui, &mut sun.yaw_speed);
                    imgui::Slider::new("Sun yaw", 0.0, glm::two_pi::<f32>()).build(&imgui_ui, &mut sun.yaw);
                    imgui::Slider::new("Sun intensity", 0.0, 20.0).build(&imgui_ui, &mut sun.intensity);
                }
                
                imgui::Slider::new("Ambient factor", 0.0, 20.0).build(&imgui_ui, &mut renderer.uniform_data.ambient_factor);    
                imgui::Slider::new("Stars threshold", 0.0, 16.0).build(&imgui_ui, &mut renderer.uniform_data.stars_threshold);
                imgui::Slider::new("Stars exposure", 0.0, 1000.0).build(&imgui_ui, &mut renderer.uniform_data.stars_exposure);
                imgui::Slider::new("Fog factor", 0.0, 8.0).build(&imgui_ui, &mut renderer.uniform_data.fog_density);
                imgui::Slider::new("Camera exposure", 0.0, 5.0).build(&imgui_ui, &mut renderer.uniform_data.exposure);
                imgui::Slider::new("Timescale factor", 0.001, 8.0).build(&imgui_ui, &mut timescale_factor);
    
                if imgui::Slider::new("Music volume", 0, 128).build(&imgui_ui, &mut music_volume) { Music::set_volume(music_volume); }
                imgui_ui.checkbox("Freecam", &mut do_freecam);
    
                imgui_ui.text(format!("Freecam is at ({:.4}, {:.4}, {:.4})", camera.position.x, camera.position.y, camera.position.z));
                
                if DevGui::do_standard_button(&imgui_ui, "Totoro's be gone") {
                    for i in 1..totoro_list.len() {
                        totoro_list.delete(i);
                    }
                }
    
                t.end();
            }
        }

        if let Some(mesh_data) = dev_gui.do_props_window(&imgui_ui, &mut static_props) {
            let model_indices = upload_gltf_primitives(&mut vk, &mut renderer, &mesh_data, vk_3D_graphics_pipeline);
            let s = StaticProp {
                name: mesh_data.name,
                model_indices,
                position: camera.position,
                ..Default::default()
            };
            static_props.insert(s);
        }

        dev_gui.do_material_list(&imgui_ui, &mut renderer);

        //Step the physics engine
        physics_engine.integration_parameters.dt = scaled_delta_time;
        physics_engine.step();

        let plane_model_matrix = glm::identity();
        renderer.queue_drawcall(terrain_model_idx, &[plane_model_matrix]);

        let view_movement_vector = glm::mat4(
            1.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, -1.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 1.0
        ) * glm::vec3_to_vec4(&input_output.movement_vector);
        const FREECAM_SPEED: f32 = 3.0;
        if do_freecam {
            let delta_pos = FREECAM_SPEED * glm::affine_inverse(last_view_from_world) * view_movement_vector * timer.delta_time;
            camera.position += glm::vec4_to_vec3(&delta_pos);
            camera.orientation += input_output.orientation_delta;
        }
 
        //Totoros update
        let mut matrices = vec![];
        for prop in totoro_list.iter() {
            if let Some(p) = prop {
                if glm::distance(physics_engine.rigid_body_set[p.rigid_body_handle].translation(), &glm::zero()) > 750.0 {
                    reset_totoro(&mut physics_engine, &totoro_list[main_totoro_idx]);
                }
                let body_transform = physics_engine.rigid_body_set[p.rigid_body_handle].position().to_matrix();
                let matrix = body_transform * glm::translation(&glm::vec3(0.0, 0.0, -2.25));
                matrices.push(matrix);
            }
        }
        for idx in &totoro_model_indices {
            renderer.queue_drawcall(*idx, &matrices);
        }

        //Compute this frame's view matrix
        let view_from_world = if do_freecam {
            //Camera orientation based on user input
            camera.orientation.y = camera.orientation.y.clamp(-glm::half_pi::<f32>(), glm::half_pi::<f32>());
            renderer.uniform_data.camera_position = glm::vec4(camera.position.x, camera.position.y, camera.position.z, 1.0);
            camera.make_view_matrix()
        } else {
            let min = 3.0;
            let max = 200.0;
            lookat_dist -= 0.1 * lookat_dist * input_output.scroll_amount;
            lookat_dist = f32::clamp(lookat_dist, min, max);
            
            let lookat = glm::look_at(&lookat_pos, &glm::zero(), &glm::vec3(0.0, 0.0, 1.0));
            let world_space_offset = glm::affine_inverse(lookat) * glm::vec4(-input_output.orientation_delta.x, input_output.orientation_delta.y, 0.0, 0.0);

            lookat_pos += lookat_dist * glm::vec4_to_vec3(&world_space_offset);
            let camera_pos = glm::normalize(&lookat_pos);
            lookat_pos = lookat_dist * camera_pos;
            
            let min = -0.95;
            let max = 0.95;
            let lookat_dot = glm::dot(&camera_pos, &glm::vec3(0.0, 0.0, 1.0));
            if lookat_dot > max {
                let rotation_vector = -glm::cross(&camera_pos, &glm::vec3(0.0, 0.0, 1.0));
                let current_angle = f32::acos(lookat_dot);
                let amount = f32::acos(max) - current_angle;

                let new_pos = glm::rotation(amount, &rotation_vector) * glm::vec3_to_vec4(&lookat_pos);
                lookat_pos = glm::vec4_to_vec3(&new_pos);
            } else if lookat_dot < min {
                let rotation_vector = -glm::cross(&camera_pos, &glm::vec3(0.0, 0.0, 1.0));
                let current_angle = f32::acos(lookat_dot);
                let amount = f32::acos(min) - current_angle;

                let new_pos = glm::rotation(amount, &rotation_vector) * glm::vec3_to_vec4(&(lookat_pos));                
                lookat_pos = glm::vec4_to_vec3(&new_pos);
            }

            let collider = physics_engine.collider_set.get(totoro_list[main_totoro_idx].as_ref().unwrap().collider_handle).unwrap();
            let t = collider.position().translation;
            let lookat_target = t.vector;
            let pos = lookat_pos + lookat_target;
            let m = glm::look_at(&pos, &lookat_target, &glm::vec3(0.0, 0.0, 1.0));
            renderer.uniform_data.camera_position = glm::vec4(pos.x, pos.y, pos.z, 1.0);
            m
        };
        last_view_from_world = view_from_world;
        renderer.uniform_data.view_from_world = view_from_world;

        //Push drawcalls for static props
        for (_, prop) in static_props.iter() {
            for idx in prop.model_indices.iter() {
                let mm = glm::translation(&prop.position) *
                         glm::rotation(prop.pitch, &glm::vec3(1.0, 0.0, 0.0)) *
                         glm::rotation(prop.yaw, &glm::vec3(0.0, 0.0, 1.0)) *
                         glm::rotation(prop.roll, &glm::vec3(0.0, 1.0, 0.0));
                renderer.queue_drawcall(*idx, &[mm]);
            }
        }
        
        //Update sun
        if let Some(sun) = &mut renderer.main_sun {
            sun.pitch += sun.pitch_speed * scaled_delta_time;
            sun.yaw += sun.yaw_speed * scaled_delta_time;
            if sun.pitch > glm::two_pi() {
                sun.pitch -= glm::two_pi::<f32>();
            }
            if sun.pitch < 0.0 {
                sun.pitch += glm::two_pi::<f32>();
            }
            if sun.yaw > glm::two_pi() {
                sun.yaw -= glm::two_pi::<f32>();
            }
            if sun.yaw < 0.0 {
                sun.yaw += glm::two_pi::<f32>();
            }
        }

        //Resolve the current Dear Imgui frame
        dev_gui.resolve_imgui_frame(&mut vk, &mut renderer, imgui_ui);

        //Draw
        unsafe {
            //Begin acquiring swapchain. This is called as early as possible in order to minimize time waiting
            let current_framebuffer_index = vk.ext_swapchain.acquire_next_image(renderer.window_manager.swapchain, vk::DeviceSize::MAX, vk_swapchain_semaphore, vk::Fence::null()).unwrap().0 as usize;
            
            //Does all work that needs to happen before the render pass
            let frame_info = renderer.prepare_frame(&mut vk, window_size, &view_from_world, timer.elapsed_time);

            //Put command buffer in recording state
            vk.device.begin_command_buffer(frame_info.main_command_buffer, &vk::CommandBufferBeginInfo::default()).unwrap();

            //Bindless descriptor setup for Shadow+HDR pass
            let dynamic_uniform_offset = renderer.current_in_flight_frame() as u64 * size_to_alignment!(size_of::<render::EnvironmentUniforms>() as u64, vk.physical_device_properties.limits.min_uniform_buffer_offset_alignment);
            
            //Shadow rendering
            if let Some(sun) = &renderer.main_sun {
                let sun_shadow_map = &sun.shadow_map;
                let render_area = {
                    vk::Rect2D {
                        offset: vk::Offset2D { x: 0, y: 0 },
                        extent: vk::Extent2D {
                            width: sun_shadow_map.resolution(),
                            height: sun_shadow_map.resolution() 
                        }
                    }
                };
                vk.device.cmd_set_scissor(frame_info.main_command_buffer, 0, &[render_area]);
                let clear_values = [vkutil::DEPTH_STENCIL_CLEAR];
                let rp_begin_info = vk::RenderPassBeginInfo {
                    render_pass: shadow_pass,
                    framebuffer: sun_shadow_map.framebuffer(),
                    render_area,
                    clear_value_count: clear_values.len() as u32,
                    p_clear_values: clear_values.as_ptr(),
                    ..Default::default()
                };

                vk.device.cmd_begin_render_pass(frame_info.main_command_buffer, &rp_begin_info, vk::SubpassContents::INLINE);
                vk.device.cmd_bind_descriptor_sets(
                    frame_info.main_command_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    pipeline_layout,
                    0,
                    &[renderer.bindless_descriptor_set],
                    &[dynamic_uniform_offset as u32, frame_info.instance_data_start_offset as u32]
                );
                vk.device.cmd_bind_pipeline(frame_info.main_command_buffer, vk::PipelineBindPoint::GRAPHICS, shadow_pipeline);

                let viewport = vk::Viewport {
                    x: 0.0,
                    y: 0.0,
                    width: sun_shadow_map.resolution() as f32,
                    height: sun_shadow_map.resolution() as f32,
                    min_depth: 0.0,
                    max_depth: 1.0
                };
                vk.device.cmd_set_viewport(frame_info.main_command_buffer, 0, &[viewport]);
                for drawcall in renderer.drawlist_iter() {
                    if let Some(model) = renderer.get_model(drawcall.geometry_idx) {
                        if let ShadowType::NonCaster = model.shadow_type { continue; }
    
                        let pcs = [
                            model.material_idx.to_le_bytes(),
                            model.position_offset.to_le_bytes(),
                            model.uv_offset.to_le_bytes()
                        ].concat();
                        vk.device.cmd_push_constants(frame_info.main_command_buffer, pipeline_layout, push_constant_stage_flags, 0, &pcs);
                        vk.device.cmd_bind_index_buffer(frame_info.main_command_buffer, model.index_buffer.backing_buffer(), 0, vk::IndexType::UINT32);
                        vk.device.cmd_draw_indexed(frame_info.main_command_buffer, model.index_count, drawcall.instance_count, 0, 0, drawcall.first_instance);
                    }
                }

                vk.device.cmd_end_render_pass(frame_info.main_command_buffer);
            }
            
            //Set the viewport for this frame
            let viewport = vk::Viewport {
                x: 0.0,
                y: 0.0,
                width: (renderer.window_manager.extent.width) as f32,
                height: (renderer.window_manager.extent.height) as f32,
                min_depth: 0.0,
                max_depth: 1.0
            };
            vk.device.cmd_set_viewport(frame_info.main_command_buffer, 0, &[viewport]);

            //Set scissor rect to be same as render area
            let vk_render_area = {
                vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent: renderer.window_manager.extent
                }
            };
            let scissor_area = vk::Rect2D {
                offset: vk::Offset2D::default(),
                extent: vk::Extent2D {
                    width: window_size.x,
                    height: window_size.y
                }
            };
            vk.device.cmd_set_scissor(frame_info.main_command_buffer, 0, &[scissor_area]);

            let vk_clear_values = [vkutil::COLOR_CLEAR, vkutil::DEPTH_STENCIL_CLEAR];

            //HDR render pass recording
            let rp_begin_info = vk::RenderPassBeginInfo {
                render_pass: hdr_forward_pass,
                framebuffer: frame_info.framebuffer.framebuffer_object,
                render_area: vk_render_area,
                clear_value_count: vk_clear_values.len() as u32,
                p_clear_values: vk_clear_values.as_ptr(),
                ..Default::default()
            };
            vk.device.cmd_begin_render_pass(frame_info.main_command_buffer, &rp_begin_info, vk::SubpassContents::INLINE);
            vk.device.cmd_bind_descriptor_sets(
                frame_info.main_command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                pipeline_layout,
                0,
                &[renderer.bindless_descriptor_set],
                &[dynamic_uniform_offset as u32, frame_info.instance_data_start_offset as u32]
            );

            //Iterate through draw calls
            let mut last_bound_pipeline = vk::Pipeline::default();
            for drawcall in renderer.drawlist_iter() {
                if drawcall.pipeline != last_bound_pipeline {
                    vk.device.cmd_bind_pipeline(frame_info.main_command_buffer, vk::PipelineBindPoint::GRAPHICS, drawcall.pipeline);
                    last_bound_pipeline = drawcall.pipeline;
                }
                if let Some(model) = renderer.get_model(drawcall.geometry_idx) {
                    let pcs = [
                        model.material_idx.to_le_bytes(),
                        model.position_offset.to_le_bytes(),
                        model.tangent_offset.to_le_bytes(),
                        model.normal_offset.to_le_bytes(),
                        model.uv_offset.to_le_bytes(),
                    ].concat();
                    vk.device.cmd_push_constants(frame_info.main_command_buffer, pipeline_layout, push_constant_stage_flags, 0, &pcs);
                    vk.device.cmd_bind_index_buffer(frame_info.main_command_buffer, model.index_buffer.backing_buffer(), 0, vk::IndexType::UINT32);
                    vk.device.cmd_draw_indexed(frame_info.main_command_buffer, model.index_count, drawcall.instance_count, 0, 0, drawcall.first_instance);
                }
            }

            //Record atmosphere rendering commands
            vk.device.cmd_bind_pipeline(frame_info.main_command_buffer, vk::PipelineBindPoint::GRAPHICS, atmosphere_pipeline);
            vk.device.cmd_draw(frame_info.main_command_buffer, 36, 1, 0, 0);

            vk.device.cmd_end_render_pass(frame_info.main_command_buffer);
            vk.device.end_command_buffer(frame_info.main_command_buffer).unwrap();

            //Submit Shadow+HDR passes
            let submit_info = vk::SubmitInfo {
                signal_semaphore_count: 1,
                p_signal_semaphores: &frame_info.semaphore,
                command_buffer_count: 1,
                p_command_buffers: &frame_info.main_command_buffer,
                ..Default::default()
            };
            let queue = vk.device.get_device_queue(vk.queue_family_index, 0);
            vk.device.queue_submit(queue, &[submit_info], vk::Fence::default()).unwrap();

            //PostFX pass
            vk.device.begin_command_buffer(frame_info.swapchain_command_buffer, &vk::CommandBufferBeginInfo::default()).unwrap();

            //Bindless descriptor setup for the swapchain command buffer
            vk.device.cmd_bind_descriptor_sets(
                frame_info.swapchain_command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                pipeline_layout,
                0,
                &[renderer.bindless_descriptor_set],
                &[dynamic_uniform_offset as u32, frame_info.instance_data_start_offset as u32]
            );
            
            //Set the viewport/scissor for the swapchain command buffer
            vk.device.cmd_set_viewport(frame_info.swapchain_command_buffer, 0, &[viewport]);
            vk.device.cmd_set_scissor(frame_info.swapchain_command_buffer, 0, &[scissor_area]);

            let rp_begin_info = vk::RenderPassBeginInfo {
                render_pass: swapchain_pass,
                framebuffer: renderer.window_manager.swapchain_framebuffers[current_framebuffer_index],
                render_area: vk_render_area,
                clear_value_count: vk_clear_values.len() as u32,
                p_clear_values: vk_clear_values.as_ptr(),
                ..Default::default()
            };
            vk.device.cmd_begin_render_pass(frame_info.swapchain_command_buffer, &rp_begin_info, vk::SubpassContents::INLINE);

            vk.device.cmd_bind_pipeline(frame_info.swapchain_command_buffer, vk::PipelineBindPoint::GRAPHICS, postfx_pipeline);
            vk.device.cmd_push_constants(frame_info.swapchain_command_buffer, pipeline_layout, push_constant_stage_flags, 0, &frame_info.framebuffer.texture_index.to_le_bytes());
            vk.device.cmd_draw(frame_info.swapchain_command_buffer, 3, 1, 0, 0);

            //Record Dear ImGUI drawing commands
            dev_gui.record_draw_commands(&mut vk, frame_info.swapchain_command_buffer, pipeline_layout);

            vk.device.cmd_end_render_pass(frame_info.swapchain_command_buffer);

            vk.device.end_command_buffer(frame_info.swapchain_command_buffer).unwrap();

            let submit_info = vk::SubmitInfo {
                wait_semaphore_count: 1,
                p_wait_semaphores: &frame_info.semaphore,
                p_wait_dst_stage_mask: &vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                signal_semaphore_count: 1,
                p_signal_semaphores: &frame_info.semaphore,
                command_buffer_count: 1,
                p_command_buffers: &frame_info.swapchain_command_buffer,
                ..Default::default()
            };

            let queue = vk.device.get_device_queue(vk.queue_family_index, 0);
            vk.device.reset_fences(&[frame_info.fence]).unwrap();
            vk.device.queue_submit(queue, &[submit_info], frame_info.fence).unwrap();

            let present_info = vk::PresentInfoKHR {
                swapchain_count: 1,
                p_swapchains: &renderer.window_manager.swapchain,
                p_image_indices: &(current_framebuffer_index as u32),
                wait_semaphore_count: 1,
                p_wait_semaphores: &frame_info.semaphore,
                ..Default::default()
            };
            if let Err(e) = vk.ext_swapchain.queue_present(queue, &present_info) {
                println!("{}", e);
            }
        }
    }

    //Cleanup
    unsafe {
        renderer.cleanup(&mut vk);
    }
}
