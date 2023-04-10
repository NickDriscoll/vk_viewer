#![allow(non_snake_case)]

//Alias some library names
extern crate jpeg_decoder as jpg;
extern crate ispc_texcomp as ispc;
extern crate nalgebra_glm as glm;
extern crate ozy_engine as ozy;
extern crate tinyfiledialogs as tfd;

mod asset;
mod gui;
mod input;
mod physics;
mod routines;
mod structs;

#[macro_use]
mod render;

use ::function_name::named;
use ash::vk::{self};
use gpu_allocator::MemoryLocation;
use gui::AssetWindowResponse;
use imgui::{SliderFlags};
use ozy::io::{DDSHeader, DDSHeader_DXT10, DDS_PixelFormat, OzyMesh};
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
use std::path::Path;
use std::ptr;
use std::time::{SystemTime};

use ozy::structs::{FrameTimer, OptionVec};

use input::{InputSystemOutput, InputSystem};
use physics::{PhysicsEngine, PhysicsComponent};
use structs::{Camera, TerrainSpec, Simulation};
use render::vkdevice::{self, msaa_samples_from_limit};
use render::{Primitive, Renderer, Material, CascadedShadowMap, ShadowType, SunLight};

use crate::routines::*;
use crate::asset::GLTFMeshData;
use crate::gui::{DevGui, EntityWindowResponse};
use crate::structs::Entity;

fn main() {
    //Create the window using SDL
    let sdl_context = unwrap_result(sdl2::init(), "Error initializing SDL");
    let video_subsystem = unwrap_result(sdl_context.video(), "Error initializing SDL video subsystem");
    let mut window_size = glm::vec2(1920, 1080);
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
        
        //Set up keyboard index map
        use sdl2::keyboard::Scancode;
        io.key_map[imgui::Key::Tab as usize] = Scancode::Tab as u32;
        io.key_map[imgui::Key::LeftArrow as usize] = Scancode::Left as u32;
        io.key_map[imgui::Key::RightArrow as usize] = Scancode::Right as u32;
        io.key_map[imgui::Key::UpArrow as usize] = Scancode::Up as u32;
        io.key_map[imgui::Key::DownArrow as usize] = Scancode::Down as u32;
        io.key_map[imgui::Key::PageDown as usize] = Scancode::PageDown as u32;
        io.key_map[imgui::Key::PageUp as usize] = Scancode::PageUp as u32;
        io.key_map[imgui::Key::Home as usize] = Scancode::Home as u32;
        io.key_map[imgui::Key::End as usize] = Scancode::End as u32;
        io.key_map[imgui::Key::Insert as usize] = Scancode::Insert as u32;
        io.key_map[imgui::Key::Delete as usize] = Scancode::Delete as u32;
        io.key_map[imgui::Key::Backspace as usize] = Scancode::Backspace as u32;
        io.key_map[imgui::Key::Space as usize] = Scancode::Space as u32;
        io.key_map[imgui::Key::Enter as usize] = Scancode::Return as u32;
        io.key_map[imgui::Key::KeypadEnter as usize] = Scancode::KpEnter as u32;
        io.key_map[imgui::Key::A as usize] = Scancode::A as u32;
        io.key_map[imgui::Key::C as usize] = Scancode::C as u32;
        io.key_map[imgui::Key::V as usize] = Scancode::V as u32;
        io.key_map[imgui::Key::X as usize] = Scancode::X as u32;
        io.key_map[imgui::Key::Y as usize] = Scancode::Y as u32;
        io.key_map[imgui::Key::Z as usize] = Scancode::Z as u32;
    }

    //Init the graphics device
    let mut gpu = vkdevice::VulkanGraphicsDevice::init();

    //Initialize the physics engine
    let mut physics_engine = PhysicsEngine::init();

    let shadow_pass = unsafe {
        let depth_description = vk::AttachmentDescription {
            format: vk::Format::D16_UNORM,
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

        //Create dependency between this shadow pass and the HDR pass
        let dependency = vk::SubpassDependency {
            src_subpass: 0,
            dst_subpass: vk::SUBPASS_EXTERNAL,
            src_stage_mask: vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
            dst_stage_mask: vk::PipelineStageFlags::FRAGMENT_SHADER,
            src_access_mask: vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
            dst_access_mask: vk::AccessFlags::SHADER_READ,
            dependency_flags: vk::DependencyFlags::empty()
        };

        let attachments = [depth_description];
        let renderpass_info = vk::RenderPassCreateInfo {
            p_next: &multiview_info as *const _ as *const c_void,
            attachment_count: attachments.len() as u32,
            p_attachments: attachments.as_ptr(),
            subpass_count: 1,
            p_subpasses: &subpass,
            dependency_count: 1,
            p_dependencies: &dependency,
            ..Default::default()
        };
        gpu.device.create_render_pass(&renderpass_info, vkdevice::MEMORY_ALLOCATOR).unwrap()
    };

    let hdr_forward_pass = unsafe {
        let msaa_samples = msaa_samples_from_limit(gpu.physical_device_properties.limits.framebuffer_color_sample_counts);

        let color_attachment_description = vk::AttachmentDescription {
            format: vk::Format::R16G16B16A16_SFLOAT,
            samples: msaa_samples,
            load_op: vk::AttachmentLoadOp::CLEAR,
            store_op: vk::AttachmentStoreOp::STORE,
            stencil_load_op: vk::AttachmentLoadOp::DONT_CARE,
            stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
            initial_layout: vk::ImageLayout::UNDEFINED,
            final_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            ..Default::default()
        };
        let depth_attachment_description = vk::AttachmentDescription {
            format: vk::Format::D32_SFLOAT,
            samples: msaa_samples,
            load_op: vk::AttachmentLoadOp::CLEAR,
            store_op: vk::AttachmentStoreOp::DONT_CARE,
            stencil_load_op: vk::AttachmentLoadOp::DONT_CARE,
            stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
            initial_layout: vk::ImageLayout::UNDEFINED,
            final_layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            ..Default::default()
        };
        let color_resolve_attachment_description = vk::AttachmentDescription {
            format: vk::Format::R16G16B16A16_SFLOAT,
            samples: vk::SampleCountFlags::TYPE_1,
            load_op: vk::AttachmentLoadOp::CLEAR,
            store_op: vk::AttachmentStoreOp::STORE,
            stencil_load_op: vk::AttachmentLoadOp::DONT_CARE,
            stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
            initial_layout: vk::ImageLayout::UNDEFINED,
            final_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            ..Default::default()
        };
        let attachments = [color_attachment_description, depth_attachment_description, color_resolve_attachment_description];

        let color_attachment_reference = vk::AttachmentReference {
            attachment: 0,
            layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
        };
        let depth_attachment_reference = vk::AttachmentReference {
            attachment: 1,
            layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL
        };
        let color_resolve_attachment_reference = vk::AttachmentReference {
            attachment: 2,
            layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
        };

        let subpass = vk::SubpassDescription {
            pipeline_bind_point: vk::PipelineBindPoint::GRAPHICS,
            color_attachment_count: 1,
            p_color_attachments: &color_attachment_reference,
            p_depth_stencil_attachment: &depth_attachment_reference,
            p_resolve_attachments: &color_resolve_attachment_reference,
            ..Default::default()
        };

        //Create dependencies between this pass and the PostFX pass
        let dependencies = [
            vk::SubpassDependency {
                src_subpass: 0,
                dst_subpass: vk::SUBPASS_EXTERNAL,
                src_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                dst_stage_mask: vk::PipelineStageFlags::FRAGMENT_SHADER,
                src_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                dst_access_mask: vk::AccessFlags::SHADER_READ,
                dependency_flags: vk::DependencyFlags::empty()
            }
        ];

        let renderpass_info = vk::RenderPassCreateInfo {
            attachment_count: attachments.len() as u32,
            p_attachments: attachments.as_ptr(),
            subpass_count: 1,
            p_subpasses: &subpass,
            dependency_count: dependencies.len() as u32,
            p_dependencies: dependencies.as_ptr(),
            ..Default::default()
        };
        gpu.device.create_render_pass(&renderpass_info, vkdevice::MEMORY_ALLOCATOR).unwrap()
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
        gpu.device.create_render_pass(&renderpass_info, vkdevice::MEMORY_ALLOCATOR).unwrap()
    };

    //Initialize the renderer
    let mut renderer = Renderer::init(&mut gpu, &window, swapchain_pass, hdr_forward_pass);

    //Create and upload Dear IMGUI font atlas
    unsafe {
        let atlas = imgui_context.fonts();
        let atlas_texture = atlas.build_alpha8_texture();
        let atlas_format = vk::Format::R8_UNORM;
        let atlas_layout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
        let gpu_image = vkdevice::upload_raw_image(&mut gpu, renderer.point_sampler, atlas_format, atlas_layout, atlas_texture.width, atlas_texture.height, atlas_texture.data);
        let index = renderer.global_images.insert(gpu_image);
        renderer.default_texture_idx = index as u32;
        
        atlas.clear_tex_data();                         //Free atlas memory CPU-side
        atlas.tex_id = imgui::TextureId::new(index);    //Giving Dear Imgui a reference to the font atlas GPU texture
    }

    let push_constant_stage_flags = vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT;
    let graphics_pipeline_layout = unsafe {
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
        
        gpu.device.create_pipeline_layout(&pipeline_layout_createinfo, vkdevice::MEMORY_ALLOCATOR).unwrap()
    };

    let compute_pipeline_layout = unsafe {
        let range = vk::PushConstantRange {
            stage_flags: vk::ShaderStageFlags::COMPUTE,
            offset: 0,
            size: 20
        };
        let pipeline_layout_createinfo = vk::PipelineLayoutCreateInfo {
            push_constant_range_count: 1,
            p_push_constant_ranges: &range,
            set_layout_count: 1,
            p_set_layouts: &renderer.descriptor_set_layout,
            ..Default::default()
        };
        
        gpu.device.create_pipeline_layout(&pipeline_layout_createinfo, vkdevice::MEMORY_ALLOCATOR).unwrap()
    };

    let sun_shadow_map = CascadedShadowMap::new(
        &mut gpu,
        &mut renderer,
        2048,
        shadow_pass
    );

    let sunlight_key = renderer.new_directional_light(
        SunLight {
            pitch: 1.030,
            yaw: 0.783,
            pitch_speed: 0.003,
            yaw_speed: 0.0,
            irradiance: 790.0f32 * glm::vec3(1.0, 0.891, 0.796),
            shadow_map: Some(sun_shadow_map)
        }
    ).unwrap();

    //Create graphics pipelines
    let [pbr_pipeline, terrain_pipeline, atmosphere_pipeline, shadow_pipeline, postfx_pipeline] = unsafe {
        use render::GraphicsPipelineBuilder;

        //Load shaders
        let main_shader_stages = {
            let v = vkdevice::load_shader_stage(&gpu.device, vk::ShaderStageFlags::VERTEX, "./data/shaders/vertex_main.spv");
            let f = vkdevice::load_shader_stage(&gpu.device, vk::ShaderStageFlags::FRAGMENT, "./data/shaders/pbr_metallic_roughness.spv");
            vec![v, f]
        };
        
        let terrain_shader_stages = {
            let v = vkdevice::load_shader_stage(&gpu.device, vk::ShaderStageFlags::VERTEX, "./data/shaders/vertex_main.spv");
            let f = vkdevice::load_shader_stage(&gpu.device, vk::ShaderStageFlags::FRAGMENT, "./data/shaders/terrain.spv");
            vec![v, f]
        };
        
        let atm_shader_stages = {
            let v = vkdevice::load_shader_stage(&gpu.device, vk::ShaderStageFlags::VERTEX, "./data/shaders/atmosphere_vert.spv");
            let f = vkdevice::load_shader_stage(&gpu.device, vk::ShaderStageFlags::FRAGMENT, "./data/shaders/atmosphere_frag.spv");
            vec![v, f]
        };

        let s_shader_stages = {
            let v = vkdevice::load_shader_stage(&gpu.device, vk::ShaderStageFlags::VERTEX, "./data/shaders/shadow_vert.spv");
            let f = vkdevice::load_shader_stage(&gpu.device, vk::ShaderStageFlags::FRAGMENT, "./data/shaders/shadow_frag.spv");
            vec![v, f]
        };

        let postfx_shader_stages = {
            let v = vkdevice::load_shader_stage(&gpu.device, vk::ShaderStageFlags::VERTEX, "./data/shaders/postfx_vert.spv");
            let f = vkdevice::load_shader_stage(&gpu.device, vk::ShaderStageFlags::FRAGMENT, "./data/shaders/postfx_frag.spv");
            vec![v, f]
        };

        let msaa_samples = msaa_samples_from_limit(gpu.physical_device_properties.limits.framebuffer_color_sample_counts);
        let main_info = GraphicsPipelineBuilder::init(hdr_forward_pass, graphics_pipeline_layout)
                        .set_shader_stages(main_shader_stages).set_msaa_samples(msaa_samples)
                        .set_depth_compare_op(vk::CompareOp::GREATER_OR_EQUAL).build_info();
        let terrain_info = GraphicsPipelineBuilder::init(hdr_forward_pass, graphics_pipeline_layout)
                            .set_shader_stages(terrain_shader_stages).set_msaa_samples(msaa_samples)
                            .set_depth_compare_op(vk::CompareOp::GREATER_OR_EQUAL).build_info();
        let atm_info = GraphicsPipelineBuilder::init(hdr_forward_pass, graphics_pipeline_layout)
                            .set_shader_stages(atm_shader_stages).set_msaa_samples(msaa_samples)
                            .set_depth_compare_op(vk::CompareOp::GREATER_OR_EQUAL).build_info();
        let shadow_info = GraphicsPipelineBuilder::init(shadow_pass, graphics_pipeline_layout)
                            .set_shader_stages(s_shader_stages).set_cull_mode(vk::CullModeFlags::NONE).build_info();
        let postfx_info = GraphicsPipelineBuilder::init(swapchain_pass, graphics_pipeline_layout)
                            .set_shader_stages(postfx_shader_stages).build_info();
                            
    
        let infos = [main_info, terrain_info, atm_info, shadow_info, postfx_info];
        let pipelines = GraphicsPipelineBuilder::create_pipelines(&mut gpu, &infos);

        [
            pipelines[0],
            pipelines[1],
            pipelines[2],
            pipelines[3],
            pipelines[4]
        ]
    };

    //Create compute pipeline
    let bloom_pipeline = unsafe {
        let bloom_pipeline_info = vk::ComputePipelineCreateInfo {
            stage: vkdevice::load_shader_stage(&gpu.device, vk::ShaderStageFlags::COMPUTE, "./data/shaders/bloom.spv"),
            layout: compute_pipeline_layout,
            ..Default::default()
        };
        gpu.device.create_compute_pipelines(vk::PipelineCache::default(), &[bloom_pipeline_info], vkdevice::MEMORY_ALLOCATOR).unwrap()[0]
    };

    let mut simulation_state = Simulation::new();

    //Define terrain
    let mut terrain = TerrainSpec {
        vertex_width: 256,
        vertex_height: 256,
        amplitude: 2.0,
        exponent: 2.2,
        seed: unix_epoch_ms(),
        scale: 20.0,
        ..Default::default()
    };

    let terrain_vertices = terrain.generate_vertices();
    let terrain_indices = ozy::prims::plane_index_buffer(terrain.vertex_width, terrain.vertex_height);

    let mut terrain_collider_handle = physics_engine.make_terrain_collider(&terrain_vertices.positions, terrain.vertex_width, terrain.vertex_height);
    
    //Loading terrain textures in a deferred way
    let terrain_image_paths = [
        "./data/textures/whispy_grass/color.png",
        "./data/textures/whispy_grass/normal.png",
        "./data/textures/whispy_grass/ao_roughness_metallic.png",
        "./data/textures/cliff-rockface/color.png",
        "./data/textures/cliff-rockface/normal.png",
        "./data/textures/cliff-rockface/ao_roughness_metallic.png"
        // "./data/textures/rocky_ground/color.png",
        // "./data/textures/rocky_ground/normal.png",
        // "./data/textures/rocky_ground/ao_roughness_metallic.png"
    ];
    let mut terrain_image_indices = [None; 6];
    for i in 0..terrain_image_paths.len() {
        let path = terrain_image_paths[i];
        let pathp = Path::new(path);

        if !pathp.with_extension("dds").is_file() {
            asset::compress_png_file_synchronous(&mut gpu, path);
        }
        terrain_image_indices[i] = Some(vkdevice::load_bc7_texture(&mut gpu, &mut renderer.global_images, renderer.material_sampler, pathp.with_extension("dds").to_str().unwrap()));
    }
    let [grass_color_index, grass_normal_index, grass_arm_index, rock_color_index, rock_normal_index, rock_arm_index] = terrain_image_indices;

    let terrain_grass_matidx = renderer.global_materials.insert(
        Material {
            pipeline: terrain_pipeline,
            base_color:  [1.0; 4],
            base_roughness: 1.0,
            base_metalness: 0.0,
            emissive_power: [0.0; 3],
            color_idx: grass_color_index,
            normal_idx: grass_normal_index,
            metal_roughness_idx: grass_arm_index,
            emissive_idx: None
        }
    ) as u32;
    let _terrain_rock_matidx = renderer.global_materials.insert(
        Material {
            pipeline: terrain_pipeline,
            base_color:  [1.0; 4],
            base_roughness: 1.0,
            base_metalness: 0.0,
            emissive_power: [0.0; 3],
            color_idx: rock_color_index,
            normal_idx: rock_normal_index,
            metal_roughness_idx: rock_arm_index,
            emissive_idx: None
        }
    ) as u32;
    
    //Upload terrain geometry
    let terrain_key = {
        let terrain_blocks = upload_vertex_attributes(&mut gpu, &mut renderer, &terrain_vertices);
        drop(terrain_vertices);
        let index_buffer = routines::make_index_buffer(&mut gpu, &terrain_indices);
        let prim_key = renderer.register_primitive(Primitive {
            shadow_type: ShadowType::Opaque,
            index_buffer,
            index_count: terrain_indices.len().try_into().unwrap(),
            position_block: terrain_blocks.position_block,
            tangent_block: terrain_blocks.tangent_block,
            normal_block: terrain_blocks.normal_block,
            uv_block: terrain_blocks.uv_block,
            material_idx: terrain_grass_matidx,
        });
        let model_key = renderer.new_model(0, vec![prim_key]);

        let e = Entity::new(String::from("main_terrain"), model_key, &mut physics_engine);
        simulation_state.entities.insert(e)
    };

    //Load totoro as glb
    let totoro_data = asset::gltf_meshdata("./data/models/totoro_backup.glb");
    //let totoro_data = OzyMesh::from_file("./data/models/.optimized/totoro_backup.ozy");

    //Register each primitive with the renderer
    let totoro_model = renderer.upload_gltf_model(&mut gpu, &totoro_data, pbr_pipeline);
    //let totoro_model = renderer.upload_ozymesh(&mut gpu, &totoro_data, vk_3D_graphics_pipeline);

    //Make totoro collider
    let main_totoro_key = {
        let rigid_body = RigidBodyBuilder::dynamic()
        .translation(glm::vec3(0.0, 0.0, 20.0))
        .ccd_enabled(true)
        .build();
        let collider = ColliderBuilder::ball(2.1).restitution(2.5).build();
        let rigid_body_handle = physics_engine.rigid_body_set.insert(rigid_body);
        let collider_handle = physics_engine.collider_set.insert_with_parent(collider, rigid_body_handle, &mut physics_engine.rigid_body_set);

        let p_component = PhysicsComponent {
            rigid_body_handle,
            collider_handle: Some(collider_handle),
            rigid_body_offset: glm::vec3(0.0, 0.0, 2.25),
            scale: 1.0
        };
        let e = Entity::new(String::from("Bouncy Totoro"), totoro_model, &mut physics_engine).set_physics_component(p_component);
        simulation_state.entities.insert(e)
    };

    //Load bistro
    // let bistro_key = {
    //     let data = OzyMesh::from_file("./data/models/optimized/lumberyard_bistro_exterior.ozy");
    //     let model = renderer.upload_ozymesh(&mut gpu, &data, pbr_pipeline);
    //     let mut e = Entity::new(data.name, model, &mut physics_engine);
    //     e.set_position(glm::vec3(0.0, 0.0, 100.0), &mut physics_engine);
    //     e.set_scale(5.0, &mut physics_engine);
    //     simulation_state.entities.insert(e)
    // };

    //State for freecam controls
    let mut camera = Camera::new(glm::vec3(-23.5138, -0.8549, 6.1737));
    camera.focused_entity = Some(main_totoro_key);

    let mut timer = FrameTimer::new();      //Struct for doing basic framerate independence

    renderer.uniform_data.ambient_factor = 20.0;
    renderer.uniform_data.stars_threshold = 4.0;
    renderer.uniform_data.stars_exposure = 2000.0;
    renderer.uniform_data.fog_density = 2.8;
    renderer.uniform_data.exposure = 0.004;
    
    //Load and play bgm
    let bgm = unwrap_result(Music::from_file("./data/music/relaxing_botw.mp3"), "Error loading bgm");
    bgm.play(-1).unwrap();

    let mut dev_gui = DevGui::new(&mut gpu, swapchain_pass, graphics_pipeline_layout);

    let mut input_system = InputSystem::init(&sdl_context);

    let mut totoro_counter = 0;

    //Main application loop
    'running: loop {
        timer.update(); //Update frame timer
        let scaled_delta_time = if timer.delta_time > 1.0 / 30.0 {
            simulation_state.timescale / 30.0
        } else {
            timer.delta_time * simulation_state.timescale
        };

        //Reset renderer
        renderer.reset();

        //Input sampling
        let imgui_io = imgui_context.io_mut();
        let user_input = match input_system.poll(&timer, imgui_io) {
            InputSystemOutput::Output(o) => { o }
            InputSystemOutput::ExitProgram => { break 'running; }
        };

        //Handling of some input results before update
        if user_input.gui_toggle {
            dev_gui.do_gui = !dev_gui.do_gui
        }
        if user_input.regen_terrain {
            if let Some(entity) = simulation_state.entities.get(terrain_key) {
                if let Some(model) = renderer.get_model(entity.model) {
                    let p_key = model.primitive_keys[0];
                    regenerate_terrain(
                        &mut gpu,
                        &mut renderer,
                        &mut physics_engine,
                        &mut terrain_collider_handle,
                        p_key,
                        &mut terrain
                    );
                }
            }
        }
        if user_input.reset_totoro {
            reset_totoro(&mut physics_engine, &simulation_state.entities[main_totoro_key]);
        }

        if user_input.spawn_totoro_prop {
            let shoot_dir = camera.look_direction();
            let init_pos = camera.position + 5.0 * shoot_dir;

            let rigid_body = RigidBodyBuilder::dynamic()
            .translation(init_pos)
            .linvel(shoot_dir * 40.0)
            .ccd_enabled(true)
            .build();
            let collider = ColliderBuilder::ball(2.25).restitution(0.5).build();
            let rigid_body_handle = physics_engine.rigid_body_set.insert(rigid_body);
            let collider_handle = physics_engine.collider_set.insert_with_parent(collider, rigid_body_handle, &mut physics_engine.rigid_body_set);
            let p_component = PhysicsComponent {
                rigid_body_handle,
                collider_handle: Some(collider_handle),
                rigid_body_offset: glm::vec3(0.0, 0.0, 2.25),
                scale: 1.0
            };
            let e = Entity::new(format!("Dynamic Totoro #{}", totoro_counter), totoro_model, &mut physics_engine).set_physics_component(p_component);
            totoro_counter += 1;
            simulation_state.entities.insert(e);
            renderer.increment_model_count(totoro_model);
        }

        //Handle needing to resize the window
        unsafe {
            if user_input.resize_window {
                //Window resizing requires us to "flush the pipeline"
                gpu.device.wait_for_fences(&renderer.in_flight_fences(), true, vk::DeviceSize::MAX).unwrap();

                //Free the now-invalid swapchain data
                for framebuffer in renderer.window_manager.swapchain_framebuffers {
                    gpu.device.destroy_framebuffer(framebuffer, vkdevice::MEMORY_ALLOCATOR);
                }
                for view in renderer.window_manager.swapchain_image_views {
                    gpu.device.destroy_image_view(view, vkdevice::MEMORY_ALLOCATOR);
                }
                gpu.ext_swapchain.destroy_swapchain(renderer.window_manager.swapchain, vkdevice::MEMORY_ALLOCATOR);

                gpu.ext_surface.destroy_surface(renderer.window_manager.surface, vkdevice::MEMORY_ALLOCATOR);
                gpu.device.destroy_semaphore(renderer.window_manager.swapchain_semaphore, vkdevice::MEMORY_ALLOCATOR);

                //Recreate swapchain and associated data
                renderer.window_manager = render::WindowManager::init(&mut gpu, &window, swapchain_pass);

                //Recreate internal rendering buffers
                let extent = vk::Extent3D {
                    width: renderer.window_manager.extent.width,
                    height: renderer.window_manager.extent.height,
                    depth: 1
                };
                renderer.resize_hdr_framebuffers(&mut gpu, extent, hdr_forward_pass);

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

        let imgui_ui = imgui_context.new_frame();   //Transition Dear ImGUI into recording state

        //Terrain generation window
        if dev_gui.do_terrain_window(&imgui_ui, &mut terrain) {
            if let Some(entity) = simulation_state.entities.get(terrain_key) {
                if let Some(model) = renderer.get_model(entity.model) {
                    let p_key = model.primitive_keys[0];
                    regenerate_terrain(
                        &mut gpu,
                        &mut renderer,
                        &mut physics_engine,
                        &mut terrain_collider_handle,
                        p_key,
                        &mut terrain
                    );
                }
            }
        }

        if dev_gui.do_gui {
            if let Some(t) = imgui_ui.window("Main control panel (press ESC to hide/unhide)")
                .menu_bar(true)
                .begin() {
                if let Some(mb) = imgui_ui.begin_menu_bar() {
                    if let Some(mt) = imgui_ui.begin_menu("File") {
                        if imgui_ui.menu_item("New") {}
                        if imgui_ui.menu_item("Load") {}
                        if imgui_ui.menu_item("Save As...") {
                            if let Some(_path) = tfd::save_file_dialog("Save As...", "./data/scenes") {
                                
                            }
                        }
                        if imgui_ui.menu_item("Quit") { break 'running; }
                        mt.end();
                    
                    }
                    if let Some(mt) = imgui_ui.begin_menu("View") {
                        if imgui_ui.menu_item("Asset optimization") { dev_gui.do_asset_window = true; }
                        if imgui_ui.menu_item("Camera") { dev_gui.do_camera_window = true; }
                        if imgui_ui.menu_item("Entities") { dev_gui.do_entity_window = true; }
                        mt.end();
                    }
                    if let Some(mt) = imgui_ui.begin_menu("Cheats") {
                        if imgui_ui.menu_item("Made you look") {}
                        mt.end();
                    }
                    if let Some(mt) = imgui_ui.begin_menu("Environment") {
                        if imgui_ui.menu_item("Terrain generator") { dev_gui.do_terrain_window = true; }
                        if imgui_ui.menu_item("Sun variables") { dev_gui.do_sun_window = true; }
                        mt.end();
                    }
                    mb.end();
                }
    
                imgui_ui.text(format!("Rendering at {:.0} FPS ({:.2} ms frametime, frame {})", user_input.framerate, 1000.0 / user_input.framerate, timer.frame_count));
                
                let (message, color) =  match input_system.controllers[0] {
                    Some(_) => { ("Controller is connected.", [0.0, 1.0, 0.0, 1.0]) }
                    None => { ("Controller is not connected.", [1.0, 0.0, 0.0, 1.0]) }
                };
                let color_token = imgui_ui.push_style_color(imgui::StyleColor::Text, color);
                imgui_ui.text(message);
                color_token.pop();
                
                imgui_ui.slider("Ambient factor", 0.0, 500.0, &mut renderer.uniform_data.ambient_factor);    
                imgui_ui.slider("Stars threshold", 0.0, 16.0, &mut renderer.uniform_data.stars_threshold);
                imgui_ui.slider("Stars exposure", 0.0, 5000.0, &mut renderer.uniform_data.stars_exposure);
                imgui_ui.slider("Fog factor", 0.0, 8.0, &mut renderer.uniform_data.fog_density);
                imgui_ui.slider_config("Camera exposure", 0.0, 0.02).flags(SliderFlags::NO_ROUND_TO_FORMAT).build(&mut renderer.uniform_data.exposure);
                imgui_ui.slider_config("Bloom strength", 0.0, 1.0).flags(SliderFlags::NO_ROUND_TO_FORMAT).build(&mut renderer.uniform_data.bloom_strength);
                imgui_ui.slider("Timescale factor", 0.001, 8.0, &mut simulation_state.timescale);
    
                if imgui_ui.slider("Music volume", 0, 128, &mut music_volume) { Music::set_volume(music_volume); }
    
                imgui_ui.text(format!("Freecam is at ({:.4}, {:.4}, {:.4})", camera.position.x, camera.position.y, camera.position.z));
                
                if DevGui::do_standard_button(&imgui_ui, "Totoro's be gone") {
                    let mut remove_keys = vec![];
                    for e in simulation_state.entities.iter() {
                        let key = e.0;
                        let entity = e.1;
                        if entity.name.to_lowercase().contains("dynamic totoro") {
                            remove_keys.push(key);
                        }
                    }
                    for key in remove_keys {
                        simulation_state.entities.remove(key);
                    }
                    totoro_counter = 0;
                }
                
                #[cfg(target_os = "windows")]
                if DevGui::do_standard_button(&imgui_ui, "Just crash my whole PC why don't ya") {
                    if let tfd::YesNo::Yes = tfd::message_box_yes_no("Dude...", "Are you really sure you want to do this? Make sure all the work you have open on your PC has been saved", tfd::MessageBoxIcon::Warning, tfd::YesNo::No) {
                        bsod::bsod();
                    }
                }

                if DevGui::do_standard_button(&imgui_ui, "Unfocus camera") {
                    camera.focused_entity = None;
                }

                let mut state = renderer.uniform_data.real_sky != 0.0;
                if imgui_ui.checkbox("Realistic sky", &mut state) {
                    renderer.uniform_data.real_sky = if state {
                        1.0
                    } else {
                        0.0
                    };
                }
    
                t.end();
            }
        }
    
        if let Some(sun) = renderer.get_directional_light_mut(sunlight_key) {
            dev_gui.do_sun_window(&imgui_ui, sun);
        }

        dev_gui.do_camera_window(&imgui_ui, &mut camera);

        match dev_gui.do_asset_window(&imgui_ui, "./data/models") {
            AssetWindowResponse::OptimizeGLB(path) => {
                println!("Optimizing {}", path);
                asset::optimize_glb(&mut gpu, &path);
            }
            AssetWindowResponse::None => {}
        }

        match dev_gui.do_entity_window(&imgui_ui, window_size, &mut simulation_state.entities, camera.focused_entity, &mut physics_engine) {
            EntityWindowResponse::LoadGLTF(path) => {
                let mesh_data = asset::gltf_meshdata(&path);
                let gltf_model = renderer.upload_gltf_model(&mut gpu, &mesh_data, pbr_pipeline);
                let spawn_point = camera.position + camera.look_direction() * 5.0;
                let mut s = Entity::new(mesh_data.name, gltf_model, &mut physics_engine);
                s.set_position(spawn_point, &mut physics_engine);
                simulation_state.entities.insert(s);
            }
            EntityWindowResponse::LoadOzyMesh(path) => {
                let mesh_data = OzyMesh::from_file(&path);
                let ozy_model = renderer.upload_ozymesh(&mut gpu, &mesh_data, pbr_pipeline);
                let spawn_point = camera.position + camera.look_direction() * 5.0;
                let mut s = Entity::new(mesh_data.name, ozy_model, &mut physics_engine);
                s.set_position(spawn_point, &mut physics_engine);
                simulation_state.entities.insert(s);
            }
            EntityWindowResponse::CloneEntity(key) => {
                if let Some(entity) = simulation_state.entities.get(key) {
                    renderer.increment_model_count(entity.model);
                    let new_entity = Entity {
                        name: entity.name.clone(),
                        model: entity.model,
                        physics_component: physics_engine.clone_physics_component(&entity.physics_component)
                    };
                    simulation_state.entities.insert(new_entity);
                }
            }
            EntityWindowResponse::DeleteEntity(key) => {
                if let Some(entity) = simulation_state.entities.get(key) {
                    renderer.delete_model(entity.model);
                    simulation_state.entities.remove(key);
                }
            }
            EntityWindowResponse::FocusCamera(k) => { camera.focused_entity = k; }
            EntityWindowResponse::Interacted => {}
            EntityWindowResponse::None => {}
        }
        
        dev_gui.do_material_list(&imgui_ui, &mut renderer);

        //Step the physics engine
        physics_engine.integration_parameters.dt = scaled_delta_time;
        physics_engine.step();

        //Move the main totoro back to the center if it's too far from the center of the world
        if let Some(entity) = simulation_state.entities.get_mut(main_totoro_key) {
            if let Some(body) = physics_engine.rigid_body_set.get(entity.physics_component.rigid_body_handle) {
                if glm::distance(body.translation(), &glm::zero()) > 750.0 {
                    reset_totoro(&mut physics_engine, entity);
                }
            }
        }

        renderer.uniform_data.view_from_world = camera.update(&simulation_state, &physics_engine, &mut renderer, &user_input, timer.delta_time);

        //Push drawcalls for entities
        for (_, entity) in simulation_state.entities.iter() {
            let body = physics_engine.rigid_body_set.get(entity.physics_component.rigid_body_handle).expect("All entities should have a rigid body component.");
            let mm = body.position().to_matrix() * glm::translation(&(-entity.physics_component.rigid_body_offset)) * ozy::routines::uniform_scale(entity.physics_component.scale);
            renderer.drawcall(entity.model, vec![mm]);
        }
        
        //Update sun
        if let Some(sun) = renderer.get_directional_light_mut(sunlight_key) {
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
        dev_gui.resolve_imgui_frame(&mut gpu, &mut renderer, &mut imgui_context);

        //Draw
        unsafe {
            //Begin acquiring swapchain. This is called as early as possible in order to minimize time waiting
            let current_framebuffer_index = gpu.ext_swapchain.acquire_next_image(renderer.window_manager.swapchain, vk::DeviceSize::MAX, renderer.window_manager.swapchain_semaphore, vk::Fence::null()).unwrap().0 as usize;
                    
            //Does all work that needs to happen before the render passes
            let frame_info = renderer.prepare_frame(&mut gpu, window_size, &camera, timer.elapsed_time);

            //Put main command buffer in recording state
            gpu.device.begin_command_buffer(frame_info.main_command_buffer, &vk::CommandBufferBeginInfo::default()).unwrap();

            //Shadow render pass
            if let Some(sun) = renderer.get_directional_light(sunlight_key) {
                if let Some(sun_shadow_map) = &sun.shadow_map {
                    let render_area = {
                        vk::Rect2D {
                            offset: vk::Offset2D { x: 0, y: 0 },
                            extent: vk::Extent2D {
                                width: sun_shadow_map.resolution(),
                                height: sun_shadow_map.resolution() 
                            }
                        }
                    };
                    gpu.device.cmd_set_scissor(frame_info.main_command_buffer, 0, &[render_area]);
                    let clear_values = [vk::ClearValue {
                        depth_stencil: vk::ClearDepthStencilValue {
                            depth: 1.0,
                            stencil: 0
                        }
                    }];
                    let rp_begin_info = vk::RenderPassBeginInfo {
                        render_pass: shadow_pass,
                        framebuffer: sun_shadow_map.framebuffer(),
                        render_area,
                        clear_value_count: clear_values.len() as u32,
                        p_clear_values: clear_values.as_ptr(),
                        ..Default::default()
                    };

                    gpu.device.cmd_begin_render_pass(frame_info.main_command_buffer, &rp_begin_info, vk::SubpassContents::INLINE);
                    gpu.device.cmd_bind_descriptor_sets(
                        frame_info.main_command_buffer,
                        vk::PipelineBindPoint::GRAPHICS,
                        graphics_pipeline_layout,
                        0,
                        &[renderer.bindless_descriptor_set],
                        &[frame_info.dynamic_uniform_offset as u32, frame_info.instance_data_start_offset as u32]
                    );
                    gpu.device.cmd_bind_pipeline(frame_info.main_command_buffer, vk::PipelineBindPoint::GRAPHICS, shadow_pipeline);

                    let viewport = vk::Viewport {
                        x: 0.0,
                        y: 0.0,
                        width: sun_shadow_map.resolution() as f32,
                        height: sun_shadow_map.resolution() as f32,
                        min_depth: 0.0,
                        max_depth: 1.0
                    };
                    gpu.device.cmd_set_viewport(frame_info.main_command_buffer, 0, &[viewport]);
                    for scall in renderer.drawlist_iter() {
                        if let Some(model) = renderer.get_primitive(scall.primitive_key) {
                            if let ShadowType::None = model.shadow_type { continue; }

                            let position_offset: u32 = model.position_block.start_offset as u32 / 4;
                            let uv_offset: u32 = model.uv_block.start_offset as u32 / 2;
                            let pcs = [
                                model.material_idx.to_le_bytes(),
                                position_offset.to_le_bytes(),
                                uv_offset.to_le_bytes()
                            ].concat();
                            gpu.device.cmd_push_constants(frame_info.main_command_buffer, graphics_pipeline_layout, push_constant_stage_flags, 0, &pcs);
                            gpu.device.cmd_bind_index_buffer(frame_info.main_command_buffer, model.index_buffer.buffer(), 0, vk::IndexType::UINT32);
                            gpu.device.cmd_draw_indexed(frame_info.main_command_buffer, model.index_count, scall.instance_count, 0, 0, scall.first_instance);
                        }
                    }

                    gpu.device.cmd_end_render_pass(frame_info.main_command_buffer);
                }
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
            gpu.device.cmd_set_viewport(frame_info.main_command_buffer, 0, &[viewport]);

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
            gpu.device.cmd_set_scissor(frame_info.main_command_buffer, 0, &[scissor_area]);

            let vk_clear_values = [
                vkdevice::COLOR_CLEAR,
                    vk::ClearValue {
                    depth_stencil: vk::ClearDepthStencilValue {
                        depth: 0.0,
                        stencil: 0
                    }
                },
                vkdevice::COLOR_CLEAR
            ];

            //HDR render pass recording
            let rp_begin_info = vk::RenderPassBeginInfo {
                render_pass: hdr_forward_pass,
                framebuffer: frame_info.framebuffer.framebuffer_object,
                render_area: vk_render_area,
                clear_value_count: vk_clear_values.len() as u32,
                p_clear_values: vk_clear_values.as_ptr(),
                ..Default::default()
            };
            gpu.device.cmd_begin_render_pass(frame_info.main_command_buffer, &rp_begin_info, vk::SubpassContents::INLINE);
            gpu.device.cmd_bind_descriptor_sets(
                frame_info.main_command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                graphics_pipeline_layout,
                0,
                &[renderer.bindless_descriptor_set],
                &[frame_info.dynamic_uniform_offset as u32, frame_info.instance_data_start_offset as u32]
            );

            //Iterate through draw calls
            let mut last_bound_pipeline = vk::Pipeline::default();
            for drawcall in renderer.drawlist_iter() {
                if drawcall.pipeline != last_bound_pipeline {
                    gpu.device.cmd_bind_pipeline(frame_info.main_command_buffer, vk::PipelineBindPoint::GRAPHICS, drawcall.pipeline);
                    last_bound_pipeline = drawcall.pipeline;
                }
                if let Some(model) = renderer.get_primitive(drawcall.primitive_key) {
                    let position_offset: u32 = model.position_block.start_offset as u32 / 4;
                    let tangent_offset: u32 = model.tangent_block.start_offset as u32 / 4;
                    let normal_offset: u32 = model.normal_block.start_offset as u32 / 4;
                    let uv_offset: u32 = model.uv_block.start_offset as u32 / 2;
                    let pcs = [
                        model.material_idx.to_le_bytes(),
                        position_offset.to_le_bytes(),
                        tangent_offset.to_le_bytes(),
                        normal_offset.to_le_bytes(),
                        uv_offset.to_le_bytes(),
                    ].concat();
                    gpu.device.cmd_push_constants(frame_info.main_command_buffer, graphics_pipeline_layout, push_constant_stage_flags, 0, &pcs);
                    gpu.device.cmd_bind_index_buffer(frame_info.main_command_buffer, model.index_buffer.buffer(), 0, vk::IndexType::UINT32);
                    gpu.device.cmd_draw_indexed(frame_info.main_command_buffer, model.index_count, drawcall.instance_count, 0, 0, drawcall.first_instance);
                }
            }

            //Record atmosphere rendering commands
            gpu.device.cmd_bind_pipeline(frame_info.main_command_buffer, vk::PipelineBindPoint::GRAPHICS, atmosphere_pipeline);
            gpu.device.cmd_push_constants(frame_info.main_command_buffer, graphics_pipeline_layout, push_constant_stage_flags, 0, &0u32.to_le_bytes());
            gpu.device.cmd_draw(frame_info.main_command_buffer, 36, 1, 0, 0);
            
            gpu.device.cmd_end_render_pass(frame_info.main_command_buffer);

            //Record bloom commands
            {
                const THREADS_X: u32 = 16;
                const THREADS_Y: u32 = 16;
                let hdr_resolve_idx = frame_info.framebuffer.color_resolve_index.try_into().unwrap();
                let bloom_chain_idx = frame_info.bloom_buffer_idx.try_into().unwrap();
                let bloom_chain_mipcount = renderer.global_images.get_element(bloom_chain_idx).unwrap().mip_count;

                gpu.device.cmd_bind_pipeline(frame_info.main_command_buffer, vk::PipelineBindPoint::COMPUTE, bloom_pipeline);
                gpu.device.cmd_bind_descriptor_sets(
                    frame_info.main_command_buffer,
                    vk::PipelineBindPoint::COMPUTE,
                    compute_pipeline_layout,
                    0,
                    &[renderer.bindless_descriptor_set],
                    &[frame_info.dynamic_uniform_offset as u32, frame_info.instance_data_start_offset as u32]
                );

                //Downsampling
                for i in 0..bloom_chain_mipcount-1 {
                    let (input_idx, output_idx, old_layout) = if i == 0 {
                        (hdr_resolve_idx, bloom_chain_idx, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    } else {
                        (bloom_chain_idx, bloom_chain_idx, vk::ImageLayout::GENERAL)
                    };

                    let bloom_barriers = [
                        vk::ImageMemoryBarrier2 {
                            image: renderer.global_images.get_element(input_idx).unwrap().image,
                            old_layout,
                            new_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                            src_access_mask: vk::AccessFlags2::COLOR_ATTACHMENT_WRITE | vk::AccessFlags2::SHADER_WRITE,
                            dst_access_mask: vk::AccessFlags2::SHADER_READ,
                            src_stage_mask: vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT | vk::PipelineStageFlags2::COMPUTE_SHADER,
                            dst_stage_mask: vk::PipelineStageFlags2::COMPUTE_SHADER,
                            src_queue_family_index: gpu.main_queue_family_index,
                            dst_queue_family_index: gpu.main_queue_family_index,
                            subresource_range: vk::ImageSubresourceRange {
                                aspect_mask: vk::ImageAspectFlags::COLOR,
                                base_array_layer: 0,
                                layer_count: 1,
                                base_mip_level: i,
                                level_count: 1
                            },
                            ..Default::default()
                        },
                        vk::ImageMemoryBarrier2 {
                            image: renderer.global_images.get_element(output_idx).unwrap().image,
                            old_layout: vk::ImageLayout::UNDEFINED,
                            new_layout: vk::ImageLayout::GENERAL,
                            src_access_mask: vk::AccessFlags2::COLOR_ATTACHMENT_WRITE | vk::AccessFlags2::SHADER_WRITE,
                            dst_access_mask: vk::AccessFlags2::SHADER_WRITE,
                            src_stage_mask: vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT | vk::PipelineStageFlags2::COMPUTE_SHADER,
                            dst_stage_mask: vk::PipelineStageFlags2::COMPUTE_SHADER,
                            src_queue_family_index: gpu.main_queue_family_index,
                            dst_queue_family_index: gpu.main_queue_family_index,
                            subresource_range: vk::ImageSubresourceRange {
                                aspect_mask: vk::ImageAspectFlags::COLOR,
                                base_array_layer: 0,
                                layer_count: 1,
                                base_mip_level: i + 1,
                                level_count: 1
                            },
                            ..Default::default()
                        }
                    ];
                    let bloom_dependency = vk::DependencyInfo {
                        image_memory_barrier_count: bloom_barriers.len() as u32,
                        p_image_memory_barriers: bloom_barriers.as_ptr(),
                        ..Default::default()
                    };
                    //gpu.device.cmd_pipeline_barrier2(frame_info.main_command_buffer, &bloom_dependency);
                    gpu.ext_sync2.cmd_pipeline_barrier2(frame_info.main_command_buffer, &bloom_dependency);

                    let group_count_x = (window_size.x >> (i + 1)) / THREADS_X + 1;
                    let group_count_y = (window_size.y >> (i + 1)) / THREADS_Y + 1;

                    let out_idx_corrected = output_idx * Renderer::MAX_STORAGE_MIP_COUNT + i as usize + 1;
                    let constants = [
                        input_idx as u32,
                        out_idx_corrected as u32,
                        i,
                        0
                    ];
                    gpu.device.cmd_push_constants(frame_info.main_command_buffer, compute_pipeline_layout, vk::ShaderStageFlags::COMPUTE, 0, slice_to_bytes(&constants));
                    gpu.device.cmd_dispatch(frame_info.main_command_buffer, group_count_x, group_count_y, 1);
                }

                //Upsampling
                for i in 0..bloom_chain_mipcount-1 {
                    let current_mip = bloom_chain_mipcount - 1 - i;
                    let write_old_layout = if current_mip == 1 {
                        vk::ImageLayout::UNDEFINED
                    } else {
                        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
                    };

                    let bloom_barriers = [
                        vk::ImageMemoryBarrier2 {
                            image: renderer.global_images.get_element(bloom_chain_idx).unwrap().image,
                            old_layout: vk::ImageLayout::GENERAL,
                            new_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                            src_access_mask: vk::AccessFlags2::SHADER_WRITE,
                            dst_access_mask: vk::AccessFlags2::SHADER_READ,
                            src_stage_mask: vk::PipelineStageFlags2::COMPUTE_SHADER,
                            dst_stage_mask: vk::PipelineStageFlags2::COMPUTE_SHADER,
                            src_queue_family_index: gpu.main_queue_family_index,
                            dst_queue_family_index: gpu.main_queue_family_index,
                            subresource_range: vk::ImageSubresourceRange {
                                aspect_mask: vk::ImageAspectFlags::COLOR,
                                base_array_layer: 0,
                                layer_count: 1,
                                base_mip_level: current_mip,
                                level_count: 1
                            },
                            ..Default::default()
                        },
                        vk::ImageMemoryBarrier2 {
                            image: renderer.global_images.get_element(bloom_chain_idx).unwrap().image,
                            old_layout: write_old_layout,
                            new_layout: vk::ImageLayout::GENERAL,
                            src_access_mask: vk::AccessFlags2::SHADER_READ,
                            dst_access_mask: vk::AccessFlags2::SHADER_WRITE,
                            src_stage_mask: vk::PipelineStageFlags2::COMPUTE_SHADER,
                            dst_stage_mask: vk::PipelineStageFlags2::COMPUTE_SHADER,
                            src_queue_family_index: gpu.main_queue_family_index,
                            dst_queue_family_index: gpu.main_queue_family_index,
                            subresource_range: vk::ImageSubresourceRange {
                                aspect_mask: vk::ImageAspectFlags::COLOR,
                                base_array_layer: 0,
                                layer_count: 1,
                                base_mip_level: current_mip - 1,
                                level_count: 1
                            },
                            ..Default::default()
                        }
                    ];
                    let bloom_dependency = vk::DependencyInfo {
                        image_memory_barrier_count: bloom_barriers.len() as u32,
                        p_image_memory_barriers: bloom_barriers.as_ptr(),
                        ..Default::default()
                    };
                    //gpu.device.cmd_pipeline_barrier2(frame_info.main_command_buffer, &bloom_dependency);
                    gpu.ext_sync2.cmd_pipeline_barrier2(frame_info.main_command_buffer, &bloom_dependency);
                    
                    let group_count_x = (window_size.x >> (current_mip - 1)) / THREADS_X + 1;
                    let group_count_y = (window_size.y >> (current_mip - 1)) / THREADS_Y + 1;

                    let out_idx_corrected = bloom_chain_idx * Renderer::MAX_STORAGE_MIP_COUNT + current_mip as usize - 1;
                    let constants = [
                        bloom_chain_idx as u32,
                        out_idx_corrected as u32,
                        current_mip,
                        1,
                        hdr_resolve_idx as u32
                    ];
                    gpu.device.cmd_push_constants(frame_info.main_command_buffer, compute_pipeline_layout, vk::ShaderStageFlags::COMPUTE, 0, slice_to_bytes(&constants));
                    gpu.device.cmd_dispatch(frame_info.main_command_buffer, group_count_x, group_count_y, 1);
                }

                //Finally, a barrier on the top mip of the bloom chain
                let final_bloom_barrier = vk::ImageMemoryBarrier2 {
                    image: renderer.global_images.get_element(bloom_chain_idx).unwrap().image,
                    old_layout: vk::ImageLayout::GENERAL,
                    new_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    src_access_mask: vk::AccessFlags2::SHADER_WRITE,
                    dst_access_mask: vk::AccessFlags2::SHADER_READ,
                    src_stage_mask: vk::PipelineStageFlags2::COMPUTE_SHADER,
                    dst_stage_mask: vk::PipelineStageFlags2::FRAGMENT_SHADER,
                    src_queue_family_index: gpu.main_queue_family_index,
                    dst_queue_family_index: gpu.main_queue_family_index,
                    subresource_range: vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_array_layer: 0,
                        layer_count: 1,
                        base_mip_level: 0,
                        level_count: 1
                    },
                    ..Default::default()

                };
                let dep = vk::DependencyInfo {
                    image_memory_barrier_count: 1,
                    p_image_memory_barriers: &final_bloom_barrier,
                    ..Default::default()
                };
                gpu.ext_sync2.cmd_pipeline_barrier2(frame_info.main_command_buffer, &dep);
            }

            gpu.device.end_command_buffer(frame_info.main_command_buffer).unwrap();

            //Submit Shadow+HDR+bloom passes
            let submit_info = vk::SubmitInfo {
                signal_semaphore_count: 1,
                p_signal_semaphores: &frame_info.semaphore,
                command_buffer_count: 1,
                p_command_buffers: &frame_info.main_command_buffer,
                ..Default::default()
            };
            let queue = gpu.device.get_device_queue(gpu.main_queue_family_index, 0);
            gpu.device.queue_submit(queue, &[submit_info], vk::Fence::default()).unwrap();

            //Swapchain output
            gpu.device.begin_command_buffer(frame_info.swapchain_command_buffer, &vk::CommandBufferBeginInfo::default()).unwrap();

            //Bindless descriptor setup for the swapchain command buffer
            gpu.device.cmd_bind_descriptor_sets(
                frame_info.swapchain_command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                graphics_pipeline_layout,
                0,
                &[renderer.bindless_descriptor_set],
                &[frame_info.dynamic_uniform_offset as u32, frame_info.instance_data_start_offset as u32]
            );
            
            //Set the viewport/scissor for the swapchain command buffer
            gpu.device.cmd_set_viewport(frame_info.swapchain_command_buffer, 0, &[viewport]);
            gpu.device.cmd_set_scissor(frame_info.swapchain_command_buffer, 0, &[scissor_area]);

            let rp_begin_info = vk::RenderPassBeginInfo {
                render_pass: swapchain_pass,
                framebuffer: renderer.window_manager.swapchain_framebuffers[current_framebuffer_index],
                render_area: vk_render_area,
                clear_value_count: vk_clear_values.len() as u32,
                p_clear_values: vk_clear_values.as_ptr(),
                ..Default::default()
            };
            gpu.device.cmd_begin_render_pass(frame_info.swapchain_command_buffer, &rp_begin_info, vk::SubpassContents::INLINE);

            gpu.device.cmd_bind_pipeline(frame_info.swapchain_command_buffer, vk::PipelineBindPoint::GRAPHICS, postfx_pipeline);

            let postfx_constants = [
                frame_info.framebuffer.color_resolve_index,
                frame_info.bloom_buffer_idx
            ];
            gpu.device.cmd_push_constants(frame_info.swapchain_command_buffer, graphics_pipeline_layout, push_constant_stage_flags, 0, slice_to_bytes(&postfx_constants));
            gpu.device.cmd_draw(frame_info.swapchain_command_buffer, 3, 1, 0, 0);

            //Record Dear ImGUI drawing commands
            dev_gui.record_draw_commands(&mut gpu, frame_info.swapchain_command_buffer, graphics_pipeline_layout);

            gpu.device.cmd_end_render_pass(frame_info.swapchain_command_buffer);

            gpu.device.end_command_buffer(frame_info.swapchain_command_buffer).unwrap();

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

            let queue = gpu.device.get_device_queue(gpu.main_queue_family_index, 0);
            gpu.device.reset_fences(&[frame_info.fence]).unwrap();
            gpu.device.queue_submit(queue, &[submit_info], frame_info.fence).unwrap();

            let present_semaphores = [frame_info.semaphore, renderer.window_manager.swapchain_semaphore];
            let present_info = vk::PresentInfoKHR {
                swapchain_count: 1,
                p_swapchains: &renderer.window_manager.swapchain,
                p_image_indices: &(current_framebuffer_index as u32),
                wait_semaphore_count: present_semaphores.len() as u32,
                p_wait_semaphores: present_semaphores.as_ptr(),
                ..Default::default()
            };
            if let Err(e) = gpu.ext_swapchain.queue_present(queue, &present_info) {
                println!("{}", e);
            }
        }
    } //After main application loop

    //Cleanup
    unsafe {
        renderer.cleanup(&mut gpu);
    }
}
