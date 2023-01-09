#![allow(non_snake_case)]

//Alias some library names
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
use imgui::{FontAtlasRefMut, SliderFlags};
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
use std::time::SystemTime;

use ozy::structs::{FrameTimer, OptionVec};

use input::InputSystemOutput;
use physics::{PhysicsEngine, PhysicsComponent};
use structs::{Camera, TerrainSpec, SimulationSOA};
use render::vkdevice;
use render::{Primitive, Renderer, Material, CascadedShadowMap, ShadowType, SunLight};

use crate::routines::*;
use crate::asset::GLTFMeshData;
use crate::gui::{DevGui, EntityWindowResponse};
use crate::structs::Entity;

//Entry point
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
        io.key_map[imgui::Key::KeyPadEnter as usize] = Scancode::KpEnter as u32;
        io.key_map[imgui::Key::A as usize] = Scancode::A as u32;
        io.key_map[imgui::Key::C as usize] = Scancode::C as u32;
        io.key_map[imgui::Key::V as usize] = Scancode::V as u32;
        io.key_map[imgui::Key::X as usize] = Scancode::X as u32;
        io.key_map[imgui::Key::Y as usize] = Scancode::Y as u32;
        io.key_map[imgui::Key::Z as usize] = Scancode::Z as u32;
    }

    //Initialize the Vulkan API
    let mut vk = vkdevice::VulkanGraphicsDevice::init();

    //Initialize the physics engine
    let mut physics_engine = PhysicsEngine::init();

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

        //Create dependency between this shadow pass and the HDR pass
        let dependency = vk::SubpassDependency {
            src_subpass: 0,
            dst_subpass: vk::SUBPASS_EXTERNAL,
            src_stage_mask: vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
            dst_stage_mask: vk::PipelineStageFlags::FRAGMENT_SHADER,
            src_access_mask: vk::AccessFlags::MEMORY_WRITE,
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
        vk.device.create_render_pass(&renderpass_info, vkdevice::MEMORY_ALLOCATOR).unwrap()
    };

    let probe_pass = unsafe {
        let depth_description = vk::AttachmentDescription {
            format: vk::Format::R16G16B16A16_SFLOAT,
            samples: vk::SampleCountFlags::TYPE_1,
            load_op: vk::AttachmentLoadOp::CLEAR,
            store_op: vk::AttachmentStoreOp::STORE,
            initial_layout: vk::ImageLayout::UNDEFINED,
            final_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            ..Default::default()
        };

        let color_attachment_reference = vk::AttachmentReference {
            attachment: 0,
            layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
        };

        // let depth_attachment_reference = vk::AttachmentReference {
        //     attachment: 0,
        //     layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL
        // };

        let subpass = vk::SubpassDescription {
            pipeline_bind_point: vk::PipelineBindPoint::GRAPHICS,
            color_attachment_count: 1,
            p_color_attachments: &color_attachment_reference,
            //p_depth_stencil_attachment: &depth_attachment_reference,
            ..Default::default()
        };

        //Multiview info
        let mask = (1 << 6) - 1;
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
        vk.device.create_render_pass(&renderpass_info, vkdevice::MEMORY_ALLOCATOR).unwrap()
    };

    let hdr_forward_pass = unsafe {
        let color_attachment_description = vk::AttachmentDescription {
            format: vk::Format::R16G16B16A16_SFLOAT,
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

        //Create dependences between this pass and previous shadow pass, plus this pass and the PostFX pass
        let dependencies = [
            vk::SubpassDependency {
                src_subpass: vk::SUBPASS_EXTERNAL,
                dst_subpass: 0,
                src_stage_mask: vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
                dst_stage_mask: vk::PipelineStageFlags::FRAGMENT_SHADER,
                src_access_mask: vk::AccessFlags::MEMORY_WRITE,
                dst_access_mask: vk::AccessFlags::SHADER_READ,
                dependency_flags: vk::DependencyFlags::empty()
            },
            vk::SubpassDependency {
                src_subpass: 0,
                dst_subpass: vk::SUBPASS_EXTERNAL,
                src_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                dst_stage_mask: vk::PipelineStageFlags::FRAGMENT_SHADER,
                src_access_mask: vk::AccessFlags::MEMORY_WRITE,
                dst_access_mask: vk::AccessFlags::SHADER_READ,
                dependency_flags: vk::DependencyFlags::empty()

            }
        ];

        let attachments = [color_attachment_description, depth_attachment_description];
        let renderpass_info = vk::RenderPassCreateInfo {
            attachment_count: attachments.len() as u32,
            p_attachments: attachments.as_ptr(),
            subpass_count: 1,
            p_subpasses: &subpass,
            dependency_count: dependencies.len() as u32,
            p_dependencies: dependencies.as_ptr(),
            ..Default::default()
        };
        vk.device.create_render_pass(&renderpass_info, vkdevice::MEMORY_ALLOCATOR).unwrap()
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
        vk.device.create_render_pass(&renderpass_info, vkdevice::MEMORY_ALLOCATOR).unwrap()
    };

    //Initialize the renderer
    let mut renderer = Renderer::init(&mut vk, &window, swapchain_pass, hdr_forward_pass);

    //Create and upload Dear IMGUI font atlas
    match imgui_context.fonts() {
        FontAtlasRefMut::Owned(atlas) => unsafe {
            let atlas_texture = atlas.build_alpha8_texture();
            let atlas_format = vk::Format::R8_UNORM;
            let atlas_layout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
            let gpu_image = vkdevice::upload_raw_image(&mut vk, renderer.point_sampler, atlas_format, atlas_layout, atlas_texture.width, atlas_texture.height, atlas_texture.data);
            let index = renderer.global_images.insert(gpu_image);
            renderer.default_texture_idx = index as u32;
            
            atlas.clear_tex_data();                         //Free atlas memory CPU-side
            atlas.tex_id = imgui::TextureId::new(index);    //Giving Dear Imgui a reference to the font atlas GPU texture
            index as u32
        }
        FontAtlasRefMut::Shared(_) => {
            panic!("Not dealing with this case.");
        }
    };

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
        
        vk.device.create_pipeline_layout(&pipeline_layout_createinfo, vkdevice::MEMORY_ALLOCATOR).unwrap()
    };

    let compute_pipeline_layout = unsafe {
        let push_constant_range = vk::PushConstantRange {
            stage_flags: vk::ShaderStageFlags::COMPUTE,
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
     
        vk.device.create_pipeline_layout(&pipeline_layout_createinfo, vkdevice::MEMORY_ALLOCATOR).unwrap()
    };

    let sun_shadow_map = CascadedShadowMap::new(
        &mut vk,
        &mut renderer,
        2048,
        &glm::perspective_fov_rh_zo(glm::half_pi::<f32>(), window_size.x as f32, window_size.y as f32, 0.1, 1000.0),
        shadow_pass
    );
    renderer.uniform_data.sun_shadowmap_idx = sun_shadow_map.texture_index() as u32;

    let sunlight_key = renderer.new_directional_light(
        SunLight {
            pitch: 0.118,
            yaw: 0.783,
            pitch_speed: 0.003,
            yaw_speed: 0.0,
            irradiance: 790.0f32 * glm::vec3(1.0, 0.891, 0.796),
            shadow_map: sun_shadow_map
        }
    ).unwrap();

    //Create graphics pipelines
    let [pbr_pipeline, terrain_pipeline, atmosphere_pipeline, shadow_pipeline, postfx_pipeline] = unsafe {
        use render::GraphicsPipelineBuilder;

        //Load shaders
        let main_shader_stages = {
            let v = vkdevice::load_shader_stage(&vk.device, vk::ShaderStageFlags::VERTEX, "./data/shaders/vertex_main.spv");
            let f = vkdevice::load_shader_stage(&vk.device, vk::ShaderStageFlags::FRAGMENT, "./data/shaders/pbr_metallic_roughness.spv");
            vec![v, f]
        };
        
        let terrain_shader_stages = {
            let v = vkdevice::load_shader_stage(&vk.device, vk::ShaderStageFlags::VERTEX, "./data/shaders/vertex_main.spv");
            let f = vkdevice::load_shader_stage(&vk.device, vk::ShaderStageFlags::FRAGMENT, "./data/shaders/terrain.spv");
            vec![v, f]
        };
        
        let atm_shader_stages = {
            let v = vkdevice::load_shader_stage(&vk.device, vk::ShaderStageFlags::VERTEX, "./data/shaders/atmosphere_vert.spv");
            let f = vkdevice::load_shader_stage(&vk.device, vk::ShaderStageFlags::FRAGMENT, "./data/shaders/atmosphere_frag.spv");
            vec![v, f]
        };

        let s_shader_stages = {
            let v = vkdevice::load_shader_stage(&vk.device, vk::ShaderStageFlags::VERTEX, "./data/shaders/shadow_vert.spv");
            let f = vkdevice::load_shader_stage(&vk.device, vk::ShaderStageFlags::FRAGMENT, "./data/shaders/shadow_frag.spv");
            vec![v, f]
        };

        let postfx_shader_stages = {
            let v = vkdevice::load_shader_stage(&vk.device, vk::ShaderStageFlags::VERTEX, "./data/shaders/postfx_vert.spv");
            let f = vkdevice::load_shader_stage(&vk.device, vk::ShaderStageFlags::FRAGMENT, "./data/shaders/postfx_frag.spv");
            vec![v, f]
        };

        let main_info = GraphicsPipelineBuilder::init(hdr_forward_pass, graphics_pipeline_layout)
                        .set_shader_stages(main_shader_stages).build_info();
        let terrain_info = GraphicsPipelineBuilder::init(hdr_forward_pass, graphics_pipeline_layout)
                            .set_shader_stages(terrain_shader_stages).build_info();
        let atm_info = GraphicsPipelineBuilder::init(hdr_forward_pass, graphics_pipeline_layout)
                            .set_shader_stages(atm_shader_stages).build_info();
        let shadow_info = GraphicsPipelineBuilder::init(shadow_pass, graphics_pipeline_layout)
                            .set_shader_stages(s_shader_stages).set_cull_mode(vk::CullModeFlags::BACK).build_info();
        let postfx_info = GraphicsPipelineBuilder::init(swapchain_pass, graphics_pipeline_layout)
                            .set_shader_stages(postfx_shader_stages).build_info();
                            
    
        let infos = [main_info, terrain_info, atm_info, shadow_info, postfx_info];
        let pipelines = GraphicsPipelineBuilder::create_pipelines(&mut vk, &infos);

        [
            pipelines[0],
            pipelines[1],
            pipelines[2],
            pipelines[3],
            pipelines[4]
        ]
    };

    //Create compute pipelines
    let lum_binning_pipeline = unsafe {
        let stage = vkdevice::load_shader_stage(&vk.device, vk::ShaderStageFlags::COMPUTE, "./data/shaders/lum_binning.spv");
        let create_info = vk::ComputePipelineCreateInfo {
            stage,
            layout: graphics_pipeline_layout,
            ..Default::default()
        };
        vk.device.create_compute_pipelines(vk::PipelineCache::null(), &[create_info], vkdevice::MEMORY_ALLOCATOR).unwrap()[0]
    };

    let mut simulation_state = SimulationSOA::new();

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
        "./data/textures/rocky_ground/color.png",
        "./data/textures/rocky_ground/normal.png",
        "./data/textures/rocky_ground/ao_roughness_metallic.png"
    ];
    let mut terrain_image_indices = [None; 6];
    for i in 0..terrain_image_paths.len() {
        let path = terrain_image_paths[i];
        let pathp = Path::new(path);

        if !pathp.with_extension("dds").is_file() {
            asset::compress_png_file_synchronous(&mut vk, path);
        }
        terrain_image_indices[i] = Some(vkdevice::load_bc7_texture(&mut vk, &mut renderer.global_images, renderer.material_sampler, pathp.with_extension("dds").to_str().unwrap()));
    }
    let [grass_color_index, grass_normal_index, grass_arm_index, rock_color_index, rock_normal_index, rock_arm_index] = terrain_image_indices;

    let terrain_grass_matidx = renderer.global_materials.insert(
        Material {
            pipeline: terrain_pipeline,
            base_color:  [1.0; 4],
            base_roughness: 1.0,
            base_metalness: 0.0,
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
            color_idx: rock_color_index,
            normal_idx: rock_normal_index,
            metal_roughness_idx: rock_arm_index,
            emissive_idx: None
        }
    ) as u32;
    
    //Upload terrain geometry
    let terrain_key = {
        let terrain_offsets = upload_vertex_attributes(&mut vk, &mut renderer, &terrain_vertices);
        drop(terrain_vertices);
        let index_buffer = routines::make_index_buffer(&mut vk, &terrain_indices);
        let prim_key = renderer.register_primitive(Primitive {
            shadow_type: ShadowType::OpaqueCaster,
            index_buffer,
            index_count: terrain_indices.len().try_into().unwrap(),
            position_offset: terrain_offsets.position_offset,
            tangent_offset: terrain_offsets.tangent_offset,
            normal_offset: terrain_offsets.normal_offset,
            uv_offset: terrain_offsets.uv_offset,
            material_idx: terrain_grass_matidx,
        });
        let model_key = renderer.new_model(0, vec![prim_key]);

        let e = Entity::new(String::from("main_terrain"), model_key, &mut physics_engine);
        simulation_state.entities.insert(e)
    };

    let mut lookat_dist = 7.5;
    let mut lookat_pos = lookat_dist * glm::normalize(&glm::vec3(-1.0f32, 0.0, 1.75));

    //Load totoro as glb
    let totoro_data = asset::gltf_meshdata("./data/models/totoro_backup.glb");
    //let totoro_data = OzyMesh::from_file("./data/models/.optimized/totoro_backup.ozy");

    //Register each primitive with the renderer
    let totoro_model = renderer.upload_gltf_model(&mut vk, &totoro_data, pbr_pipeline);
    //let totoro_model = renderer.upload_ozymesh(&mut vk, &totoro_data, vk_3D_graphics_pipeline);

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

    let mut focused_entity = Some(main_totoro_key);

    //Create semaphore used to wait on swapchain image
    let vk_swapchain_semaphore = unsafe { vk.device.create_semaphore(&vk::SemaphoreCreateInfo::default(), vkdevice::MEMORY_ALLOCATOR).unwrap() };

    //State for freecam controls
    let mut camera = Camera::new(glm::vec3(0.0f32, -30.0, 15.0));
    let mut last_view_from_world = glm::identity();

    let mut timer = FrameTimer::new();      //Struct for doing basic framerate independence

    renderer.uniform_data.ambient_factor = 20.0;
    renderer.uniform_data.stars_threshold = 4.0;
    renderer.uniform_data.stars_exposure = 2000.0;
    renderer.uniform_data.fog_density = 2.8;
    renderer.uniform_data.exposure = 0.004;
    
    //Load and play bgm
    let bgm = unwrap_result(Music::from_file("./data/music/relaxing_botw.mp3"), "Error loading bgm");
    bgm.play(-1).unwrap();

    let mut dev_gui = DevGui::new(&mut vk, swapchain_pass, graphics_pipeline_layout);

    let mut input_system = input::InputSystem::init(&sdl_context);

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
                        &mut vk,
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
            let e = Entity::new(String::from("fired totoro"), totoro_model, &mut physics_engine).set_physics_component(p_component);
            simulation_state.entities.insert(e);
            renderer.increment_model_count(totoro_model);
        }

        //Handle needing to resize the window
        unsafe {
            if user_input.resize_window {
                //Window resizing requires us to "flush the pipeline" as it were. wahh
                vk.device.wait_for_fences(&renderer.in_flight_fences(), true, vk::DeviceSize::MAX).unwrap();

                //Free the now-invalid swapchain data
                for framebuffer in renderer.window_manager.swapchain_framebuffers {
                    vk.device.destroy_framebuffer(framebuffer, vkdevice::MEMORY_ALLOCATOR);
                }
                for view in renderer.window_manager.swapchain_image_views {
                    vk.device.destroy_image_view(view, vkdevice::MEMORY_ALLOCATOR);
                }
                vk.ext_swapchain.destroy_swapchain(renderer.window_manager.swapchain, vkdevice::MEMORY_ALLOCATOR);

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
            if let Some(entity) = simulation_state.entities.get(terrain_key) {
                if let Some(model) = renderer.get_model(entity.model) {
                    let p_key = model.primitive_keys[0];
                    regenerate_terrain(
                        &mut vk,
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
            if let Some(t) = imgui::Window::new("Main control panel (press ESC to hide/unhide)").menu_bar(true).begin(&imgui_ui) {
                if let Some(mb) = imgui_ui.begin_menu_bar() {
                    if let Some(mt) = imgui_ui.begin_menu("File") {
                        if imgui::MenuItem::new("New").build(&imgui_ui) {}
                        if imgui::MenuItem::new("Load").build(&imgui_ui) {}
                        if imgui::MenuItem::new("Save As...").build(&imgui_ui) {
                            if let Some(_path) = tfd::save_file_dialog("Save As...", "./data/scenes") {
                                
                            }
                        }
                        if imgui::MenuItem::new("Quit").build(&imgui_ui) { break 'running; }
                        mt.end();
                    
                    }
                    if let Some(mt) = imgui_ui.begin_menu("View") {
                        if imgui::MenuItem::new("Asset window").build(&imgui_ui) { dev_gui.do_asset_window = true; }
                        if imgui::MenuItem::new("Props window").build(&imgui_ui) { dev_gui.do_props_window = true; }
                        mt.end();
                    }
                    if let Some(mt) = imgui_ui.begin_menu("Cheats") {
                        if imgui::MenuItem::new("Made you look").build(&imgui_ui) {}
                        mt.end();
                    }
                    if let Some(mt) = imgui_ui.begin_menu("Environment") {
                        if imgui::MenuItem::new("Terrain generator").build(&imgui_ui) { dev_gui.do_terrain_window = true; }
                        if imgui::MenuItem::new("Sun variables").build(&imgui_ui) { dev_gui.do_sun_window = true; }
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
    
                if let Some(sun) = renderer.get_directional_light_mut(sunlight_key) {
                    dev_gui.do_sun_window(&imgui_ui, sun);
                }
                
                imgui::Slider::new("Ambient factor", 0.0, 500.0).build(&imgui_ui, &mut renderer.uniform_data.ambient_factor);    
                imgui::Slider::new("Stars threshold", 0.0, 16.0).build(&imgui_ui, &mut renderer.uniform_data.stars_threshold);
                imgui::Slider::new("Stars exposure", 0.0, 5000.0).build(&imgui_ui, &mut renderer.uniform_data.stars_exposure);
                imgui::Slider::new("Fog factor", 0.0, 8.0).build(&imgui_ui, &mut renderer.uniform_data.fog_density);
                imgui::Slider::new("Camera exposure", 0.0, 0.02).flags(SliderFlags::NO_ROUND_TO_FORMAT).build(&imgui_ui, &mut renderer.uniform_data.exposure);
                imgui::Slider::new("Timescale factor", 0.001, 8.0).build(&imgui_ui, &mut simulation_state.timescale);
    
                if imgui::Slider::new("Music volume", 0, 128).build(&imgui_ui, &mut music_volume) { Music::set_volume(music_volume); }
    
                imgui_ui.text(format!("Freecam is at ({:.4}, {:.4}, {:.4})", camera.position.x, camera.position.y, camera.position.z));
                
                if DevGui::do_standard_button(&imgui_ui, "Totoro's be gone") {
                    let mut remove_keys = vec![];
                    for e in simulation_state.entities.iter() {
                        let key = e.0;
                        let entity = e.1;
                        if entity.name.contains("fired totoro") {
                            remove_keys.push(key);
                        }
                    }
                    for key in remove_keys {
                        simulation_state.entities.remove(key);
                    }
                }
                
                #[cfg(target_os = "windows")]
                if DevGui::do_standard_button(&imgui_ui, "Just crash my whole PC why don't ya") {
                    if let tfd::YesNo::Yes = tfd::message_box_yes_no("Dude...", "Are you really sure you want to do this? Make sure all the work you have open on your PC has been saved", tfd::MessageBoxIcon::Warning, tfd::YesNo::No) {
                        bsod::bsod();
                    }
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

        match dev_gui.do_asset_window(&imgui_ui, "./data/models") {
            AssetWindowResponse::OptimizeGLB(path) => {
                println!("Optimizing {}", path);
                asset::optimize_glb_mesh(&mut vk, &path);
            }
            AssetWindowResponse::None => {}
        }

        match dev_gui.do_entity_window(&imgui_ui, &mut simulation_state.entities, focused_entity, &mut physics_engine.rigid_body_set) {
            EntityWindowResponse::LoadGLTF(path) => {
                let mesh_data = asset::gltf_meshdata(&path);
                let model = renderer.upload_gltf_model(&mut vk, &mesh_data, pbr_pipeline);
                let spawn_point = camera.position + camera.look_direction() * 5.0;
                let mut s = Entity::new(mesh_data.name, model, &mut physics_engine);
                s.set_position(spawn_point, &mut physics_engine);
                simulation_state.entities.insert(s);
            }
            EntityWindowResponse::LoadOzyMesh(path) => {
                let mesh_data = OzyMesh::from_file(&path);
                let model = renderer.upload_ozymesh(&mut vk, &mesh_data, pbr_pipeline);
                let spawn_point = camera.position + camera.look_direction() * 5.0;
                let mut s = Entity::new(mesh_data.name, model, &mut physics_engine);
                s.set_position(spawn_point, &mut physics_engine);
                simulation_state.entities.insert(s);
            }
            EntityWindowResponse::DeleteEntity(key) => {
                if let Some(entity) = simulation_state.entities.get(key) {
                    renderer.delete_model(entity.model);
                    simulation_state.entities.remove(key);
                }
            }
            EntityWindowResponse::FocusCamera(k) => { focused_entity = k; }
            EntityWindowResponse::Interacted => {}
            EntityWindowResponse::None => {}
        }
        
        dev_gui.do_material_list(&imgui_ui, &mut renderer);

        //Step the physics engine
        physics_engine.integration_parameters.dt = scaled_delta_time;
        physics_engine.step();

        let view_movement_vector = glm::mat4(
            1.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, -1.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 1.0
        ) * glm::vec3_to_vec4(&user_input.movement_vector);

        //Move the main totoro back to the center if it's too far from the center of the world
        if let Some(entity) = simulation_state.entities.get_mut(main_totoro_key) {
            if let Some(body) = physics_engine.rigid_body_set.get(entity.physics_component.rigid_body_handle) {
                if glm::distance(body.translation(), &glm::zero()) > 750.0 {
                    reset_totoro(&mut physics_engine, entity);
                }
            }
        }

        //Compute this frame's view matrix
        let view_from_world = match focused_entity {
            Some(key) => {
                match simulation_state.entities.get(key) {
                    Some(prop) => {
                        let min = 3.0;
                        let max = 200.0;
                        lookat_dist -= 0.1 * lookat_dist * user_input.scroll_amount;
                        lookat_dist = f32::clamp(lookat_dist, min, max);
                        
                        let lookat = glm::look_at(&lookat_pos, &glm::zero(), &glm::vec3(0.0, 0.0, 1.0));
                        let world_space_offset = glm::affine_inverse(lookat) * glm::vec4(-user_input.orientation_delta.x, user_input.orientation_delta.y, 0.0, 0.0);
            
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

                        let lookat_target = match physics_engine.rigid_body_set.get(prop.physics_component.rigid_body_handle) {
                            Some(body) => {
                                body.translation()
                            }
                            None => {
                                crash_with_error_dialog("All entities should have a rigid body component");
                            }
                        };
            
                        let pos = lookat_pos + lookat_target;
                        let m = glm::look_at(&pos, &lookat_target, &glm::vec3(0.0, 0.0, 1.0));
                        renderer.uniform_data.camera_position = glm::vec4(pos.x, pos.y, pos.z, 1.0);
                        m
                    }
                    None => {
                        //Freecam update                
                        const FREECAM_SPEED: f32 = 3.0;
                        let delta_pos = FREECAM_SPEED * glm::affine_inverse(last_view_from_world) * view_movement_vector * timer.delta_time;
                        camera.position += glm::vec4_to_vec3(&delta_pos);
                        camera.orientation += user_input.orientation_delta;
        
                        camera.orientation.y = camera.orientation.y.clamp(-glm::half_pi::<f32>(), glm::half_pi::<f32>());
                        renderer.uniform_data.camera_position = glm::vec4(camera.position.x, camera.position.y, camera.position.z, 1.0);
                        camera.make_view_matrix()
                    }
                }
            }
            None => {
                //Freecam update                
                const FREECAM_SPEED: f32 = 3.0;
                let delta_pos = FREECAM_SPEED * glm::affine_inverse(last_view_from_world) * view_movement_vector * timer.delta_time;
                camera.position += glm::vec4_to_vec3(&delta_pos);
                camera.orientation += user_input.orientation_delta;

                camera.orientation.y = camera.orientation.y.clamp(-glm::half_pi::<f32>(), glm::half_pi::<f32>());
                renderer.uniform_data.camera_position = glm::vec4(camera.position.x, camera.position.y, camera.position.z, 1.0);
                camera.make_view_matrix()
            }
        };
        last_view_from_world = view_from_world;
        renderer.uniform_data.view_from_world = view_from_world;

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
        dev_gui.resolve_imgui_frame(&mut vk, &mut renderer, imgui_ui);
        
        //Does all work that needs to happen before the render passes
        let frame_info = renderer.prepare_frame(&mut vk, window_size, &view_from_world, timer.elapsed_time);

        //Draw
        unsafe {
            //Begin acquiring swapchain. This is called as early as possible in order to minimize time waiting
            let current_framebuffer_index = vk.ext_swapchain.acquire_next_image(renderer.window_manager.swapchain, vk::DeviceSize::MAX, vk_swapchain_semaphore, vk::Fence::null()).unwrap().0 as usize;

            //Put command buffer in recording state
            vk.device.begin_command_buffer(frame_info.main_command_buffer, &vk::CommandBufferBeginInfo::default()).unwrap();

            //Bindless descriptor setup for Shadow+HDR pass
            let dynamic_uniform_offset = renderer.current_in_flight_frame() as u64 * size_to_alignment!(size_of::<render::EnvironmentUniforms>() as u64, vk.physical_device_properties.limits.min_uniform_buffer_offset_alignment);
            
            //Shadow render pass
            if let Some(sun) = renderer.get_directional_light(sunlight_key) {
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
                let clear_values = [vkdevice::DEPTH_STENCIL_CLEAR];
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
                    graphics_pipeline_layout,
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
                    if let Some(model) = renderer.get_primitive(drawcall.primitive_key) {
                        if let ShadowType::NonCaster = model.shadow_type { continue; }

                        let pcs = [
                            model.material_idx.to_le_bytes(),
                            model.position_offset.to_le_bytes(),
                            model.uv_offset.to_le_bytes()
                        ].concat();
                        vk.device.cmd_push_constants(frame_info.main_command_buffer, graphics_pipeline_layout, push_constant_stage_flags, 0, &pcs);
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

            let vk_clear_values = [vkdevice::COLOR_CLEAR, vkdevice::DEPTH_STENCIL_CLEAR];

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
                graphics_pipeline_layout,
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
                if let Some(model) = renderer.get_primitive(drawcall.primitive_key) {
                    let pcs = [
                        model.material_idx.to_le_bytes(),
                        model.position_offset.to_le_bytes(),
                        model.tangent_offset.to_le_bytes(),
                        model.normal_offset.to_le_bytes(),
                        model.uv_offset.to_le_bytes(),
                    ].concat();
                    vk.device.cmd_push_constants(frame_info.main_command_buffer, graphics_pipeline_layout, push_constant_stage_flags, 0, &pcs);
                    vk.device.cmd_bind_index_buffer(frame_info.main_command_buffer, model.index_buffer.backing_buffer(), 0, vk::IndexType::UINT32);
                    vk.device.cmd_draw_indexed(frame_info.main_command_buffer, model.index_count, drawcall.instance_count, 0, 0, drawcall.first_instance);
                }
            }

            //Record atmosphere rendering commands
            vk.device.cmd_bind_pipeline(frame_info.main_command_buffer, vk::PipelineBindPoint::GRAPHICS, atmosphere_pipeline);
            vk.device.cmd_push_constants(frame_info.main_command_buffer, graphics_pipeline_layout, push_constant_stage_flags, 0, &0u32.to_le_bytes());
            vk.device.cmd_draw(frame_info.main_command_buffer, 36, 1, 0, 0);
            vk.device.cmd_end_render_pass(frame_info.main_command_buffer);


            //Luminance binning compute pass
            vk.device.cmd_bind_pipeline(frame_info.main_command_buffer, vk::PipelineBindPoint::COMPUTE, lum_binning_pipeline);
            vk.device.cmd_bind_descriptor_sets(
                frame_info.main_command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                compute_pipeline_layout,
                0,
                &[renderer.bindless_descriptor_set],
                &[dynamic_uniform_offset as u32, frame_info.instance_data_start_offset as u32]
            );
            vk.device.cmd_push_constants(frame_info.main_command_buffer, compute_pipeline_layout, vk::ShaderStageFlags::COMPUTE, 0, &frame_info.framebuffer.texture_index.to_le_bytes());

            let group_count_x = 1;
            let group_count_y = 1;
            vk.device.cmd_dispatch(frame_info.main_command_buffer, group_count_x, group_count_y, 1);
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
                graphics_pipeline_layout,
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
            vk.device.cmd_push_constants(frame_info.swapchain_command_buffer, graphics_pipeline_layout, push_constant_stage_flags, 0, &frame_info.framebuffer.texture_index.to_le_bytes());
            vk.device.cmd_draw(frame_info.swapchain_command_buffer, 3, 1, 0, 0);

            //Record Dear ImGUI drawing commands
            dev_gui.record_draw_commands(&mut vk, frame_info.swapchain_command_buffer, graphics_pipeline_layout);

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

            let present_semaphores = [frame_info.semaphore, vk_swapchain_semaphore];
            let present_info = vk::PresentInfoKHR {
                swapchain_count: 1,
                p_swapchains: &renderer.window_manager.swapchain,
                p_image_indices: &(current_framebuffer_index as u32),
                wait_semaphore_count: present_semaphores.len() as u32,
                p_wait_semaphores: present_semaphores.as_ptr(),
                ..Default::default()
            };
            if let Err(e) = vk.ext_swapchain.queue_present(queue, &present_info) {
                println!("{}", e);
            }
        }
    } //After main application loop

    //Cleanup
    unsafe {
        renderer.cleanup(&mut vk);
    }
}
