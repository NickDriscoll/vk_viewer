#![allow(non_snake_case)]

//Alias some library names
extern crate nalgebra_glm as glm;
extern crate tinyfiledialogs as tfd;
extern crate ozy_engine as ozy;

mod gltfutil;
mod gui;
mod input;
mod physics;
mod render;
mod routines;
mod structs;

#[macro_use]
mod vkutil;

use ash::vk;
use gltfutil::GLTFPrimitive;
use gpu_allocator::MemoryLocation;
use imgui::{FontAtlasRefMut, TextureId};
use rapier3d::prelude::*;
use routines::struct_to_bytes;
use sdl2::event::Event;
use sdl2::mixer;
use sdl2::mixer::Music;
use slotmap::DenseSlotMap;
use std::collections::HashMap;
use std::fmt::Display;
use std::fs::{File};
use std::ffi::CStr;
use std::mem::size_of;
use std::ptr;
use std::time::SystemTime;

use ozy::structs::{FrameTimer, OptionVec};

use input::UserInput;
use vkutil::{ColorSpace, FreeList, GPUBuffer, VirtualImage, VulkanAPI};
use physics::PhysicsEngine;
use structs::{Camera, TerrainSpec, PhysicsProp};
use render::{Primitive, Renderer, Material, CascadedShadowMap, ShadowType};

use crate::routines::*;
use crate::gltfutil::GLTFData;
use crate::gui::DevGui;
use crate::structs::StaticProp;

struct UninterleavedVertices {
    pub positions: Vec<f32>,
    pub tangents: Vec<f32>,
    pub normals: Vec<f32>,
    pub uvs: Vec<f32>,
}

struct VertexFetchOffsets {
    pub position_offset: u32,
    pub tangent_offset: u32,
    pub normal_offset: u32,
    pub uv_offset: u32,
}

fn uninterleave_vertex_buffer(vertex_buffer: &[f32]) -> UninterleavedVertices {
    let floats_per_vertex = 15;
    let mut positions = vec![0.0; vertex_buffer.len() / floats_per_vertex * 4];
    let mut tangents = vec![0.0; vertex_buffer.len() / floats_per_vertex * 4];
    let mut normals = vec![0.0; vertex_buffer.len() / floats_per_vertex * 4];
    let mut uvs = vec![0.0; vertex_buffer.len() / floats_per_vertex * 2];

    for i in 0..(vertex_buffer.len() / floats_per_vertex) {
        positions[4 * i] = vertex_buffer[floats_per_vertex * i];
        positions[4 * i + 1] = vertex_buffer[floats_per_vertex * i + 1];
        positions[4 * i + 2] = vertex_buffer[floats_per_vertex * i + 2];
        positions[4 * i + 3] = 1.0;

        tangents[4 * i] = vertex_buffer[floats_per_vertex * i + 3];
        tangents[4 * i + 1] = vertex_buffer[floats_per_vertex * i + 4];
        tangents[4 * i + 2] = vertex_buffer[floats_per_vertex * i + 5];
        tangents[4 * i + 3] = vertex_buffer[floats_per_vertex * i + 6];

        normals[4 * i] = vertex_buffer[floats_per_vertex * i + 10];
        normals[4 * i + 1] = vertex_buffer[floats_per_vertex * i + 11];
        normals[4 * i + 2] = vertex_buffer[floats_per_vertex * i + 12];
        normals[4 * i + 3] = 0.0;

        uvs[2 * i] = vertex_buffer[floats_per_vertex * i + 13];
        uvs[2 * i + 1] = vertex_buffer[floats_per_vertex * i + 14];
    }

    UninterleavedVertices {
        positions,
        tangents,
        normals,
        uvs
    }
}

fn uninterleave_and_upload_vertices(vk: &mut VulkanAPI, renderer: &mut Renderer, vertex_buffer: &[f32]) -> VertexFetchOffsets {
    let attributes = uninterleave_vertex_buffer(vertex_buffer);
    
    let position_offset = renderer.append_vertex_positions(vk, &attributes.positions);
    let tangent_offset = renderer.append_vertex_tangents(vk, &attributes.tangents);
    let normal_offset = renderer.append_vertex_normals(vk, &attributes.normals);
    let uv_offset = renderer.append_vertex_uvs(vk, &attributes.uvs);

    VertexFetchOffsets {
        position_offset,
        tangent_offset,
        normal_offset,
        uv_offset
    }
}

fn upload_primitive_vertices(vk: &mut VulkanAPI, renderer: &mut Renderer, prim: &GLTFPrimitive) -> VertexFetchOffsets {
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

fn replace_uploaded_uninterleaved_vertices(vk: &mut VulkanAPI, renderer: &mut Renderer, vertex_buffer: &[f32], offset: u64) {
    let attributes = uninterleave_vertex_buffer(vertex_buffer);
    
    renderer.replace_vertex_positions(vk, &attributes.positions, offset);
    renderer.replace_vertex_tangents(vk, &attributes.tangents, offset);
    renderer.replace_vertex_normals(vk, &attributes.normals, offset);
    renderer.replace_vertex_uvs(vk, &attributes.uvs, offset);
}

fn upload_gltf_primitives(vk: &mut VulkanAPI, renderer: &mut Renderer, data: &GLTFData, pipeline: vk::Pipeline) -> Vec<usize> {
    let mut indices = vec![];
    let mut tex_id_map = HashMap::new();
    for prim in &data.primitives {
        let color_idx = if let Some(idx) = prim.material.color_index {
            match tex_id_map.get(&idx) {
                Some(id) => { *id }
                None => {
                    let image = VirtualImage::from_png_bytes(vk, data.texture_bytes[idx].as_slice());
                    let image_info = vk::DescriptorImageInfo {
                        sampler: renderer.material_sampler,
                        image_view: image.vk_view,
                        image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
                    };
                    let global_tex_id = renderer.global_textures.insert(image_info) as u32;
                    tex_id_map.insert(idx, global_tex_id);
                    global_tex_id
                }
            }
        } else {
            renderer.default_diffuse_idx
        };

        let normal_idx = match prim.material.normal_index {
            Some(idx) => {
                match tex_id_map.get(&idx) {
                    Some(id) => { *id }
                    None => {
                        let image = VirtualImage::from_png_bytes(vk, data.texture_bytes[idx].as_slice());
                        let image_info = vk::DescriptorImageInfo {
                            sampler: renderer.material_sampler,
                            image_view: image.vk_view,
                            image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
                        };
                        let global_tex_id = renderer.global_textures.insert(image_info) as u32;
                        tex_id_map.insert(idx, global_tex_id);
                        global_tex_id
                    }
                }
            }
            None => { renderer.default_normal_idx }
        };

        let metal_roughness_idx = match prim.material.metallic_roughness_index {
            Some(idx) => {
                match tex_id_map.get(&idx) {
                    Some(id) => { *id }
                    None => {
                        let image = VirtualImage::from_png_bytes(vk, data.texture_bytes[idx].as_slice());
                        let image_info = vk::DescriptorImageInfo {
                            sampler: renderer.material_sampler,
                            image_view: image.vk_view,
                            image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
                        };
                        let global_tex_id = renderer.global_textures.insert(image_info) as u32;
                        tex_id_map.insert(idx, global_tex_id);
                        global_tex_id
                    }
                }

            }
            None => {
                renderer.default_metal_roughness_idx
            }
        };

        let emissive_idx = match prim.material.emissive_index {
            Some(idx) => {
                match tex_id_map.get(&idx) {
                    Some(id) => { *id }
                    None => {
                        let image = VirtualImage::from_png_bytes(vk, data.texture_bytes[idx].as_slice());
                        let image_info = vk::DescriptorImageInfo {
                            sampler: renderer.material_sampler,
                            image_view: image.vk_view,
                            image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
                        };
                        let global_tex_id = renderer.global_textures.insert(image_info) as u32;
                        tex_id_map.insert(idx, global_tex_id);
                        global_tex_id
                    }
                }

            }
            None => {
                renderer.default_emissive_idx
            }
        };

        let material = Material {
            pipeline,
            base_color: prim.material.base_color,
            base_roughness: prim.material.base_roughness,
            color_idx,
            normal_idx,
            metal_roughness_idx,
            emissive_idx
        };
        let material_idx = renderer.global_materials.insert(material) as u32;

        let offsets = upload_primitive_vertices(vk, renderer, &prim);

        let index_buffer = vkutil::make_index_buffer(vk, &prim.indices);
        let model_idx = renderer.register_model(Primitive {
            shadow_type: ShadowType::OpaqueCaster,
            index_buffer,
            index_count: prim.indices.len().try_into().unwrap(),
            position_offset: offsets.position_offset,
            tangent_offset: offsets.tangent_offset,
            normal_offset: offsets.normal_offset,
            uv_offset: offsets.uv_offset,
            material_idx
        });
        indices.push(model_idx);
    }
    indices
}

fn reset_totoro(physics_engine: &mut PhysicsEngine, totoro: &Option<PhysicsProp>) {
    let handle = totoro.as_ref().unwrap().rigid_body_handle;
    if let Some(body) = physics_engine.rigid_body_set.get_mut(handle) {
        body.set_linvel(glm::zero(), true);
        body.set_position(Isometry::from_parts(Translation::new(0.0, 0.0, 20.0), *body.rotation()), true);
    }
}

//Entry point
fn main() {
    //Create the window using SDL
    let sdl_context = unwrap_result(sdl2::init(), "Error initializing SDL");
    let video_subsystem = unwrap_result(sdl_context.video(), "Error initializing SDL video subsystem");
    let mut window_size = glm::vec2(1280, 1024);
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
    let mut vk = vkutil::VulkanAPI::init(&window);
    
    //Initialize the renderer
    let mut renderer = Renderer::init(&mut vk);

    //Initialize the physics engine
    let mut physics_engine = PhysicsEngine::new();

    let default_image_info;

    //Create and upload Dear IMGUI font atlas
    match imgui_context.fonts() {
        FontAtlasRefMut::Owned(atlas) => unsafe {
            let atlas_texture = atlas.build_alpha8_texture();
            let atlas_format = vk::Format::R8_UNORM;
            let descriptor_info = vkutil::upload_raw_image(&mut vk, renderer.point_sampler, atlas_format, atlas_texture.width, atlas_texture.height, atlas_texture.data);
            default_image_info = descriptor_info;
            let index = renderer.global_textures.insert(descriptor_info);
            
            atlas.clear_tex_data();  //Free atlas memory CPU-side
            atlas.tex_id = imgui::TextureId::new(index);    //Giving Dear Imgui a reference to the font atlas GPU texture
            index as u32
        }
        FontAtlasRefMut::Shared(_) => {
            panic!("Not dealing with this case.");
        }
    };

    //Search for an SRGB swapchain format    
    let surf_formats = unsafe { vk.ext_surface.get_physical_device_surface_formats(vk.physical_device, vk.surface).unwrap() };
    let mut vk_surface_format = vk::SurfaceFormatKHR::default();
    for sformat in surf_formats.iter() {
        if sformat.format == vk::Format::B8G8R8A8_SRGB {
            vk_surface_format = *sformat;
            break;
        }
    }
    drop(surf_formats);

    let shadow_pass = unsafe {
        let depth_description = vk::AttachmentDescription {
            format: vk::Format::D32_SFLOAT,
            samples: vk::SampleCountFlags::TYPE_1,
            load_op: vk::AttachmentLoadOp::CLEAR,
            store_op: vk::AttachmentStoreOp::STORE,
            initial_layout: vk::ImageLayout::UNDEFINED,
            final_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
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

        let attachments = [depth_description];
        let renderpass_info = vk::RenderPassCreateInfo {
            attachment_count: attachments.len() as u32,
            p_attachments: attachments.as_ptr(),
            subpass_count: 1,
            p_subpasses: &subpass,
            ..Default::default()
        };
        vk.device.create_render_pass(&renderpass_info, vkutil::MEMORY_ALLOCATOR).unwrap()
    };

    let sun_shadow_map = CascadedShadowMap::new(
        &mut vk,
        &mut renderer,
        shadow_pass,
        2048,        
        &glm::perspective_fov_rh_zo(glm::half_pi::<f32>(), window_size.x as f32, window_size.y as f32, 0.1, 1000.0)
    );
    renderer.uniform_data.sun_shadowmap_idx = sun_shadow_map.texture_index() as u32;

    let main_forward_pass = unsafe {
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

    //Create the main swapchain for window present
    let mut vk_swapchain = vkutil::Swapchain::init(&mut vk, main_forward_pass);

    let push_constant_shader_stage_flags = vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT;
    let pipeline_layout = unsafe {
        let push_constant_range = vk::PushConstantRange {
            stage_flags: push_constant_shader_stage_flags,
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

    //Create pipelines
    let [vk_3D_graphics_pipeline, terrain_pipeline, atmosphere_pipeline, shadow_pipeline] = unsafe {
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

        let main_info = vkutil::GraphicsPipelineBuilder::init(main_forward_pass, pipeline_layout)
                        .set_shader_stages(main_shader_stages).build_info();
        let terrain_info = vkutil::GraphicsPipelineBuilder::init(main_forward_pass, pipeline_layout)
                            .set_shader_stages(terrain_shader_stages).build_info();
        let atm_info = vkutil::GraphicsPipelineBuilder::init(main_forward_pass, pipeline_layout)
                            .set_shader_stages(atm_shader_stages).build_info();
        let shadow_info = vkutil::GraphicsPipelineBuilder::init(shadow_pass, pipeline_layout)
                            .set_shader_stages(s_shader_stages).set_cull_mode(vk::CullModeFlags::NONE).build_info();
    
        let infos = [main_info, terrain_info, atm_info, shadow_info];
        let pipelines = vkutil::GraphicsPipelineBuilder::create_pipelines(&mut vk, &infos);

        [
            pipelines[0],
            pipelines[1],
            pipelines[2],
            pipelines[3]
        ]
    };

    let mut sun_pitch_speed = 0.003;
    let mut sun_yaw_speed = 0.0;
    let mut sun_pitch = 0.118;
    let mut sun_yaw = 0.783;
    let mut trees_width = 1;
    let mut trees_height = 1;
    let mut timescale_factor = 1.0;
    let terrain_generation_scale = 20.0;

    //Define terrain
    let terrain_vertex_width = 256;
    let mut terrain_fixed_seed = false;
    let mut terrain_interactive_generation = false;
    let mut terrain = TerrainSpec {
        vertex_width: terrain_vertex_width,
        vertex_height: terrain_vertex_width,
        amplitude: 2.0,
        exponent: 2.2,
        seed: unix_epoch_ms(),
        ..Default::default()
    };

    let terrain_vertices = terrain.generate_vertices(terrain_generation_scale);
    let terrain_indices = ozy::prims::plane_index_buffer(terrain_vertex_width, terrain_vertex_width);

    let mut terrain_collider_handle = physics_engine.make_terrain_collider(&terrain_vertices, terrain_vertex_width);
    
    //Loading terrain textures
    let grass_color_global_index = vkutil::load_global_png(&mut vk, &mut renderer.global_textures, renderer.material_sampler, "./data/textures/whispy_grass/color.png", ColorSpace::SRGB);
    let grass_normal_global_index = vkutil::load_global_png(&mut vk, &mut renderer.global_textures, renderer.material_sampler, "./data/textures/whispy_grass/normal.png", ColorSpace::LINEAR);
    let grass_aoroughmetal_global_index = vkutil::load_global_png(&mut vk, &mut renderer.global_textures, renderer.material_sampler, "./data/textures/whispy_grass/ao_roughness_metallic.png", ColorSpace::LINEAR);

    //let grass_color_global_index = vkutil::load_global_bc7(&mut vk, &mut renderer.global_textures, renderer.material_sampler, "./data/textures/whispy_grass/color.dds", ColorSpace::SRGB);
    //let grass_normal_global_index = vkutil::load_global_bc7(&mut vk, &mut renderer.global_textures, renderer.material_sampler, "./data/textures/whispy_grass/normal.dds", ColorSpace::LINEAR);
    //let grass_metalrough_global_index = vkutil::load_global_bc7(&mut vk, &mut renderer.global_textures, renderer.material_sampler, "./data/textures/whispy_grass/metallic_roughness.dds", ColorSpace::LINEAR);

    let rock_color_global_index = vkutil::load_global_png(&mut vk, &mut renderer.global_textures, renderer.material_sampler, "./data/textures/rocky_ground/color.png", ColorSpace::SRGB);
    let rock_normal_global_index = vkutil::load_global_png(&mut vk, &mut renderer.global_textures, renderer.material_sampler, "./data/textures/rocky_ground/normal.png", ColorSpace::LINEAR);
    let rock_aoroughmetal_global_index = vkutil::load_global_png(&mut vk, &mut renderer.global_textures, renderer.material_sampler, "./data/textures/rocky_ground/ao_roughness_metallic.png", ColorSpace::LINEAR);
    
    //let rock_color_global_index = vkutil::load_global_bc7(&mut vk, &mut renderer.global_textures, renderer.material_sampler, "./data/textures/rocky_ground/color.dds", ColorSpace::SRGB);
    //let rock_normal_global_index = vkutil::load_global_bc7(&mut vk, &mut renderer.global_textures, renderer.material_sampler, "./data/textures/rocky_ground/normal.dds", ColorSpace::LINEAR);
    //let rock_metalrough_global_index = vkutil::load_global_bc7(&mut vk, &mut renderer.global_textures, renderer.material_sampler, "./data/textures/rocky_ground/metallic_roughness.dds", ColorSpace::LINEAR);
    
    let terrain_grass_matidx = renderer.global_materials.insert(
        Material {
            pipeline: terrain_pipeline,
            base_color:  [1.0; 4],
            base_roughness: 1.0,
            color_idx: grass_color_global_index,
            normal_idx: grass_normal_global_index,
            metal_roughness_idx: grass_aoroughmetal_global_index,
            emissive_idx: renderer.default_emissive_idx
        }
    ) as u32;
    let terrain_rock_matidx = renderer.global_materials.insert(
        Material {
            pipeline: terrain_pipeline,
            base_color:  [1.0; 4],
            base_roughness: 1.0,
            color_idx: rock_color_global_index,
            normal_idx: rock_normal_global_index,
            metal_roughness_idx: rock_aoroughmetal_global_index,
            emissive_idx: renderer.default_emissive_idx
        }
    ) as u32;
    
    //Upload terrain geometry
    let terrain_model_idx = {
        let terrain_offsets = uninterleave_and_upload_vertices(&mut vk, &mut renderer, &terrain_vertices);
        drop(terrain_vertices);
        let index_buffer = vkutil::make_index_buffer(&mut vk, &terrain_indices);
        renderer.register_model(Primitive {
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

    let mut totoro_lookat_dist = 7.5;
    let mut totoro_lookat_pos = totoro_lookat_dist * glm::normalize(&glm::vec3(-1.0f32, 0.0, 1.75));

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
        let prop = PhysicsProp {
            rigid_body_handle,
            collider_handle
        };
        totoro_list.insert(prop)
    };

    let mut static_props = DenseSlotMap::<_, StaticProp>::new();

    //Create semaphore used to wait on swapchain image
    let vk_swapchain_semaphore = unsafe { vk.device.create_semaphore(&vk::SemaphoreCreateInfo::default(), vkutil::MEMORY_ALLOCATOR).unwrap() };
    let vk_rendercomplete_semaphore = unsafe { vk.device.create_semaphore(&vk::SemaphoreCreateInfo::default(), vkutil::MEMORY_ALLOCATOR).unwrap() };

    //State for freecam controls
    let mut camera = Camera::new(glm::vec3(0.0f32, -30.0, 15.0));
    let mut last_view_from_world = glm::identity();
    let mut do_freecam = false;

    let mut timer = FrameTimer::new();      //Struct for doing basic framerate independence

    renderer.uniform_data.sun_luminance = [2.5, 2.5, 2.5, 0.0];
    renderer.uniform_data.stars_threshold = 8.0;
    renderer.uniform_data.stars_exposure = 200.0;
    renderer.uniform_data.fog_density = 0.75;
    
    //Load and play bgm
    let bgm = unwrap_result(Music::from_file("./data/music/relaxing_botw.mp3"), "Error loading bgm");
    bgm.play(-1).unwrap();

    let mut dev_gui = DevGui::new(&mut vk, main_forward_pass, pipeline_layout);

    let mut input_system = input::InputSystem::init(&sdl_context);

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
        if input_output.gui_toggle { dev_gui.do_gui = !dev_gui.do_gui }
        if input_output.regen_terrain {
            regenerate_terrain(
                &mut vk,
                &mut renderer,
                &mut physics_engine,
                &mut terrain_collider_handle,
                terrain_model_idx,
                &mut terrain,
                terrain_vertex_width,
                terrain_fixed_seed,
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
                vk.device.wait_for_fences(&[vk.graphics_command_buffer_fence], true, vk::DeviceSize::MAX).unwrap();

                //Free the now-invalid swapchain data
                for framebuffer in vk_swapchain.swapchain_framebuffers {
                    vk.device.destroy_framebuffer(framebuffer, vkutil::MEMORY_ALLOCATOR);
                }
                for view in vk_swapchain.swapchain_image_views {
                    vk.device.destroy_image_view(view, vkutil::MEMORY_ALLOCATOR);
                }
                vk.device.destroy_image_view(vk_swapchain.depth_image_view, vkutil::MEMORY_ALLOCATOR);
                vk.ext_swapchain.destroy_swapchain(vk_swapchain.swapchain, vkutil::MEMORY_ALLOCATOR);

                //Recreate swapchain and associated data
                vk_swapchain = vkutil::Swapchain::init(&mut vk, main_forward_pass);

                window_size = glm::vec2(vk_swapchain.extent.width, vk_swapchain.extent.height);
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

        //Update
        let imgui_ui = imgui_context.frame();   //Transition Dear ImGUI into recording state
        if dev_gui.do_gui && dev_gui.do_terrain_window {
            if let Some(token) = imgui::Window::new("Terrain generator").begin(&imgui_ui) { 
                let mut parameters_changed = false;

                imgui_ui.text("Global terrain variables:");
                parameters_changed |= imgui::Slider::new("Amplitude", 0.0, 8.0).build(&imgui_ui, &mut terrain.amplitude);
                parameters_changed |= imgui::Slider::new("Exponent", 1.0, 5.0).build(&imgui_ui, &mut terrain.exponent);
                parameters_changed |= imgui::Slider::new("Octaves", 1, 16).build(&imgui_ui, &mut terrain.octaves);
                parameters_changed |= imgui::Slider::new("Lacunarity", 0.0, 5.0).build(&imgui_ui, &mut terrain.lacunarity);
                parameters_changed |= imgui::Slider::new("Gain", 0.0, 2.0).build(&imgui_ui, &mut terrain.gain);
                imgui_ui.separator();

                imgui_ui.text(format!("Last seed used: 0x{:X}", terrain.seed));
                imgui_ui.checkbox("Use fixed seed", &mut terrain_fixed_seed);
                imgui_ui.checkbox("Interactive mode", &mut terrain_interactive_generation);
                if imgui_ui.button_with_size("Regenerate", [0.0, 32.0]) {
                    regenerate_terrain(
                        &mut vk,
                        &mut renderer,
                        &mut physics_engine,
                        &mut terrain_collider_handle,
                        terrain_model_idx,
                        &mut terrain,
                        terrain_vertex_width,
                        terrain_fixed_seed,
                        terrain_generation_scale
                    );
                }

                if terrain_interactive_generation && parameters_changed {
                    regenerate_terrain(
                        &mut vk,
                        &mut renderer,
                        &mut physics_engine,
                        &mut terrain_collider_handle,
                        terrain_model_idx,
                        &mut terrain,
                        terrain_vertex_width,
                        terrain_fixed_seed,
                        terrain_generation_scale
                    );
                }

                if imgui_ui.button_with_size("Close", [0.0, 32.0]) { dev_gui.do_terrain_window = false; }

                token.end();
            }
        }

        if dev_gui.do_gui && dev_gui.do_sun_shadowmap {
            let win = imgui::Window::new("Shadow atlas");
            if let Some(win_token) = win.begin(&imgui_ui) {
                imgui::Image::new(
                    TextureId::new(sun_shadow_map.texture_index()),
                    [(sun_shadow_map.resolution() * CascadedShadowMap::CASCADE_COUNT as u32) as f32 / 6.0, sun_shadow_map.resolution() as f32 / 6.0]
                ).build(&imgui_ui);

                win_token.end();
            }
        }

        let imgui_window_token = if dev_gui.do_gui {
            imgui::Window::new("Main control panel (press ESC to hide)").menu_bar(true).begin(&imgui_ui)
        } else {
            None
        };

        if let Some(_) = imgui_window_token {
            if let Some(mb) = imgui_ui.begin_menu_bar() {
                if let Some(mt) = imgui_ui.begin_menu("Environment") {
                    if imgui::MenuItem::new("Terrain generator").build(&imgui_ui) {
                        dev_gui.do_terrain_window = true;
                    }
                    mt.end();
                }
                mb.end();
            }

            imgui_ui.text(format!("Rendering at {:.0} FPS ({:.2} ms frametime, frame {})", input_output.framerate, 1000.0 / input_output.framerate, timer.frame_count));
            
            let (message, color) =  if input_system.controllers[0].is_some() {
                ("Controller is connected.", [0.0, 1.0, 0.0, 1.0])
            } else {
                ("Controller is not connected.", [1.0, 0.0, 0.0, 1.0])
            };
            let color_token = imgui_ui.push_style_color(imgui::StyleColor::Text, color);
            imgui_ui.text(message);
            color_token.pop();

            imgui::Slider::new("Sun pitch speed", 0.0, 1.0).build(&imgui_ui, &mut sun_pitch_speed);
            imgui::Slider::new("Sun pitch", 0.0, glm::two_pi::<f32>()).build(&imgui_ui, &mut sun_pitch);
            imgui::Slider::new("Sun yaw speed", 0.0, 1.0).build(&imgui_ui, &mut sun_yaw_speed);
            imgui::Slider::new("Sun yaw", 0.0, glm::two_pi::<f32>()).build(&imgui_ui, &mut sun_yaw);
            imgui::Slider::new("Stars threshold", 0.0, 16.0).build(&imgui_ui, &mut renderer.uniform_data.stars_threshold);
            imgui::Slider::new("Stars exposure", 0.0, 1000.0).build(&imgui_ui, &mut renderer.uniform_data.stars_exposure);
            imgui::Slider::new("Fog factor", 0.0, 8.0).build(&imgui_ui, &mut renderer.uniform_data.fog_density);
            imgui::Slider::new("Timescale factor", 0.001, 8.0).build(&imgui_ui, &mut timescale_factor);
            imgui::Slider::new("Trees width", 1, 10).build(&imgui_ui, &mut trees_width);
            imgui::Slider::new("Trees height", 1, 10).build(&imgui_ui, &mut trees_height);
        }

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

        let view_from_world = if do_freecam {
            //Camera orientation based on user input
            camera.orientation.y = camera.orientation.y.clamp(-glm::half_pi::<f32>(), glm::half_pi::<f32>());
            renderer.uniform_data.camera_position = glm::vec4(camera.position.x, camera.position.y, camera.position.z, 1.0);
            camera.make_view_matrix()
        } else {
            let min = 3.0;
            let max = 200.0;
            totoro_lookat_dist -= 0.1 * totoro_lookat_dist * input_output.scroll_amount;
            totoro_lookat_dist = f32::clamp(totoro_lookat_dist, min, max);
            
            let lookat = glm::look_at(&totoro_lookat_pos, &glm::zero(), &glm::vec3(0.0, 0.0, 1.0));
            let world_space_offset = glm::affine_inverse(lookat) * glm::vec4(-input_output.orientation_delta.x, input_output.orientation_delta.y, 0.0, 0.0);

            totoro_lookat_pos += totoro_lookat_dist * glm::vec4_to_vec3(&world_space_offset);
            let camera_pos = glm::normalize(&totoro_lookat_pos);
            totoro_lookat_pos = totoro_lookat_dist * camera_pos;
            
            let min = -0.95;
            let max = 0.95;
            let lookat_dot = glm::dot(&camera_pos, &glm::vec3(0.0, 0.0, 1.0));
            if lookat_dot > max {
                let rotation_vector = -glm::cross(&camera_pos, &glm::vec3(0.0, 0.0, 1.0));
                let current_angle = f32::acos(lookat_dot);
                let amount = f32::acos(max) - current_angle;

                let new_pos = glm::rotation(amount, &rotation_vector) * glm::vec3_to_vec4(&totoro_lookat_pos);
                totoro_lookat_pos = glm::vec4_to_vec3(&new_pos);
            } else if lookat_dot < min {
                let rotation_vector = -glm::cross(&camera_pos, &glm::vec3(0.0, 0.0, 1.0));
                let current_angle = f32::acos(lookat_dot);
                let amount = f32::acos(min) - current_angle;

                let new_pos = glm::rotation(amount, &rotation_vector) * glm::vec3_to_vec4(&(totoro_lookat_pos));                
                totoro_lookat_pos = glm::vec4_to_vec3(&new_pos);
            }

            let collider = physics_engine.collider_set.get(totoro_list[main_totoro_idx].as_ref().unwrap().collider_handle).unwrap();
            let t = collider.position().translation;
            let lookat_target = t.vector;
            let pos = totoro_lookat_pos + lookat_target;
            let m = glm::look_at(&pos, &lookat_target, &glm::vec3(0.0, 0.0, 1.0));
            renderer.uniform_data.camera_position = glm::vec4(pos.x, pos.y, pos.z, 1.0);
            m
        };
        last_view_from_world = view_from_world;
        renderer.uniform_data.view_from_world = view_from_world;

        for (_, prop) in static_props.iter() {
            for idx in prop.model_indices.iter() {
                renderer.queue_drawcall(*idx, &[prop.model_matrix]);
            }
        }
        
        //Update sun
        sun_pitch += sun_pitch_speed * scaled_delta_time;
        sun_yaw += sun_yaw_speed * scaled_delta_time;
        if sun_pitch > glm::two_pi() {
            sun_pitch -= glm::two_pi::<f32>();
        }
        if sun_yaw > glm::two_pi() {
            sun_yaw -= glm::two_pi::<f32>();
        }

        if let Some(t) = imgui_window_token {
            if imgui::Slider::new("Music volume", 0, 128).build(&imgui_ui, &mut music_volume) { Music::set_volume(music_volume); }
            imgui_ui.checkbox("Freecam", &mut do_freecam);
            imgui_ui.checkbox("Shadow map", &mut dev_gui.do_sun_shadowmap);

            imgui_ui.text(format!("Freecam is at ({:.4}, {:.4}, {:.4})", camera.position.x, camera.position.y, camera.position.z));
            
            if imgui_ui.button_with_size("Totoro's be gone", [0.0, 32.0]) {
                for i in 1..totoro_list.len() {
                    totoro_list.delete(i);
                }
            }
            if imgui_ui.button_with_size("Load static prop", [0.0, 32.0]) {
                if let Some(path) = tfd::open_file_dialog("Choose glb", "./data/models", Some((&["*.glb"], ".glb (Binary gLTF)"))) {
                    let data = gltfutil::gltf_meshdata(&path);                    
                    let model_indices = upload_gltf_primitives(&mut vk, &mut renderer, &data, vk_3D_graphics_pipeline);
                    let model_matrix = glm::translation(&camera.position);
                    let s = StaticProp {
                        model_indices,
                        model_matrix
                    };
                    static_props.insert(s);
                }
            }
            if imgui_ui.button_with_size("Exit", [0.0, 32.0]) {
                break 'running;
            }

            t.end();
        }

        //Resolve the current Dear Imgui frame
        dev_gui.resolve_imgui_frame(&mut vk, &mut renderer, imgui_ui);

        //Pre-render phase

        //We need to wait until it's safe to write GPU data
        unsafe {
            vk.device.wait_for_fences(&[vk.graphics_command_buffer_fence], true, vk::DeviceSize::MAX).unwrap();
        }

        //Update bindless texture sampler descriptors
        if renderer.global_textures.updated {
            renderer.global_textures.updated = false;

            let mut image_infos = vec![default_image_info; renderer.global_textures.size() as usize];
            for i in 0..renderer.global_textures.len() {
                if let Some(info) = renderer.global_textures[i] {
                    image_infos[i] = info;
                }
            }

            let sampler_write = vk::WriteDescriptorSet {
                dst_set: renderer.bindless_descriptor_set,
                descriptor_count: renderer.global_textures.size() as u32,
                descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                p_image_info: image_infos.as_ptr(),
                dst_array_element: 0,
                dst_binding: renderer.samplers_descriptor_index,
                ..Default::default()
            };
            unsafe { vk.device.update_descriptor_sets(&[sampler_write], &[]); }
        }

        //Update bindless material definitions
        if renderer.global_materials.updated {
            renderer.global_materials.updated = false;

            let mut upload_mats = Vec::with_capacity(renderer.global_materials.len());
            for i in 0..renderer.global_materials.len() {
                if let Some(mat) = &renderer.global_materials[i] {
                    upload_mats.push(mat.data());
                }
            }

            renderer.material_buffer.upload_buffer(&mut vk, &upload_mats);
        }
        
        //Update uniform/storage buffers
        {
            let uniforms = &mut renderer.uniform_data;
            //Update static scene data
            uniforms.clip_from_screen = glm::mat4(
                2.0 / window_size.x as f32, 0.0, 0.0, -1.0,
                0.0, 2.0 / window_size.y as f32, 0.0, -1.0,
                0.0, 0.0, 1.0, 0.0,
                0.0, 0.0, 0.0, 1.0
            );

            let near_distance = 0.1;
            let far_distance = 1000.0;
            let projection_matrix = glm::perspective_fov_rh_zo(glm::half_pi::<f32>(), window_size.x as f32, window_size.y as f32, near_distance, far_distance);
            uniforms.clip_from_view = glm::mat4(
                1.0, 0.0, 0.0, 0.0,
                0.0, -1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0,
                0.0, 0.0, 0.0, 1.0,
            ) * projection_matrix;
    
            uniforms.clip_from_world = uniforms.clip_from_view * view_from_world;
            
            //Compute sun direction from pitch and yaw
            uniforms.sun_direction = 
                glm::rotation(sun_yaw, &glm::vec3(0.0, 0.0, 1.0)) *
                glm::rotation(sun_pitch, &glm::vec3(0.0, 1.0, 0.0)) *
                glm::vec4(-1.0, 0.0, 0.0, 0.0);

            uniforms.sun_shadow_matrices = sun_shadow_map.compute_shadow_cascade_matrices(
                &uniforms.sun_direction.xyz(),
                &uniforms.view_from_world,
                &uniforms.clip_from_view
            );

            uniforms.sun_shadow_distances = sun_shadow_map.clip_distances().clone();
            
            //Compute the view-projection matrix for the skybox (the conversion functions are just there to nullify the translation component of the view matrix)
            //The skybox vertices should be rotated along with the camera, but they shouldn't be translated in order to maintain the illusion
            //that the sky is infinitely far away
            uniforms.clip_from_skybox = uniforms.clip_from_view * glm::mat3_to_mat4(&glm::mat4_to_mat3(&view_from_world));

            uniforms.time = timer.elapsed_time;

            let uniform_bytes = struct_to_bytes(&renderer.uniform_data);
            renderer.uniform_buffer.upload_buffer(&mut vk, uniform_bytes);
        };

        //Update model matrix storage buffer
        renderer.instance_buffer.upload_buffer(&mut vk, &renderer.get_instance_data());

        //Draw
        unsafe {
            //Begin acquiring swapchain. This is called as early as possible in order to minimize time waiting
            let current_framebuffer_index = vk.ext_swapchain.acquire_next_image(vk_swapchain.swapchain, vk::DeviceSize::MAX, vk_swapchain_semaphore, vk::Fence::null()).unwrap().0 as usize;

            //Put command buffer in recording state
            vk.device.begin_command_buffer(vk.graphics_command_buffer, &vk::CommandBufferBeginInfo::default()).unwrap();

            //Once-per-frame bindless descriptor setup
            vk.device.cmd_bind_descriptor_sets(vk.graphics_command_buffer, vk::PipelineBindPoint::GRAPHICS, pipeline_layout, 0, &[renderer.bindless_descriptor_set], &[]);

            //Shadow rendering
            let render_area = {
                vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent: vk::Extent2D {
                        width: sun_shadow_map.resolution() * CascadedShadowMap::CASCADE_COUNT as u32,
                        height: sun_shadow_map.resolution() 
                    }
                }
            };
            vk.device.cmd_set_scissor(vk.graphics_command_buffer, 0, &[render_area]);
            let clear_values = [vkutil::DEPTH_STENCIL_CLEAR];
            let rp_begin_info = vk::RenderPassBeginInfo {
                render_pass: shadow_pass,
                framebuffer: sun_shadow_map.framebuffer(),
                render_area,
                clear_value_count: clear_values.len() as u32,
                p_clear_values: clear_values.as_ptr(),
                ..Default::default()
            };
            vk.device.cmd_begin_render_pass(vk.graphics_command_buffer, &rp_begin_info, vk::SubpassContents::INLINE);
            vk.device.cmd_bind_pipeline(vk.graphics_command_buffer, vk::PipelineBindPoint::GRAPHICS, shadow_pipeline);
            for i in 0..CascadedShadowMap::CASCADE_COUNT {
                let viewport = vk::Viewport {
                    x: (i as u32 * sun_shadow_map.resolution()) as f32,
                    y: 0.0,
                    width: sun_shadow_map.resolution() as f32,
                    height: sun_shadow_map.resolution() as f32,
                    min_depth: 0.0,
                    max_depth: 1.0
                };
                vk.device.cmd_set_viewport(vk.graphics_command_buffer, 0, &[viewport]);

                for drawcall in renderer.drawlist_iter() {
                    if let Some(model) = renderer.get_model(drawcall.geometry_idx) {
                        if let ShadowType::NonCaster = model.shadow_type { continue; }

                        let pcs = [
                            model.material_idx.to_le_bytes(),
                            model.position_offset.to_le_bytes(),
                            model.uv_offset.to_le_bytes(),
                            (i as u32).to_le_bytes()
                        ].concat();
                        vk.device.cmd_push_constants(vk.graphics_command_buffer, pipeline_layout, push_constant_shader_stage_flags, 0, &pcs);
                        vk.device.cmd_bind_index_buffer(vk.graphics_command_buffer, model.index_buffer.backing_buffer(), 0, vk::IndexType::UINT32);
                        vk.device.cmd_draw_indexed(vk.graphics_command_buffer, model.index_count, drawcall.instance_count, 0, 0, drawcall.first_instance);
                    }
                }
            }
            vk.device.cmd_end_render_pass(vk.graphics_command_buffer);

            // let sun_shadow_atlas_memory = vk::ImageMemoryBarrier {                
            //     src_access_mask: vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
            //     dst_access_mask: vk::AccessFlags::SHADER_READ,
            //     old_layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            //     new_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            //     image: sun_shadow_map.image(),
            //     subresource_range: vk::ImageSubresourceRange {
            //         aspect_mask: vk::ImageAspectFlags::DEPTH,
            //         base_mip_level: 0,
            //         level_count: 1,
            //         base_array_layer: 0,
            //         layer_count: 1
            //     },
            //     ..Default::default()
            // };
            // vk.device.cmd_pipeline_barrier(
            //     vk.graphics_command_buffer,
            //     vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
            //     vk::PipelineStageFlags::FRAGMENT_SHADER,
            //     vk::DependencyFlags::empty(),
            //     &[],
            //     &[],
            //     &[sun_shadow_atlas_memory]
            // );
            
            //Set the viewport for this frame
            let viewport = vk::Viewport {
                x: 0.0,
                y: 0.0,
                width: (vk_swapchain.extent.width) as f32,
                height: (vk_swapchain.extent.height) as f32,
                min_depth: 0.0,
                max_depth: 1.0
            };
            vk.device.cmd_set_viewport(vk.graphics_command_buffer, 0, &[viewport]);

            //Set scissor rect to be same as render area
            let vk_render_area = {
                vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent: vk_swapchain.extent
                }
            };
            let scissor_area = vk::Rect2D {
                offset: vk::Offset2D::default(),
                extent: vk::Extent2D {
                    width: window_size.x,
                    height: window_size.y
                }
            };
            vk.device.cmd_set_scissor(vk.graphics_command_buffer, 0, &[scissor_area]);

            let vk_clear_values = [vkutil::COLOR_CLEAR, vkutil::DEPTH_STENCIL_CLEAR];
            let rp_begin_info = vk::RenderPassBeginInfo {
                render_pass: main_forward_pass,
                framebuffer: vk_swapchain.swapchain_framebuffers[current_framebuffer_index],
                render_area: vk_render_area,
                clear_value_count: vk_clear_values.len() as u32,
                p_clear_values: vk_clear_values.as_ptr(),
                ..Default::default()
            };
            vk.device.cmd_begin_render_pass(vk.graphics_command_buffer, &rp_begin_info, vk::SubpassContents::INLINE);

            //Iterate through draw calls
            let mut last_bound_pipeline = vk::Pipeline::default();
            for drawcall in renderer.drawlist_iter() {
                if drawcall.pipeline != last_bound_pipeline {
                    vk.device.cmd_bind_pipeline(vk.graphics_command_buffer, vk::PipelineBindPoint::GRAPHICS, drawcall.pipeline);
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
                    vk.device.cmd_push_constants(vk.graphics_command_buffer, pipeline_layout, push_constant_shader_stage_flags, 0, &pcs);
                    vk.device.cmd_bind_index_buffer(vk.graphics_command_buffer, model.index_buffer.backing_buffer(), 0, vk::IndexType::UINT32);
                    vk.device.cmd_draw_indexed(vk.graphics_command_buffer, model.index_count, drawcall.instance_count, 0, 0, drawcall.first_instance);
                }
            }

            //Record atmosphere rendering commands
            vk.device.cmd_bind_pipeline(vk.graphics_command_buffer, vk::PipelineBindPoint::GRAPHICS, atmosphere_pipeline);
            vk.device.cmd_draw(vk.graphics_command_buffer, 36, 1, 0, 0);

            //Record Dear ImGUI drawing commands
            dev_gui.record_draw_commands(&mut vk, pipeline_layout);

            vk.device.cmd_end_render_pass(vk.graphics_command_buffer);

            vk.device.end_command_buffer(vk.graphics_command_buffer).unwrap();

            let submit_info = vk::SubmitInfo {
                wait_semaphore_count: 1,
                p_wait_semaphores: &vk_swapchain_semaphore,
                p_wait_dst_stage_mask: &vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                signal_semaphore_count: 1,
                p_signal_semaphores: &vk_rendercomplete_semaphore,
                command_buffer_count: 1,
                p_command_buffers: &vk.graphics_command_buffer,
                ..Default::default()
            };

            let queue = vk.device.get_device_queue(vk.graphics_queue_family_index, 0);
            vk.device.reset_fences(&[vk.graphics_command_buffer_fence]).unwrap();
            vk.device.queue_submit(queue, &[submit_info], vk.graphics_command_buffer_fence).unwrap();

            let present_info = vk::PresentInfoKHR {
                swapchain_count: 1,
                p_swapchains: &vk_swapchain.swapchain,
                p_image_indices: &(current_framebuffer_index as u32),
                wait_semaphore_count: 1,
                p_wait_semaphores: &vk_rendercomplete_semaphore,
                ..Default::default()
            };
            if let Err(e) = vk.ext_swapchain.queue_present(queue, &present_info) {
                println!("{}", e);
            }
        }
    }

    //Cleanup
    unsafe {
        vk.device.wait_for_fences(&[vk.graphics_command_buffer_fence], true, vk::DeviceSize::MAX).unwrap();
    }
}
