#![allow(non_snake_case)]

//Alias some library names
extern crate nalgebra_glm as glm;
extern crate tinyfiledialogs as tfd;
extern crate ozy_engine as ozy;

mod gltfutil;
mod gui;
mod input;
mod render;
mod routines;
mod structs;

#[macro_use]
mod vkutil;

use ash::vk;
use gltfutil::GLTFPrimitive;
use gpu_allocator::MemoryLocation;
use imgui::{FontAtlasRefMut};
use routines::struct_to_bytes;
use sdl2::event::Event;
use sdl2::mixer;
use sdl2::mixer::Music;
use std::fmt::Display;
use std::fs::{File};
use std::ffi::CStr;
use std::mem::size_of;
use std::time::SystemTime;

use ozy::structs::{FrameTimer, OptionVec};

use input::UserInput;
use vkutil::{ColorSpace, FreeList, GPUBuffer, VirtualImage, VulkanAPI};
use structs::{Camera, NoiseParameters, TerrainSpec};
use render::{DrawData, Renderer, MaterialData};

use crate::gltfutil::GLTFData;
use crate::gui::DevGui;

fn crash_with_error_dialog(message: &str) -> ! {
    crash_with_error_dialog_titled("Oops...", message);
}
fn crash_with_error_dialog_titled(title: &str, message: &str) -> ! {
    tfd::message_box_ok(title, &message.replace("'", ""), tfd::MessageBoxIcon::Error);
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

fn time_from_epoch_ms() -> u128 {
    SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis()
}

fn regenerate_terrain(vk: &mut VulkanAPI, spec: &mut TerrainSpec, fixed_seed: bool) -> Vec<f32> {
    if !fixed_seed {
        spec.seed = time_from_epoch_ms();
    }
    spec.generate_vertices()
}

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

fn upload_gltf_primitives(vk: &mut VulkanAPI, renderer: &mut Renderer, data: &GLTFData) -> Vec<usize> {
    let mut indices = vec![];
    for prim in &data.primitives {
        let color_idx;
        if prim.material.color_bytes.len() != 0 {
            let color_image = VirtualImage::from_png_bytes(vk, prim.material.color_bytes.as_slice());
            let image_info = vk::DescriptorImageInfo {
                sampler: renderer.material_sampler,
                image_view: color_image.vk_view,
                image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
            };
            color_idx = renderer.global_textures.insert(image_info) as u32;
        } else {
            color_idx = renderer.default_color_idx;
        }

        let normal_idx = match &prim.material.normal_bytes {
            Some(bytes) => {
                let normal_image = VirtualImage::from_png_bytes(vk, bytes.as_slice());
                let image_info = vk::DescriptorImageInfo {
                    sampler: renderer.material_sampler,
                    image_view: normal_image.vk_view,
                    image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
                };
                renderer.global_textures.insert(image_info) as u32
            }
            None => { renderer.default_normal_idx }
        };

        let material_idx = renderer.global_materials.insert(MaterialData::new(prim.material.base_color, color_idx, normal_idx)) as u32;

        let offsets = upload_primitive_vertices(vk, renderer, &prim);

        let index_buffer = vkutil::make_index_buffer(vk, &prim.indices);
        let model_idx = renderer.register_model(DrawData {
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
    drop(surf_formats);
    
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
    let mut vk_display = vkutil::Display::init(&mut vk, &vk_ext_swapchain, vk_render_pass);

    let push_constant_shader_stage_flags = vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT;
    let pipeline_creator = unsafe {
        let push_constant_range = vk::PushConstantRange {
            stage_flags: push_constant_shader_stage_flags,
            offset: 0,
            size: vk.push_constant_size
        };
        let pipeline_layout_createinfo = vk::PipelineLayoutCreateInfo {
            push_constant_range_count: 1,
            p_push_constant_ranges: &push_constant_range,
            set_layout_count: 1,
            p_set_layouts: &renderer.descriptor_set_layout,
            ..Default::default()
        };
        
        let layout = vk.device.create_pipeline_layout(&pipeline_layout_createinfo, vkutil::MEMORY_ALLOCATOR).unwrap();
        vkutil::PipelineCreator::init(layout)
    };

    //Create pipelines
    let [vk_3D_graphics_pipeline, terrain_pipeline, atmosphere_pipeline, vk_wireframe_graphics_pipeline] = unsafe {
        //Load shaders
        let main_shader_stages = {
            let v = vkutil::load_shader_stage(&vk.device, vk::ShaderStageFlags::VERTEX, "./data/shaders/vertex_main.spv");
            let f = vkutil::load_shader_stage(&vk.device, vk::ShaderStageFlags::FRAGMENT, "./data/shaders/pbr_metallic_roughness.spv");
            [v, f]
        };
        
        let terrain_shader_stages = {
            let v = vkutil::load_shader_stage(&vk.device, vk::ShaderStageFlags::VERTEX, "./data/shaders/vertex_main.spv");
            let f = vkutil::load_shader_stage(&vk.device, vk::ShaderStageFlags::FRAGMENT, "./data/shaders/terrain.spv");
            [v, f]
        };
        
        let atm_shader_stages = {
            let v = vkutil::load_shader_stage(&vk.device, vk::ShaderStageFlags::VERTEX, "./data/shaders/atmosphere_vert.spv");
            let f = vkutil::load_shader_stage(&vk.device, vk::ShaderStageFlags::FRAGMENT, "./data/shaders/atmosphere_frag.spv");
            [v, f]
        };

        let mut main_create_info = vkutil::VirtualPipelineCreateInfo::new(vk_render_pass, vkutil::VertexInputConfiguration::empty(), &main_shader_stages);
        let main_pipeline = pipeline_creator.create_pipeline(&vk, &main_create_info);
        main_create_info.shader_stages = &terrain_shader_stages;
        let ter_pipeline = pipeline_creator.create_pipeline(&vk, &main_create_info);

        main_create_info.rasterization_state = Some(vk::PipelineRasterizationStateCreateInfo {
            polygon_mode: vk::PolygonMode::LINE,
            ..pipeline_creator.default_rasterization_state
        });
        let wire_pipeline = pipeline_creator.create_pipeline(&vk, &main_create_info);
        main_create_info.rasterization_state = None;

        let atm_create_info = vkutil::VirtualPipelineCreateInfo::new(vk_render_pass, vkutil::VertexInputConfiguration::empty(), &atm_shader_stages);
        let atm_pipeline = pipeline_creator.create_pipeline(&vk, &atm_create_info);

        [main_pipeline, ter_pipeline, atm_pipeline, wire_pipeline]
    };

    let noise_parameters = {
        const OCTAVES: usize = 8;
        const LACUNARITY: f64 = 1.75;
        const GAIN: f64 = 0.5;
        let mut ps = Vec::with_capacity(OCTAVES);
        
        let mut amplitude = 2.0;
        let mut frequency = 0.15;
        for _ in 0..OCTAVES {
            ps.push(NoiseParameters {
                amplitude,
                frequency
            });
            amplitude *= GAIN;
            frequency *= LACUNARITY;
        }
        ps
    };

    //Define terrain
    let terrain_width_height = 256;
    let mut terrain_fixed_seed = false;
    let mut terrain_interactive_generation = false;
    let mut terrain = TerrainSpec {
        vertex_width: terrain_width_height,
        vertex_height: terrain_width_height,
        noise_parameters,
        amplitude: 2.0,
        exponent: 2.2,
        seed: time_from_epoch_ms()
    };

    let terrain_vertices = terrain.generate_vertices();
    
    //Loading terrain textures
    let grass_color_global_index = vkutil::load_global_bc7(&mut vk, &mut renderer.global_textures, renderer.material_sampler, "./data/textures/whispy_grass/color.dds", ColorSpace::SRGB);
    let grass_normal_global_index = vkutil::load_global_bc7(&mut vk, &mut renderer.global_textures, renderer.material_sampler, "./data/textures/whispy_grass/normal.dds", ColorSpace::LINEAR);
    let rock_color_global_index = vkutil::load_global_bc7(&mut vk, &mut renderer.global_textures, renderer.material_sampler, "./data/textures/rocky_ground/color.dds", ColorSpace::SRGB);
    let rock_normal_global_index = vkutil::load_global_bc7(&mut vk, &mut renderer.global_textures, renderer.material_sampler, "./data/textures/rocky_ground/normal.dds", ColorSpace::LINEAR);

    let terrain_grass_matidx = renderer.global_materials.insert(MaterialData::new([1.0; 4], grass_color_global_index, grass_normal_global_index)) as u32;
    let terrain_rock_matidx = renderer.global_materials.insert(MaterialData::new([1.0; 4], rock_color_global_index, rock_normal_global_index)) as u32;

    //Load gltf object
    let glb_name = "./data/models/nice_tree.glb";
    let tree_data = gltfutil::gltf_meshdata(glb_name);

    //Register each primitive with the renderer
    let tree_model_indices = upload_gltf_primitives(&mut vk, &mut renderer, &tree_data);

    //Load totoro as glb
    let totoro_data = gltfutil::gltf_meshdata("./data/models/totoro_backup.glb");

    //Register each primitive with the renderer
    let totoro_model_indices = upload_gltf_primitives(&mut vk, &mut renderer, &totoro_data);

    //Upload terrain geometry
    let terrain_model_idx = {
        let terrain_indices = ozy::prims::plane_index_buffer(terrain_width_height, terrain_width_height);
        let terrain_offsets = uninterleave_and_upload_vertices(&mut vk, &mut renderer, &terrain_vertices);
        drop(terrain_vertices);
        let index_buffer = vkutil::make_index_buffer(&mut vk, &terrain_indices);
        renderer.register_model(DrawData {
            index_buffer,
            index_count: terrain_indices.len().try_into().unwrap(),
            position_offset: terrain_offsets.position_offset,
            tangent_offset: terrain_offsets.tangent_offset,
            normal_offset: terrain_offsets.normal_offset,
            uv_offset: terrain_offsets.uv_offset,
            material_idx: terrain_grass_matidx
        })
    };

    //Create semaphore used to wait on swapchain image
    let vk_swapchain_semaphore = unsafe { vk.device.create_semaphore(&vk::SemaphoreCreateInfo::default(), vkutil::MEMORY_ALLOCATOR).unwrap() };
    let vk_rendercomplete_semaphore = unsafe { vk.device.create_semaphore(&vk::SemaphoreCreateInfo::default(), vkutil::MEMORY_ALLOCATOR).unwrap() };

    //State for freecam controls
    let mut camera = Camera::new(glm::vec3(0.0f32, -30.0, 15.0));
    let mut last_view_from_world = glm::identity();
    let mut do_freecam = true;

    let mut totoro_position: glm::TVec3<f32> = glm::zero();
    let mut totoro_lookat_dist = 7.5;
    let mut totoro_lookat_pos = totoro_lookat_dist * glm::normalize(&glm::vec3(-1.0f32, 0.0, 1.75));

    let mut timer = FrameTimer::new();      //Struct for doing basic framerate independence

    let mut sun_speed = 0.003;
    let mut sun_pitch = 0.118;
    let mut sun_yaw = 0.783;
    let mut trees_width = 1;
    let mut trees_height = 1;

    renderer.uniform_data.sun_luminance = [2.0, 2.0, 2.0, 0.0];
    renderer.uniform_data.stars_threshold = 8.0;
    renderer.uniform_data.stars_exposure = 200.0;
    renderer.uniform_data.fog_density = 0.75;
    
    //Load and play bgm
    let bgm = unwrap_result(Music::from_file("./data/music/relaxing_botw.mp3"), "Error loading bgm");
    bgm.play(-1).unwrap();

    let mut dev_gui = DevGui::new(&mut vk, vk_render_pass, &pipeline_creator);
    
    let mut wireframe = false;
    let mut main_pipeline = vk_3D_graphics_pipeline;

    let mut input_system = input::InputSystem::init(&sdl_context);

    //Main application loop
    'running: loop {
        timer.update(); //Update frame timer

        //Reset renderer
        renderer.reset();

        //Input sampling
        let imgui_io = imgui_context.io_mut();
        let input_output = match input_system.do_thing(&timer, imgui_io) {
            UserInput::Output(o) => { o }
            UserInput::ExitProgram => { break 'running; }
        };

        //Handling of some input results before update
        if input_output.gui_toggle { dev_gui.do_main_window = !dev_gui.do_main_window }
        if input_output.regen_terrain {
            if let Some(ter) = renderer.get_model(terrain_model_idx) {
                let offset = ter.position_offset;
                let verts = regenerate_terrain(&mut vk, &mut terrain, terrain_fixed_seed);
                replace_uploaded_uninterleaved_vertices(&mut vk, &mut renderer, &verts, offset.into());
            }
        }

        //Handle needing to resize the window
        unsafe {
            if input_output.resize_window {
                vk.device.wait_for_fences(&[vk.graphics_command_buffer_fence], true, vk::DeviceSize::MAX).unwrap();

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
                vk_display = vkutil::Display::init(&mut vk, &vk_ext_swapchain, vk_render_pass);

                window_size = glm::vec2(vk_display.extent.width, vk_display.extent.height);
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
        if dev_gui.do_main_window && dev_gui.do_terrain_window {
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
                    if let Some(terrain_model) = renderer.get_model(terrain_model_idx) {
                        let vert_offset = terrain_model.position_offset;
                        let verts = regenerate_terrain(&mut vk, &mut terrain, terrain_fixed_seed);
                        replace_uploaded_uninterleaved_vertices(&mut vk, &mut renderer, &verts, vert_offset.into());
                    }
                }

                if terrain_interactive_generation && parameters_changed {
                    if let Some(terrain_model) = renderer.get_model(terrain_model_idx) {
                        let vert_offset = terrain_model.position_offset;
                        let verts = regenerate_terrain(&mut vk, &mut terrain, terrain_fixed_seed);
                        replace_uploaded_uninterleaved_vertices(&mut vk, &mut renderer, &verts, vert_offset.into());
                    }
                }

                if imgui_ui.button_with_size("Close", [0.0, 32.0]) { dev_gui.do_terrain_window = false; }

                token.end();
            }
        }

        let plane_model_matrix = glm::scaling(&glm::vec3(30.0, 30.0, 30.0));
        renderer.queue_drawcall(terrain_model_idx, terrain_pipeline, &[plane_model_matrix]);

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
        } else {
            let delta_pos = FREECAM_SPEED * glm::affine_inverse(last_view_from_world) * view_movement_vector * timer.delta_time;
            totoro_position += glm::vec4_to_vec3(&delta_pos);
        }
 
        //Totoro update
        let a = ozy::routines::uniform_scale(2.0);
        let b = glm::translation(&totoro_position) * glm::rotation(timer.elapsed_time, &glm::vec3(0.0, 0.0, 1.0));
        let totoro_model_matrix = b * a;
        for idx in &totoro_model_indices {
            renderer.queue_drawcall(*idx, main_pipeline, &[totoro_model_matrix]);
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

            let lookat_target = glm::vec3(totoro_position.x, totoro_position.y, totoro_position.z + 3.0);
            let pos = totoro_lookat_pos + lookat_target;
            let m = glm::look_at(&pos, &lookat_target, &glm::vec3(0.0, 0.0, 1.0));
            renderer.uniform_data.camera_position = glm::vec4(pos.x, pos.y, pos.z, 1.0);
            m
        };
        last_view_from_world = view_from_world;
        renderer.uniform_data.view_from_world = view_from_world;

        let imgui_window_token = if dev_gui.do_main_window {
            imgui::Window::new("Main control panel (press ESC to hide)").menu_bar(true).begin(&imgui_ui)
        } else {
            None
        };

        if let Some(_) = imgui_window_token {
            if let Some(mb) = imgui_ui.begin_menu_bar() {
                if let Some(mt) = imgui_ui.begin_menu("Environment") {
                    if imgui::MenuItem::new("Terrain builder").build(&imgui_ui) {
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

            imgui::Slider::new("Sun speed", 0.0, 1.0).build(&imgui_ui, &mut sun_speed);
            imgui::Slider::new("Sun pitch", 0.0, glm::two_pi::<f32>()).build(&imgui_ui, &mut sun_pitch);
            imgui::Slider::new("Sun yaw", 0.0, glm::two_pi::<f32>()).build(&imgui_ui, &mut sun_yaw);
            imgui::Slider::new("Stars threshold", 0.0, 16.0).build(&imgui_ui, &mut renderer.uniform_data.stars_threshold);
            imgui::Slider::new("Stars exposure", 0.0, 1000.0).build(&imgui_ui, &mut renderer.uniform_data.stars_exposure);
            imgui::Slider::new("Fog factor", 0.0, 8.0).build(&imgui_ui, &mut renderer.uniform_data.fog_density);
            imgui::Slider::new("Trees width", 1, 10).build(&imgui_ui, &mut trees_width);
            imgui::Slider::new("Trees height", 1, 10).build(&imgui_ui, &mut trees_height);
        }

        let bb_mats = {
            let mut ms = Vec::with_capacity((trees_width * trees_height) as usize);
            for i in 0..trees_width {
                for j in 0..trees_height {
                    let mat = glm::translation(&glm::vec3(25.0 * i as f32 - 50.0, 25.0 * j as f32 - 50.0, 0.0)) * ozy::routines::uniform_scale(3.0);
                    ms.push(mat);
                }
            }
            ms
        };
        for idx in &tree_model_indices {
            renderer.queue_drawcall(*idx, main_pipeline, &bb_mats);
        }
        
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

            imgui_ui.text(format!("Freecam is at ({:.4}, {:.4}, {:.4})", camera.position.x, camera.position.y, camera.position.z));
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
            vk.device.reset_fences(&[vk.graphics_command_buffer_fence]).unwrap();
        }

        //Destroy Dear ImGUI allocations from last frame
        {
            let last_frame = dev_gui.current_frame.overflowing_sub(1).0 % DevGui::FRAMES_IN_FLIGHT;
            let geo_count = dev_gui.frames[last_frame].index_buffers.len();
            for geo in dev_gui.frames[last_frame].index_buffers.drain(0..geo_count) {
                geo.free(&mut vk);
            }
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
                dst_set: renderer.descriptor_sets[0],
                descriptor_count: renderer.global_textures.size() as u32,
                descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                p_image_info: image_infos.as_ptr(),
                dst_array_element: 0,
                dst_binding: 8,
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
                    upload_mats.push(mat.clone());
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

            let projection_matrix = glm::perspective_fov_rh_zo(glm::half_pi::<f32>(), window_size.x as f32, window_size.y as f32, 0.1, 10000.0);
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
            let current_framebuffer_index = vk_ext_swapchain.acquire_next_image(vk_display.swapchain, vk::DeviceSize::MAX, vk_swapchain_semaphore, vk::Fence::null()).unwrap().0 as usize;

            //Put command buffer in recording state
            vk.device.begin_command_buffer(vk.graphics_command_buffer, &vk::CommandBufferBeginInfo::default()).unwrap();
            
            //Set the viewport for this frame
            let viewport = vk::Viewport {
                x: 0.0,
                y: 0.0,
                width: (vk_display.extent.width) as f32,
                height: (vk_display.extent.height) as f32,
                min_depth: 0.0,
                max_depth: 1.0
            };
            vk.device.cmd_set_viewport(vk.graphics_command_buffer, 0, &[viewport]);

            //Set scissor rect to be same as render area
            let vk_render_area = {
                vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
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
            vk.device.cmd_set_scissor(vk.graphics_command_buffer, 0, &[scissor_area]);

            let vk_clear_values = [vkutil::COLOR_CLEAR, vkutil::DEPTH_STENCIL_CLEAR];
            let rp_begin_info = vk::RenderPassBeginInfo {
                render_pass: vk_render_pass,
                framebuffer: vk_display.swapchain_framebuffers[current_framebuffer_index],
                render_area: vk_render_area,
                clear_value_count: vk_clear_values.len() as u32,
                p_clear_values: vk_clear_values.as_ptr(),
                ..Default::default()
            };
            vk.device.cmd_begin_render_pass(vk.graphics_command_buffer, &rp_begin_info, vk::SubpassContents::INLINE);

            //Once-per-frame bindless descriptor setup
            vk.device.cmd_bind_descriptor_sets(vk.graphics_command_buffer, vk::PipelineBindPoint::GRAPHICS, pipeline_creator.pipeline_layout, 0, &renderer.descriptor_sets, &[]);

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
                    vk.device.cmd_push_constants(vk.graphics_command_buffer, pipeline_creator.pipeline_layout, push_constant_shader_stage_flags, 0, &pcs);
                    vk.device.cmd_bind_index_buffer(vk.graphics_command_buffer, model.index_buffer.backing_buffer(), 0, vk::IndexType::UINT32);
                    vk.device.cmd_draw_indexed(vk.graphics_command_buffer, model.index_count, drawcall.instance_count, 0, 0, drawcall.first_instance);
                }
            }

            //Record atmosphere rendering commands
            vk.device.cmd_bind_pipeline(vk.graphics_command_buffer, vk::PipelineBindPoint::GRAPHICS, atmosphere_pipeline);
            vk.device.cmd_draw(vk.graphics_command_buffer, 36, 1, 0, 0);

            //Record Dear ImGUI drawing commands
            dev_gui.record_draw_commands(&mut vk, pipeline_creator.pipeline_layout);

            vk.device.cmd_end_render_pass(vk.graphics_command_buffer);

            vk.device.end_command_buffer(vk.graphics_command_buffer).unwrap();

            let pipeline_stage_flags = vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT;
            let submit_info = vk::SubmitInfo {
                wait_semaphore_count: 1,
                p_wait_semaphores: &vk_swapchain_semaphore,
                p_wait_dst_stage_mask: &pipeline_stage_flags,
                signal_semaphore_count: 1,
                p_signal_semaphores: &vk_rendercomplete_semaphore,
                command_buffer_count: 1,
                p_command_buffers: &vk.graphics_command_buffer,
                ..Default::default()
            };

            let queue = vk.device.get_device_queue(vk.graphics_queue_family_index, 0);
            vk.device.queue_submit(queue, &[submit_info], vk.graphics_command_buffer_fence).unwrap();

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

    //Cleanup
    unsafe {
        vk.device.wait_for_fences(&[vk.graphics_command_buffer_fence], true, vk::DeviceSize::MAX).unwrap();
    }
}
