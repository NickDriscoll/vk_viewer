use ash::vk;
use imgui::{DrawCmd, Ui, TreeNodeId};
use crate::render::{Renderer};
use crate::structs::{TerrainSpec, EntityKey};
use render::vkdevice::*;
use crate::*;

pub enum AssetWindowResponse {
    OptimizeGLB(String),
    None
}

pub enum EntityWindowResponse {
    CloneEntity(EntityKey),
    DeleteEntity(EntityKey),
    LoadGLTF(String),
    LoadOzyMesh(String),
    FocusCamera(Option<EntityKey>),
    Interacted,
    None
}

#[derive(Default)]
pub struct DevGui {
    pub pipeline: vk::Pipeline,
    pub frames: Vec<DevGuiFrame>,
    pub current_frame: usize,
    pub do_gui: bool,
    pub do_terrain_window: bool,
    pub do_entity_window: bool,
    pub do_asset_window: bool,
    pub do_mat_list: bool,
    pub do_sun_window: bool,
    pub do_camera_window: bool
}

impl DevGui {
    pub const FLOATS_PER_VERTEX: usize = 8;

    pub fn do_standard_button<S: AsRef<str>>(ui: &Ui, label: S) -> bool { ui.button_with_size(label, [0.0, 32.0]) }

    pub fn new(gpu: &mut VulkanGraphicsDevice, render_pass: vk::RenderPass, pipeline_layout: vk::PipelineLayout) -> Self {
        use render::GraphicsPipelineBuilder;

        let mut frames = Vec::with_capacity(Renderer::FRAMES_IN_FLIGHT);
        for _ in 0..Renderer::FRAMES_IN_FLIGHT {
            frames.push(DevGuiFrame::default());
        }
        
        let im_shader_stages = {
            let v = load_shader_stage(&gpu.device, vk::ShaderStageFlags::VERTEX, "./data/shaders/imgui_vert.spv");
            let f = load_shader_stage(&gpu.device, vk::ShaderStageFlags::FRAGMENT, "./data/shaders/imgui_frag.spv");
            vec![v, f]
        };
        let im_info = GraphicsPipelineBuilder::init(render_pass, pipeline_layout)
            .set_shader_stages(im_shader_stages)
            .set_depth_test(vk::FALSE)           
            .set_cull_mode(vk::CullModeFlags::NONE) 
            .build_info();

        let pipeline = unsafe { GraphicsPipelineBuilder::create_pipelines(gpu, &[im_info])[0] };

        DevGui {
            pipeline,
            frames,
            current_frame: 0,
            do_gui: true,
            do_entity_window: true,
            ..Default::default()
        }
    }

    pub fn do_asset_window(&mut self, ui: &Ui, path: &str) -> AssetWindowResponse {
        let mut response = AssetWindowResponse::None;
        if !self.do_asset_window || !self.do_gui { return response; }
        if let Some(win_t) = ui.window("Asset manager").begin() {
            let p = Path::new(path);
            let mut i = 0;
            for entry in p.read_dir().unwrap() {
                let entry = entry.unwrap();
                let metadata = entry.metadata().unwrap();
                let e_path = entry.path();
                let name = e_path.file_stem().unwrap().to_string_lossy();

                // if let Some(tree_token) = imgui::TreeNode::new(TreeNodeId::Str(&format!("{}", i))).label::<TreeNodeId<&str>, &str>(&prop.name).push(ui) {

                // }

                if metadata.is_file() {
                    ui.text(entry.file_name().to_string_lossy());
                    let parent_dir = p.parent().unwrap();
                    let opt_path = format!("{}/.optimized/{}.ozy", parent_dir.as_os_str().to_str().unwrap(), name);
                    let opt_dir = Path::new(&opt_path);
                    if opt_dir.exists() {
                        ui.text("Asset has been optimized.");
                    } else {
                        if Self::do_standard_button(ui, format!("Optimize###{}", i)) {
                            response = AssetWindowResponse::OptimizeGLB(String::from(entry.path().to_str().unwrap()));
                        }
                    }
                }
                ui.separator();

                i += 1;
            }
            if DevGui::do_standard_button(ui, "Close") { self.do_asset_window = false; }
            win_t.end();
        }
        response
    }

    pub fn do_entity_window(&mut self, ui: &Ui, window_size: glm::TVec2<u32>, entities: &mut DenseSlotMap<EntityKey, Entity>, focused_entity: Option<EntityKey>, rigid_body_set: &mut RigidBodySet) -> EntityWindowResponse {
        let mut out = EntityWindowResponse::None;
        if !self.do_entity_window || !self.do_gui { return out; }
        
        let mut interacted = false;
        let window_x = window_size.x as f32;
        let window_y = window_size.y as f32;
        if let Some(win_token) = ui.window("Entity window")
            .position(mint::Vector2{x: window_x, y: 0.0}, imgui::Condition::Always)
            .position_pivot(mint::Vector2 {x: 1.0, y: 0.0})
            .size(mint::Vector2 { x: window_x * 1.0 / 4.0, y: window_y }, imgui::Condition::Always)
            .resizable(false)
            .collapsible(false)
            .begin() {
            let mut i = 0;
            let mut cloned_item = None;
            let mut deleted_item = None;
            for prop in entities.iter_mut() {
                let prop_key = prop.0;
                let prop = prop.1;

                if let Some(token) = ui.tree_node_config(TreeNodeId::Str(&format!("{}", i))).label::<TreeNodeId<&str>, &str>(&prop.name).push() {
                    if let Some(body) = rigid_body_set.get_mut(prop.physics_component.rigid_body_handle) {
                        let mut pos = body.position().clone();
                        let rot = body.rotation();
                        let mut angles = quaternion_to_euler(&rot);

                        interacted |= imgui::Drag::new("X").speed(0.1).build(ui, &mut pos.translation.x);
                        interacted |= imgui::Drag::new("Y").speed(0.1).build(ui, &mut pos.translation.y);
                        interacted |= imgui::Drag::new("Z").speed(0.1).build(ui, &mut pos.translation.z);   
                        interacted |= imgui::Drag::new("Pitch").speed(0.05).build(ui, &mut angles[0]);
                        interacted |= imgui::Drag::new("Yaw").speed(0.05).build(ui, &mut angles[2]);
                        interacted |= imgui::Drag::new("Roll").speed(0.05).build(ui, &mut angles[1]);                        
                        interacted |= imgui::Drag::new("Scale").speed(0.05).build(ui, &mut prop.physics_component.scale);

                        if interacted {
                            body.set_position(pos, true);
                            body.set_rotation(angles, true);
                        }
                    }


                    let mut b = false;
                    if let Some(key) = focused_entity {
                        b = prop_key == key;
                    }
                    if ui.checkbox("Focus camera", &mut b) {
                        out = if !b { EntityWindowResponse::FocusCamera(None) }
                              else { EntityWindowResponse::FocusCamera(Some(prop_key)) };
                    }

                    if Self::do_standard_button(ui, "Clone") {
                        cloned_item = Some(prop_key);
                    }
                    ui.same_line();
                    if Self::do_standard_button(ui, "Delete") {
                        deleted_item = Some(prop_key);
                    }
                    ui.separator();
                    token.pop();
                }

                i += 1;
            }
            ui.separator();

            if let Some(key) = cloned_item {
                out = EntityWindowResponse::CloneEntity(key);
            }

            if let Some(key) = deleted_item {
                out = EntityWindowResponse::DeleteEntity(key);
            }
            
            if Self::do_standard_button(ui, "Import glTF") {
                if let Some(path) = tfd::open_file_dialog("Choose glb", "./data/models", Some((&["*.glb"], ".glb (Binary gLTF)"))) {
                    out = EntityWindowResponse::LoadGLTF(path);
                }
            }
            
            if Self::do_standard_button(ui, "Import OzyMesh") {
                if let Some(path) = tfd::open_file_dialog("Choose OzyMesh", "./data/models/.optimized", Some((&["*.ozy"], ".ozy (Optimized model)"))) {
                    out = EntityWindowResponse::LoadOzyMesh(path);
                }
            }

            if DevGui::do_standard_button(ui, "Close") { self.do_entity_window = false; }
            win_token.end();
        }
        if interacted { out = EntityWindowResponse::Interacted; }

        out
    }

    pub fn do_material_list(&mut self, imgui_ui: &Ui, renderer: &mut Renderer) {
        if !self.do_mat_list || !self.do_gui { return; }
        if let Some(token) = imgui_ui.window("Loaded materials").begin() {
            for i in 0..renderer.global_materials.len() {
                if let Some(mat) = &renderer.global_materials[i] {
                    imgui_ui.text(format!("{:#?}", mat));
                    
                    imgui_ui.separator();
                }
            }
            if DevGui::do_standard_button(imgui_ui, "Close") { self.do_mat_list = false; }
            token.end();
        }
    }

    pub fn do_terrain_window(&mut self, imgui_ui: &Ui, terrain: &mut TerrainSpec) -> bool {
        let mut regen_terrain = false;
        if self.do_gui && self.do_terrain_window {
            if let Some(token) = imgui_ui.window("Terrain generator").begin() {
                let mut parameters_changed = false;

                imgui_ui.text("Global terrain variables:");
                parameters_changed |= imgui_ui.slider("Amplitude", 0.0, 8.0, &mut terrain.amplitude);
                parameters_changed |= imgui_ui.slider("Exponent", 1.0, 5.0, &mut terrain.exponent);
                parameters_changed |= imgui_ui.slider("Octaves", 1, 16, &mut terrain.octaves);
                parameters_changed |= imgui_ui.slider("Lacunarity", 0.0, 5.0, &mut terrain.lacunarity);
                parameters_changed |= imgui_ui.slider("Gain", 0.0, 2.0, &mut terrain.gain);
                parameters_changed |= imgui_ui.slider("Scale", 1.0, 50.0, &mut terrain.scale);
                imgui_ui.separator();

                imgui_ui.text(format!("Last seed used: 0x{:X}", terrain.seed));
                imgui_ui.checkbox("Use fixed seed", &mut terrain.fixed_seed);
                imgui_ui.checkbox("Interactive mode", &mut terrain.interactive_generation);
                
                let regen_button = DevGui::do_standard_button(&imgui_ui, "Regenerate");

                if DevGui::do_standard_button(&imgui_ui, "Close") { self.do_terrain_window = false; }

                token.end();

                regen_terrain = (terrain.interactive_generation && parameters_changed) || regen_button;
            }
        }
        regen_terrain
    }

    pub fn do_camera_window(&mut self, ui: &Ui, camera: &mut Camera) {
        if !self.do_camera_window || !self.do_gui { return; }
        if let Some(token) = ui.window("Camera").begin() {
            ui.slider("FOV", 0.001, glm::pi(), &mut camera.fov);

            token.end();
        }
    }

    pub fn do_sun_window(&mut self, ui: &Ui, sun: &mut SunLight) {
        if !self.do_sun_window || !self.do_gui { return; }
        if let Some(token) = ui.window("Sun controls").begin() {
            ui.slider("Sun pitch speed", 0.0, 1.0, &mut sun.pitch_speed);
            ui.slider("Sun pitch", 0.0, glm::two_pi::<f32>(), &mut sun.pitch);
            ui.slider("Sun yaw speed", -1.0, 1.0, &mut sun.yaw_speed);
            ui.slider("Sun yaw", 0.0, glm::two_pi::<f32>(), &mut sun.yaw);
            ui.separator();
            if let Some(shadow_map) = &mut sun.shadow_map {
                for i in 0..CascadedShadowMap::CASCADE_COUNT {
                    ui.slider(format!("Cascade distance #{}", i), 0.0, 500.0, &mut shadow_map.distances[i]);
                }
                ui.separator();
            }
            if DevGui::do_standard_button(ui, "Close") { self.do_sun_window = false; }

            token.end();
        }
    }

    //This is where we upload the Dear Imgui geometry for the current frame
    pub fn resolve_imgui_frame(&mut self, gpu: &mut VulkanGraphicsDevice, renderer: &mut Renderer, context: &mut imgui::Context) {
        //Destroy Dear ImGUI allocations from last dead frame
        if self.do_gui {
            let index_buffers = &mut self.frames[self.current_frame].index_buffers;
            for geo in index_buffers.drain(0..index_buffers.len()) {
                geo.free(gpu);
            }
        }

        let mut index_buffers = Vec::with_capacity(16);
        let mut draw_cmd_lists = Vec::with_capacity(16);
        let mut offsets = Vec::with_capacity(16);

        let imgui_draw_data = context.render();

        let most_recent_dead_frame_idx = self.current_frame.overflowing_sub(Renderer::FRAMES_IN_FLIGHT - 1).0 % Renderer::FRAMES_IN_FLIGHT;
        let most_recent_dead_frame = &self.frames[most_recent_dead_frame_idx];

        let enough_free_space_at_beginning = imgui_draw_data.total_vtx_count as u64 <= most_recent_dead_frame.start_offset;
        let start_offset = if enough_free_space_at_beginning {
            0
        } else {
            most_recent_dead_frame.end_offset
        };
        let mut current_offset = start_offset;

        if imgui_draw_data.total_vtx_count > 0 {
            for list in imgui_draw_data.draw_lists() {
                let vtx_buffer = list.vtx_buffer();
                let mut verts = vec![0.0; vtx_buffer.len() * Self::FLOATS_PER_VERTEX];

                let mut current_vertex = 0;
                for vtx in vtx_buffer.iter() {
                    let idx = current_vertex * Self::FLOATS_PER_VERTEX;
                    verts[idx]     = vtx.pos[0];
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

                offsets.push(current_offset);
                renderer.replace_imgui_vertices(gpu, &verts, current_offset);
                current_offset += vtx_buffer.len() as u64;

                let index_buffer = make_index_buffer(gpu, &inds);
                index_buffers.push(index_buffer);

                let mut cmd_list = Vec::with_capacity(list.commands().count());
                for command in list.commands() { cmd_list.push(command); }
                draw_cmd_lists.push(cmd_list);
            }
        }

        let new_frame = DevGuiFrame {
            offsets,
            start_offset,
            end_offset: current_offset,
            index_buffers,
            draw_cmd_lists
        };
        self.frames[self.current_frame] = new_frame;
    }

    pub unsafe fn record_draw_commands(&mut self, gpu: &mut VulkanGraphicsDevice, command_buffer: vk::CommandBuffer, layout: vk::PipelineLayout) {
        //Record Dear ImGUI drawing commands
        let gui_frame = &self.frames[self.current_frame];
        for i in 0..gui_frame.draw_cmd_lists.len() {
            if i == 0 {
                gpu.device.cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::GRAPHICS, self.pipeline);
            }

            let cmd_list = &gui_frame.draw_cmd_lists[i];
            for cmd in cmd_list {
                match cmd {
                    DrawCmd::Elements {count, cmd_params} => {
                        let i_offset = cmd_params.idx_offset;
                        let i_buffer = gui_frame.index_buffers[i].buffer();

                        let ext_x = cmd_params.clip_rect[2] - cmd_params.clip_rect[0];
                        let ext_y = cmd_params.clip_rect[3] - cmd_params.clip_rect[1];
                        let scissor_rect = vk::Rect2D {
                            offset: vk::Offset2D {
                                x: cmd_params.clip_rect[0] as i32,
                                y: cmd_params.clip_rect[1] as i32
                            },
                            extent: vk::Extent2D {
                                width: ext_x as u32,
                                height: ext_y as u32
                            }
                        };
                        gpu.device.cmd_set_scissor(command_buffer, 0, &[scissor_rect]);

                        let tex_id = cmd_params.texture_id.id() as u32;
                        let pcs = [
                            tex_id.to_le_bytes(),
                            (gui_frame.offsets[i] as u32).to_le_bytes()
                        ].concat();
                        gpu.device.cmd_push_constants(command_buffer, layout, vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT, 0, &pcs);
                        gpu.device.cmd_bind_index_buffer(command_buffer, i_buffer, 0, vk::IndexType::UINT32);
                        gpu.device.cmd_draw_indexed(command_buffer, *count as u32, 1, i_offset as u32, 0, 0);
                    }
                    DrawCmd::ResetRenderState => { println!("DrawCmd::ResetRenderState."); }
                    DrawCmd::RawCallback {..} => { println!("DrawCmd::RawCallback."); }
                }
            }
        }

        self.current_frame = (self.current_frame + 1) % Renderer::FRAMES_IN_FLIGHT;
    }
}

//All data needed to render one Dear ImGUI frame
pub struct DevGuiFrame {
    pub offsets: Vec<u64>,
    pub start_offset: u64,
    pub end_offset: u64,
    pub index_buffers: Vec<GPUBuffer>,
    pub draw_cmd_lists: Vec<Vec<DrawCmd>>
}

impl Default for DevGuiFrame {
    fn default() -> Self {
        DevGuiFrame {
            offsets: Vec::new(),
            start_offset: 0,
            end_offset: 0,
            index_buffers: Vec::new(),
            draw_cmd_lists: Vec::new()
        }
    }
}