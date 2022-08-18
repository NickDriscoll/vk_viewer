use ash::vk;
use imgui::DrawCmd;
use crate::render::Renderer;
use crate::vkutil::*;

pub struct DevGui {
    pub pipeline: vk::Pipeline,
    pub frames: Vec<DevGuiFrame>,
    pub current_frame: usize,
    pub do_main_window: bool,
    pub do_terrain_window: bool
}

impl DevGui {
    pub const FRAMES_IN_FLIGHT: usize = 2;
    pub const FLOATS_PER_VERTEX: usize = 8;

    pub fn new(vk: &mut VulkanAPI, render_pass: vk::RenderPass, pipeline_layout: vk::PipelineLayout) -> Self {
        let mut frames = Vec::with_capacity(Self::FRAMES_IN_FLIGHT);
        for _ in 0..Self::FRAMES_IN_FLIGHT {
            frames.push(DevGuiFrame::default());
        }
        

        let im_shader_stages = {
            let v = load_shader_stage(&vk.device, vk::ShaderStageFlags::VERTEX, "./data/shaders/imgui_vert.spv");
            let f = load_shader_stage(&vk.device, vk::ShaderStageFlags::FRAGMENT, "./data/shaders/imgui_frag.spv");
            vec![v, f]
        };
        let im_info = GraphicsPipelineBuilder::init(pipeline_layout)
            .set_shader_stages(im_shader_stages)
            .set_render_pass(render_pass)
            .set_depth_test(vk::FALSE)           
            .set_cull_mode(vk::CullModeFlags::NONE) 
            .build_info();

        let pipeline = unsafe { GraphicsPipelineBuilder::create_pipelines(vk, &[im_info])[0] };

        DevGui {
            pipeline,
            frames,
            current_frame: 0,
            do_main_window: true,
            do_terrain_window: false
        }
    }

    //This is where we upload the Dear Imgui geometry for the current frame
    pub fn resolve_imgui_frame(&mut self, vk: &mut VulkanAPI, renderer: &mut Renderer, imgui_ui: imgui::Ui) {
        let mut index_buffers = Vec::with_capacity(16);
        let mut cmds = Vec::with_capacity(16);
        let mut offsets = Vec::with_capacity(16);
        let imgui_draw_data = imgui_ui.render();

        let last_frame = &self.frames[self.current_frame.overflowing_sub(1).0 % Self::FRAMES_IN_FLIGHT];

        let enough_free_space_at_beginning = imgui_draw_data.total_vtx_count as u64 <= last_frame.start_offset;
        let start_offset = if enough_free_space_at_beginning {
            0
        } else {
            last_frame.end_offset
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
                renderer.replace_imgui_vertices(vk, &verts, current_offset);
                current_offset += vtx_buffer.len() as u64;

                let index_buffer = make_index_buffer(vk, &inds);
                index_buffers.push(index_buffer);

                let mut cmd_list = Vec::with_capacity(list.commands().count());
                for command in list.commands() { cmd_list.push(command); }
                cmds.push(cmd_list);
            }
        }

        self.frames[self.current_frame] = DevGuiFrame {
            offsets,
            start_offset,
            end_offset: current_offset,
            index_buffers,
            draw_cmd_lists: cmds
        };
    }

    pub unsafe fn record_draw_commands(&mut self, vk: &mut VulkanAPI, layout: vk::PipelineLayout) {
        //Destroy Dear ImGUI allocations from last frame
        {
            let last_frame = self.current_frame.overflowing_sub(1).0 % Self::FRAMES_IN_FLIGHT;
            let geo_count = self.frames[last_frame].index_buffers.len();
            for geo in self.frames[last_frame].index_buffers.drain(0..geo_count) {
                geo.free(vk);
            }
        }

        //Record Dear ImGUI drawing commands
        vk.device.cmd_bind_pipeline(vk.graphics_command_buffer, vk::PipelineBindPoint::GRAPHICS, self.pipeline);
        let gui_frame = &self.frames[self.current_frame];
        for i in 0..gui_frame.draw_cmd_lists.len() {
            let cmd_list = &gui_frame.draw_cmd_lists[i];
            for cmd in cmd_list {
                match cmd {
                    DrawCmd::Elements {count, cmd_params} => {
                        let i_offset = cmd_params.idx_offset;
                        let i_buffer = gui_frame.index_buffers[i].backing_buffer();

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
                        vk.device.cmd_set_scissor(vk.graphics_command_buffer, 0, &[scissor_rect]);

                        let tex_id = cmd_params.texture_id.id() as u32;
                        let pcs = [
                            tex_id.to_le_bytes(),
                            (gui_frame.offsets[i] as u32).to_le_bytes()
                        ].concat();
                        vk.device.cmd_push_constants(vk.graphics_command_buffer, layout, vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT, 0, &pcs);
                        vk.device.cmd_bind_index_buffer(vk.graphics_command_buffer, i_buffer, 0, vk::IndexType::UINT32);
                        vk.device.cmd_draw_indexed(vk.graphics_command_buffer, *count as u32, 1, i_offset as u32, 0, 0);
                    }
                    DrawCmd::ResetRenderState => { println!("DrawCmd::ResetRenderState."); }
                    DrawCmd::RawCallback {..} => { println!("DrawCmd::RawCallback."); }
                }
            }
        }
        self.current_frame = (self.current_frame + 1) % Self::FRAMES_IN_FLIGHT;
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