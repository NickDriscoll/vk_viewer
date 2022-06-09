use imgui::DrawCmd;
use crate::vkutil::VirtualGeometry;

pub struct Imgui {
    pub frames: Vec<ImguiFrame>,
    pub current_frame: usize,
    pub do_gui: bool,
    pub do_terrain_window: bool
}

impl Imgui {
    pub const FRAMES_IN_FLIGHT: usize = 2;

    pub fn new() -> Self {
        let mut frames = Vec::with_capacity(Self::FRAMES_IN_FLIGHT);
        for i in 0..Self::FRAMES_IN_FLIGHT {
            frames.push(ImguiFrame::default());
        }
        Imgui {
            frames,
            current_frame: 0,
            do_gui: true,
            do_terrain_window: false
        }
    }
}

pub struct ImguiFrame {
    pub geometries: Vec<VirtualGeometry>,
    pub draw_cmd_lists: Vec<Vec<DrawCmd>>
}

impl Default for ImguiFrame {
    fn default() -> Self {
        ImguiFrame {
            geometries: Vec::new(),
            draw_cmd_lists: Vec::new()
        }
    }
}