use imgui::DrawCmd;
use crate::vkutil::VirtualGeometry;

pub struct DevGui {
    pub frames: Vec<DevGuiFrame>,
    pub current_frame: usize,
    pub do_main_window: bool,
    pub do_terrain_window: bool
}

impl DevGui {
    pub const FRAMES_IN_FLIGHT: usize = 2;

    pub fn new() -> Self {
        let mut frames = Vec::with_capacity(Self::FRAMES_IN_FLIGHT);
        for _ in 0..Self::FRAMES_IN_FLIGHT {
            frames.push(DevGuiFrame::default());
        }
        DevGui {
            frames,
            current_frame: 0,
            do_main_window: true,
            do_terrain_window: false
        }
    }
}

pub struct DevGuiFrame {
    pub offsets: Vec<u64>,
    pub start_offset: u64,
    pub end_offset: u64,
    pub geometries: Vec<VirtualGeometry>,
    pub draw_cmd_lists: Vec<Vec<DrawCmd>>
}

impl Default for DevGuiFrame {
    fn default() -> Self {
        DevGuiFrame {
            offsets: Vec::new(),
            start_offset: 0,
            end_offset: 0,
            geometries: Vec::new(),
            draw_cmd_lists: Vec::new()
        }
    }
}