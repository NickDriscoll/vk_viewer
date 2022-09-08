use imgui::Io;
use ozy::structs::FrameTimer;
use sdl2::controller::GameController;
use sdl2::{EventPump, GameControllerSubsystem};
use crate::*;

#[derive(Default)]
//The output of the input system
pub struct InputOutput {
    pub movement_multiplier: f32,
    pub movement_vector: glm::TVec3<f32>,
    pub orientation_delta: glm::TVec2<f32>,
    pub scroll_amount: f32,
    pub framerate: f32,
    pub regen_terrain: bool,
    pub reset_totoro: bool,
    pub resize_window: bool,
    pub gui_toggle: bool,
    pub spawn_totoro_prop: bool
}

pub enum UserInput {
    Output(InputOutput),
    ExitProgram
}
pub struct InputSystem {
    pub event_pump: EventPump,
    pub controller_subsystem: GameControllerSubsystem,
    pub controllers: [Option<GameController>; 4],
    pub cursor_captured: bool
}

impl InputSystem {
    pub fn init(sdl_ctxt: &sdl2::Sdl) -> Self {
        let event_pump = unwrap_result(sdl_ctxt.event_pump(), "Error initializing SDL event pump");
        let controller_subsystem = unwrap_result(sdl_ctxt.game_controller(), "Error initializing SDL controller subsystem");
        let controllers = [None, None, None, None];

        InputSystem {
            event_pump,
            controller_subsystem,
            controllers,
            cursor_captured: false
        }
    }

    pub fn do_thing(&mut self, timer: &FrameTimer, imgui_io: &mut Io) -> UserInput {
        use sdl2::controller::Button;
        use sdl2::event::WindowEvent;
        use sdl2::keyboard::{Scancode};
        use sdl2::mouse::MouseButton;

        let mut out = InputOutput::default();

        //Abstracted input variables
        out.movement_multiplier = 5.0f32;
        out.movement_vector = glm::zero();
        out.orientation_delta = glm::zero();
        out.scroll_amount = 0.0;
        out.regen_terrain = false;
        out.reset_totoro = false;
        out.resize_window = false;
        out.gui_toggle = false;

        //Sync controller array with how many controllers are actually connected
        for i in 0..self.controllers.len() {
            match &mut self.controllers[i] {
                None => {
                    if i < unwrap_result(self.controller_subsystem.num_joysticks(), "Error getting number of controllers") as usize {
                        let controller = unwrap_result(self.controller_subsystem.open(i as u32), "Error opening controller");
                        self.controllers[i] = Some(controller);
                    }
                }
                Some(controller) => {
                    if !controller.attached() {
                        self.controllers[i] = None;
                    }
                }
            }
        }

        imgui_io.delta_time = timer.delta_time;
        
        //Pump event queue
        while let Some(event) = self.event_pump.poll_event() {
            match event {
                Event::Quit{..} => { return UserInput::ExitProgram; }
                Event::Window { win_event, .. } => {
                    match win_event {
                        WindowEvent::Resized(_, _) => { out.resize_window = true; }
                        _ => {}
                    }
                }
                Event::KeyDown { scancode: Some(sc), repeat: false, .. } => {
                    match sc {
                        Scancode::Escape => { out.gui_toggle = true; }
                        Scancode::Space => { out.spawn_totoro_prop = true; }
                        Scancode::R => {
                            out.regen_terrain = true;
                        }
                        Scancode::T => {
                            out.reset_totoro = true;
                        }
                        _ => {}
                    }
                }
                Event::MouseButtonUp { mouse_btn, .. } => {
                    match mouse_btn {
                        MouseButton::Right => { self.cursor_captured = !self.cursor_captured; }
                        _ => {}
                    }
                }
                Event::MouseMotion { xrel, yrel, .. } => {
                    const DAMPENING: f32 = 0.25 / 360.0;
                    if self.cursor_captured {
                        out.orientation_delta += glm::vec2(DAMPENING * xrel as f32, DAMPENING * yrel as f32);
                    }
                }
                Event::MouseWheel { x, y, .. } => {
                    imgui_io.mouse_wheel_h = x as f32;
                    imgui_io.mouse_wheel = y as f32;
                    out.scroll_amount = imgui_io.mouse_wheel;
                }
                _ => {  }
            }
        }

        let keyboard_state = self.event_pump.keyboard_state();
        let mouse_state = self.event_pump.mouse_state();
        imgui_io.mouse_down = [mouse_state.left(), mouse_state.right(), mouse_state.middle(), mouse_state.x1(), mouse_state.x2()];
        imgui_io.mouse_pos[0] = mouse_state.x() as f32;
        imgui_io.mouse_pos[1] = mouse_state.y() as f32;

        const MAX_MOVEMENT_MULTIPLIER: f32 = 15.0;

        if let Some(controller) = &mut self.controllers[0] {
            use sdl2::controller::{Axis};

            fn get_normalized_axis(controller: &GameController, axis: Axis) -> f32 {
                controller.axis(axis) as f32 / i16::MAX as f32
            }

            if controller.button(Button::LeftShoulder) {
                out.movement_vector += glm::vec3(0.0, 0.0, -1.0);                    
            }

            if controller.button(Button::RightShoulder) {
                out.movement_vector += glm::vec3(0.0, 0.0, 1.0);                    
            }

            if controller.button(Button::Y) {
                out.regen_terrain = true;
                if let Err(e) = controller.set_rumble(0xFFFF, 0xFFFF, 50) {
                    println!("{}", e);
                }
            }

            let left_trigger = get_normalized_axis(&controller, Axis::TriggerLeft);
            out.movement_multiplier *= (MAX_MOVEMENT_MULTIPLIER - 1.0) * left_trigger + 1.0;

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

            out.movement_vector += &left_joy_vector;
            out.orientation_delta += 4.0 * timer.delta_time * glm::vec2(right_joy_vector.x, -right_joy_vector.y);
        }

        if keyboard_state.is_scancode_pressed(Scancode::LShift) || keyboard_state.is_scancode_pressed(Scancode::RShift) {
            out.movement_multiplier *= MAX_MOVEMENT_MULTIPLIER;
        }
        if keyboard_state.is_scancode_pressed(Scancode::LCtrl) || keyboard_state.is_scancode_pressed(Scancode::RCtrl) {
            out.movement_multiplier *= 0.25;
        }
        if keyboard_state.is_scancode_pressed(Scancode::W) {
            out.movement_vector += glm::vec3(0.0, 1.0, 0.0);
        }
        if keyboard_state.is_scancode_pressed(Scancode::A) {
            out.movement_vector += glm::vec3(-1.0, 0.0, 0.0);
        }
        if keyboard_state.is_scancode_pressed(Scancode::S) {
            out.movement_vector += glm::vec3(0.0, -1.0, 0.0);
        }
        if keyboard_state.is_scancode_pressed(Scancode::D) {
            out.movement_vector += glm::vec3(1.0, 0.0, 0.0);
        }
        if keyboard_state.is_scancode_pressed(Scancode::Q) {
            out.movement_vector += glm::vec3(0.0, 0.0, -1.0);
        }
        if keyboard_state.is_scancode_pressed(Scancode::E) {
            out.movement_vector += glm::vec3(0.0, 0.0, 1.0);
        }

        out.framerate = imgui_io.framerate;
        out.movement_vector *= out.movement_multiplier;

        UserInput::Output(out)
    }
}
