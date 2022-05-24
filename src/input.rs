use imgui::Io;
use ozy::structs::FrameTimer;
use sdl2::controller::GameController;
use sdl2::{EventPump, GameControllerSubsystem};
use crate::*;

pub struct InputSystem {
    event_pump: EventPump,
    controller_subsystem: GameControllerSubsystem,
    controllers: OptionVec<GameController>
}

//The output of the input system
pub struct InputOutput {
    pub movement_multiplier: f32,
    pub movement_vector: glm::TVec3<f32>,
    pub orientation_delta: glm::TVec2<f32>,
    pub scroll_amount: f32,
    pub framerate: f32,
    pub regen_terrain: bool,
    pub resize_window: bool,
    pub cursor_capture_toggle: bool,
    pub gui_toggle: bool
}

enum UserInput {
    Output(InputOutput),
    ExitProgram
}

fn do_input(
    input_system: &mut InputSystem,
    timer: &FrameTimer,
    imgui_io: &mut Io
) -> UserInput {
    use sdl2::controller::Button;
    use sdl2::event::WindowEvent;
    use sdl2::keyboard::{Scancode};
    use sdl2::mouse::MouseButton;

    //Abstracted input variables
    let mut movement_multiplier = 5.0f32;
    let mut movement_vector: glm::TVec3<f32> = glm::zero();
    let mut orientation_delta: glm::TVec2<f32> = glm::zero();
    let mut scroll_amount = 0.0;
    let framerate;
    let mut regen_terrain = false;
    let mut resize_window = false;
    let mut cursor_capture_toggle = false;
    let mut gui_toggle = false;

    //Sync controller array with how many controllers are actually connected
    for i in 0..input_system.controllers.len() {
        match &mut input_system.controllers.get_mut_element(i) {
            None => {
                if i < unwrap_result(input_system.controller_subsystem.num_joysticks(), "Error getting number of controllers") as usize {
                    let controller = unwrap_result(input_system.controller_subsystem.open(i as u32), "Error opening controller");
                    input_system.controllers.replace(i, controller);
                }
            }
            Some(controller) => {
                if !controller.attached() {
                    input_system.controllers.delete(i);
                }
            }
        }
    }

    imgui_io.delta_time = timer.delta_time;
    
    //Pump event queue
    for event in input_system.event_pump.poll_iter() {
        match event {
            Event::Quit{..} => { return UserInput::ExitProgram; }
            Event::Window { win_event, .. } => {
                match win_event {
                    WindowEvent::Resized(_, _) => { resize_window = true; }
                    _ => {}
                }
            }
            Event::KeyDown { scancode: Some(sc), repeat: false, .. } => {
                match sc {
                    Scancode::Escape => { gui_toggle = true; }
                    Scancode::R => { regen_terrain = true; }
                    _ => {}
                }
            }
            Event::MouseButtonUp { mouse_btn, ..} => {
                match mouse_btn {
                    MouseButton::Right => { cursor_capture_toggle = true; }
                    _ => {}
                }
            }
            Event::MouseMotion { xrel, yrel, .. } => {
                const DAMPENING: f32 = 0.25 / 360.0;
                orientation_delta += glm::vec2(DAMPENING * xrel as f32, DAMPENING * yrel as f32);
            }
            Event::MouseWheel { x, y, .. } => {
                imgui_io.mouse_wheel_h = x as f32;
                imgui_io.mouse_wheel = y as f32;
                scroll_amount = imgui_io.mouse_wheel;
            }
            _ => {  }
        }
    }

    let keyboard_state = input_system.event_pump.keyboard_state();
    let mouse_state = input_system.event_pump.mouse_state();
    imgui_io.mouse_down = [mouse_state.left(), mouse_state.right(), mouse_state.middle(), mouse_state.x1(), mouse_state.x2()];
    imgui_io.mouse_pos[0] = mouse_state.x() as f32;
    imgui_io.mouse_pos[1] = mouse_state.y() as f32;

    if let Some(controller) = &mut input_system.controllers.get_mut_element(0) {
        use sdl2::controller::{Axis};

        fn get_normalized_axis(controller: &GameController, axis: Axis) -> f32 {
            controller.axis(axis) as f32 / i16::MAX as f32
        }

        if controller.button(Button::LeftShoulder) {
            movement_vector += glm::vec3(0.0, 0.0, -1.0);                    
        }

        if controller.button(Button::RightShoulder) {
            movement_vector += glm::vec3(0.0, 0.0, 1.0);                    
        }

        if controller.button(Button::Y) {
            regen_terrain = true;
            if let Err(e) = controller.set_rumble(0xFFFF, 0xFFFF, 50) {
                println!("{}", e);
            }
        }

        let left_trigger = get_normalized_axis(&controller, Axis::TriggerLeft);
        movement_multiplier *= 4.0 * left_trigger + 1.0;

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

        movement_vector += &left_joy_vector;
        orientation_delta += 4.0 * timer.delta_time * glm::vec2(right_joy_vector.x, -right_joy_vector.y);
    }

    if keyboard_state.is_scancode_pressed(Scancode::LShift) {
        movement_multiplier *= 15.0;
    }
    if keyboard_state.is_scancode_pressed(Scancode::LCtrl) {
        movement_multiplier *= 0.25;
    }
    if keyboard_state.is_scancode_pressed(Scancode::W) {
        movement_vector += glm::vec3(0.0, 1.0, 0.0);
    }
    if keyboard_state.is_scancode_pressed(Scancode::A) {
        movement_vector += glm::vec3(-1.0, 0.0, 0.0);
    }
    if keyboard_state.is_scancode_pressed(Scancode::S) {
        movement_vector += glm::vec3(0.0, -1.0, 0.0);
    }
    if keyboard_state.is_scancode_pressed(Scancode::D) {
        movement_vector += glm::vec3(1.0, 0.0, 0.0);
    }
    if keyboard_state.is_scancode_pressed(Scancode::Q) {
        movement_vector += glm::vec3(0.0, 0.0, -1.0);
    }
    if keyboard_state.is_scancode_pressed(Scancode::E) {
        movement_vector += glm::vec3(0.0, 0.0, 1.0);
    }

    framerate = imgui_io.framerate;
    movement_vector *= movement_multiplier;

    UserInput::Output(InputOutput {
        movement_multiplier,
        movement_vector,
        orientation_delta,
        scroll_amount,
        framerate,
        regen_terrain,
        resize_window,
        cursor_capture_toggle,
        gui_toggle
    })
}