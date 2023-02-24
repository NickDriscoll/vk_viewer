use ozy::structs::UninterleavedVertexArrays;
use slotmap::new_key_type;

use crate::{*, render::{ModelKey}, physics::PhysicsComponent, input::UserInput};

pub struct Camera {
    pub position: glm::TVec3<f32>,
    pub orientation: glm::TVec2<f32>,
    pub fov: f32,
    pub near_distance: f32,
    pub far_distance: f32,
    forward: glm::TVec3<f32>,
    pub lookat_dist: f32,
    pub focused_entity: Option<EntityKey>,
    pub last_view_from_world: glm::TMat4<f32>
}

impl Camera {
    pub const FREECAM_SPEED: f32 = 3.0;

    pub fn new(pos: glm::TVec3<f32>) -> Self {
        Camera {
            position: pos,
            orientation: glm::zero(),
            fov: glm::half_pi(),
            near_distance: 0.1,
            far_distance: 1000.0,
            forward: glm::vec3(0.0, 1.0, 0.0),
            lookat_dist: 7.5,
            focused_entity: None,
            last_view_from_world: glm::identity()
        }
    }

    pub fn look_direction(&self) -> glm::TVec3<f32> {
        self.forward
    }

    pub fn view_matrix() {

    }

    pub fn update(&mut self, simulation_state: &SimulationSOA, physics_engine: &PhysicsEngine, renderer: &mut Renderer, user_input: &UserInput, delta_time: f32) -> glm::TMat4<f32> {
        let view_movement_vector = glm::mat4(
            1.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, -1.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 1.0
        ) * glm::vec3_to_vec4(&user_input.movement_vector);

        let matrix = match self.focused_entity {
            Some(key) => {
                match simulation_state.entities.get(key) {
                    Some(prop) => {
                        let min = 3.0;
                        let max = 200.0;
                        self.lookat_dist -= 0.1 * self.lookat_dist * user_input.scroll_amount;
                        self.lookat_dist = f32::clamp(self.lookat_dist, min, max);
                        
                        let lookat = glm::look_at(&self.position, &glm::zero(), &glm::vec3(0.0, 0.0, 1.0));
                        let world_space_offset = glm::affine_inverse(lookat) * glm::vec4(-user_input.orientation_delta.x, user_input.orientation_delta.y, 0.0, 0.0);
            
                        self.position += self.lookat_dist * glm::vec4_to_vec3(&world_space_offset);
                        let camera_pos = glm::normalize(&self.position);
                        self.position = self.lookat_dist * camera_pos;
                        
                        let min = -0.95;
                        let max = 0.95;
                        let lookat_dot = glm::dot(&camera_pos, &glm::vec3(0.0, 0.0, 1.0));
                        if lookat_dot > max {
                            let rotation_vector = -glm::cross(&camera_pos, &glm::vec3(0.0, 0.0, 1.0));
                            let current_angle = f32::acos(lookat_dot);
                            let amount = f32::acos(max) - current_angle;
            
                            let new_pos = glm::rotation(amount, &rotation_vector) * glm::vec3_to_vec4(&self.position);
                            self.position = glm::vec4_to_vec3(&new_pos);
                        } else if lookat_dot < min {
                            let rotation_vector = -glm::cross(&camera_pos, &glm::vec3(0.0, 0.0, 1.0));
                            let current_angle = f32::acos(lookat_dot);
                            let amount = f32::acos(min) - current_angle;
            
                            let new_pos = glm::rotation(amount, &rotation_vector) * glm::vec3_to_vec4(&(self.position));                
                            self.position = glm::vec4_to_vec3(&new_pos);
                        }

                        let lookat_target = match physics_engine.rigid_body_set.get(prop.physics_component.rigid_body_handle) {
                            Some(body) => {
                                body.translation()
                            }
                            None => {
                                crash_with_error_dialog("All entities should have a rigid body component");
                            }
                        };
            
                        let pos = self.position + lookat_target;
                        let m = glm::look_at(&pos, &lookat_target, &glm::vec3(0.0, 0.0, 1.0));
                        renderer.uniform_data.camera_position = glm::vec4(pos.x, pos.y, pos.z, 1.0);
                        m
                    }
                    None => {
                        //Freecam update
                        let delta_pos = Self::FREECAM_SPEED * glm::affine_inverse(self.last_view_from_world) * view_movement_vector * delta_time;
                        self.position += glm::vec4_to_vec3(&delta_pos);
                        self.orientation += user_input.orientation_delta;
        
                        self.orientation.y = self.orientation.y.clamp(-glm::half_pi::<f32>(), glm::half_pi::<f32>());
                        renderer.uniform_data.camera_position = glm::vec4(self.position.x, self.position.y, self.position.z, 1.0);
                        glm::rotation(-glm::half_pi::<f32>(), &glm::vec3(1.0, 0.0, 0.0)) *
                        glm::rotation(self.orientation.y, &glm::vec3(1.0, 0.0, 0.0)) *
                        glm::rotation(self.orientation.x, &glm::vec3(0.0, 0.0, 1.0)) *
                        glm::translation(&-self.position)
                    }
                }
            }
            None => {
                //Freecam update
                let delta_pos = Self::FREECAM_SPEED * glm::affine_inverse(self.last_view_from_world) * view_movement_vector * delta_time;
                self.position += glm::vec4_to_vec3(&delta_pos);
                self.orientation += user_input.orientation_delta;

                self.orientation.y = self.orientation.y.clamp(-glm::half_pi::<f32>(), glm::half_pi::<f32>());
                renderer.uniform_data.camera_position = glm::vec4(self.position.x, self.position.y, self.position.z, 1.0);
                glm::rotation(-glm::half_pi::<f32>(), &glm::vec3(1.0, 0.0, 0.0)) *
                glm::rotation(self.orientation.y, &glm::vec3(1.0, 0.0, 0.0)) *
                glm::rotation(self.orientation.x, &glm::vec3(0.0, 0.0, 1.0)) *
                glm::translation(&-self.position)
            }
        };
        self.last_view_from_world = matrix;

        let dir = glm::vec3(0.0, 0.0, -1.0);
        self.forward = glm::vec4_to_vec3(&(glm::affine_inverse(matrix) * glm::vec3_to_vec4(&dir)));

        matrix
    }
}

#[derive(Default)]
pub struct NoiseParameters {
    pub amplitude: f64,
    pub frequency: f64
}

pub struct TerrainSpec {
    pub vertex_width: usize,
    pub vertex_height: usize,
    pub octaves: u64,
    pub lacunarity: f64,
    pub gain: f64,
    pub amplitude: f64,
    pub exponent: f64,
    pub scale: f32,
    pub seed: u128,
    pub interactive_generation: bool,
    pub fixed_seed: bool
}

impl TerrainSpec {
    pub fn generate_vertices(&self) -> UninterleavedVertexArrays {
        use noise::Seedable;
    
        let simplex_generator = noise::OpenSimplex::new().set_seed(self.seed as u32);
        ozy::prims::perturbed_plane_vertex_buffer(self.vertex_width, self.vertex_height, self.scale, move |x, y| {
            use noise::NoiseFn;
    
            let mut z = 0.0;
            
            let mut amplitude = 2.0;
            let mut frequency = 0.15;
            let mut offset = 0.0;
            for _ in 0..self.octaves {
                let xi = offset + x * frequency;
                let yi = offset + y * frequency;
                z += amplitude * simplex_generator.get([xi, yi]);
                offset += 50.0;

                amplitude *= self.gain;
                frequency *= self.lacunarity;
            }
    
            //Apply exponent to flatten. Branch is for exponentiating a negative
            z = if z < 0.0 {
                -f64::powf(-z, self.exponent)
            } else {
                f64::powf(z, self.exponent)
            };
    
            //Apply global amplitude
            z *= self.amplitude;
    
            z
        })

    }
}

impl Default for TerrainSpec {
    fn default() -> Self {
        TerrainSpec {
            vertex_width: 128,
            vertex_height: 128,
            octaves: 8,
            lacunarity: 1.75,
            gain: 0.5,
            amplitude: 2.0,
            exponent: 2.2,
            scale: 1.0,
            seed: 0,
            interactive_generation: false,
            fixed_seed: false            
        }
    }
}

pub struct Entity {
    pub name: String,
    pub model: ModelKey,
    pub physics_component: PhysicsComponent
}

impl Entity {
    pub fn new(name: String, model: ModelKey, physics_engine: &mut PhysicsEngine) -> Self {
        let rigid_body = RigidBodyBuilder::dynamic().ccd_enabled(true).build();
        let rigid_body_handle = physics_engine.rigid_body_set.insert(rigid_body);
        let physics_component = PhysicsComponent {
            rigid_body_handle,
            collider_handle: None,
            rigid_body_offset: glm::zero(),
            scale: 1.0
        };
        Entity {
            name,
            model,
            physics_component
        }
    }

    pub fn set_position(&mut self, pos: glm::TVec3<f32>, physics_engine: &mut PhysicsEngine) {
        if let Some(body) = physics_engine.rigid_body_set.get_mut(self.physics_component.rigid_body_handle) {
            body.set_translation(pos, true);
        }
    }

    pub fn set_physics_component(mut self, component: PhysicsComponent) -> Self {
        self.physics_component = component;
        self
    }
}

new_key_type! { pub struct ModelMatrixKey; }
new_key_type! { pub struct ModelIndexKey; }
new_key_type! { pub struct EntityKey; }
pub struct SimulationSOA {
    pub entities: DenseSlotMap<EntityKey, Entity>,
    pub timescale: f32
}

impl SimulationSOA {
    pub fn new() -> Self {
        SimulationSOA{
            entities: DenseSlotMap::with_key(),
            timescale: 1.0
        }
    }
}