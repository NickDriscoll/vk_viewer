use ozy::structs::UninterleavedVertexArrays;
use slotmap::new_key_type;

use crate::{*, render::{ModelKey}, physics::PhysicsComponent};

pub struct Camera {
    pub position: glm::TVec3<f32>,
    pub orientation: glm::TVec2<f32>,
    pub fov: f32
}

impl Camera {
    pub fn new(pos: glm::TVec3<f32>) -> Self {
        Camera {
            position: pos,
            orientation: glm::zero(),
            fov: glm::half_pi()
        }
    }

    pub fn make_view_matrix(&self) -> glm::TMat4<f32> {
        glm::rotation(-glm::half_pi::<f32>(), &glm::vec3(1.0, 0.0, 0.0)) *
        glm::rotation(self.orientation.y, &glm::vec3(1.0, 0.0, 0.0)) *
        glm::rotation(self.orientation.x, &glm::vec3(0.0, 0.0, 1.0)) *
        glm::translation(&-self.position)
    }

    pub fn look_direction(&self) -> glm::TVec3<f32> {
        let view_mat = self.make_view_matrix();
        let dir = glm::vec3(0.0, 0.0, -1.0);
        glm::vec4_to_vec3(&(glm::affine_inverse(view_mat) * glm::vec3_to_vec4(&dir)))
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
    pub model_matrices: DenseSlotMap<ModelMatrixKey, glm::TMat4<f32>>,
    pub model_indices: DenseSlotMap<ModelIndexKey, Vec<usize>>,
    pub entities: DenseSlotMap<EntityKey, Entity>,
    pub timescale: f32
}

impl SimulationSOA {
    pub fn new() -> Self {
        SimulationSOA{
            model_matrices: DenseSlotMap::with_key(),
            model_indices: DenseSlotMap::with_key(),
            entities: DenseSlotMap::with_key(),
            timescale: 1.0
        }
    }
}