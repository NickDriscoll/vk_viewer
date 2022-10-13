use ozy::structs::UninterleavedVertexArrays;
use slotmap::new_key_type;

use crate::{*, render::{Model, ModelKey}};

pub struct Camera {
    pub position: glm::TVec3<f32>,
    pub orientation: glm::TVec2<f32>,
}

impl Camera {
    pub fn new(pos: glm::TVec3<f32>) -> Self {
        Camera {
            position: pos,
            orientation: glm::zero(),
        }
    }

    pub fn make_view_matrix(&self) -> glm::TMat4<f32> {
        glm::rotation(-glm::half_pi::<f32>(), &glm::vec3(1.0, 0.0, 0.0)) *
        glm::rotation(self.orientation.y, &glm::vec3(1.0, 0.0, 0.0)) *
        glm::rotation(self.orientation.x, &glm::vec3(0.0, 0.0, 1.0)) *
        glm::translation(&-self.position)
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
    pub seed: u128,
    pub interactive_generation: bool,
    pub fixed_seed: bool
}

impl TerrainSpec {
    pub fn generate_vertices(&self, scale: f32) -> UninterleavedVertexArrays {
        use noise::Seedable;
    
        let simplex_generator = noise::OpenSimplex::new().set_seed(self.seed as u32);
        ozy::prims::perturbed_plane_vertex_buffer(self.vertex_width, self.vertex_height, scale, move |x, y| {
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
            seed: 0,
            interactive_generation: false,
            fixed_seed: false            
        }
    }
}

pub struct PhysicsProp {
    pub rigid_body_handle: RigidBodyHandle,
    pub collider_handle: ColliderHandle
}

#[derive(Clone)]
pub struct StaticProp {
    pub name: String,
    pub model: ModelKey,
    pub position: glm::TVec3<f32>,
    pub pitch: f32,
    pub yaw: f32,
    pub roll: f32
}

new_key_type! { pub struct ModelMatrixKey; }
new_key_type! { pub struct ModelIndexKey; }
pub struct SimulationSOA {
    pub model_matrices: DenseSlotMap<ModelMatrixKey, glm::TMat4<f32>>,
    pub model_indices: DenseSlotMap<ModelIndexKey, Vec<usize>>,

}