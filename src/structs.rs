use crate::*;

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
    pub noise_parameters: Vec<NoiseParameters>,
    pub amplitude: f64,
    pub exponent: f64,
    pub seed: u128
}

impl TerrainSpec {
    pub fn generate_vertices(&self) -> Vec<f32> {
        use noise::Seedable;
    
        let simplex_generator = noise::OpenSimplex::new().set_seed(self.seed as u32);
        ozy::prims::perturbed_plane_vertex_buffer(self.vertex_width, self.vertex_height, 15.0, move |x, y| {
            use noise::NoiseFn;
    
            let mut z = 0.0;
    
            //Apply each level of noise with the appropriate offset, frequency, and amplitude
            let mut offset = 0.0;
            for parameters in self.noise_parameters.iter() {
                let xi = offset + x * parameters.frequency;
                let yi = offset + y * parameters.frequency;
                z += parameters.amplitude * simplex_generator.get([xi, yi]);
                offset += 50.0;
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
