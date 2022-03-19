pub struct FreeCam {
    pub position: glm::TVec3<f32>,
    pub orientation: glm::TVec2<f32>,
    pub cursor_captured: bool,
}

impl FreeCam {
    pub fn new(pos: glm::TVec3<f32>) -> Self {
        FreeCam {
            position: pos,
            orientation: glm::zero(),
            cursor_captured: false
        }
    }

    pub fn make_view_matrix(&self) -> glm::TMat4<f32> {
        glm::rotation(-glm::half_pi::<f32>(), &glm::vec3(1.0, 0.0, 0.0)) *
        glm::rotation(self.orientation.y, &glm::vec3(1.0, 0.0, 0.0)) *
        glm::rotation(self.orientation.x, &glm::vec3(0.0, 0.0, 1.0)) *
        glm::translation(&-self.position)
    }
}