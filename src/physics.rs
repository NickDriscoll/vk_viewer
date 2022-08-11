use rapier3d::prelude::*;

pub struct PhysicsEngine {
    pub gravity: glm::TVec3<f32>,
    pub integration_parameters: IntegrationParameters,
    pub rigid_body_set: RigidBodySet,
    pub collider_set: ColliderSet,
    pub island_manager: IslandManager,
    pub broad_phase: BroadPhase,
    pub narrow_phase: NarrowPhase,
    pub impulse_joint_set: ImpulseJointSet,
    pub multibody_joint_set: MultibodyJointSet,
    pub ccd_solver: CCDSolver,
    pub physics_hooks: (),
    pub event_handler: (),
    pub physics_pipeline: PhysicsPipeline
}

impl PhysicsEngine {
    pub fn new() -> Self {
        let gravity = vector![0.0, 0.0, -9.81];
        let integration_parameters = IntegrationParameters::default();
        let rigid_body_set = RigidBodySet::new();
        let collider_set = ColliderSet::new();
        let island_manager = IslandManager::new();
        let broad_phase = BroadPhase::new();
        let narrow_phase = NarrowPhase::new();
        let impulse_joint_set = ImpulseJointSet::new();
        let multibody_joint_set = MultibodyJointSet::new();
        let ccd_solver = CCDSolver::new();
        let physics_pipeline = PhysicsPipeline::new();

        PhysicsEngine {
            gravity,
            integration_parameters,
            rigid_body_set,
            collider_set,
            island_manager,
            broad_phase,
            narrow_phase,
            impulse_joint_set,
            multibody_joint_set,
            ccd_solver,
            physics_hooks: (),
            event_handler: (),
            physics_pipeline
        }
    }

    pub fn step(&mut self) {
        self.physics_pipeline.step(
            &self.gravity,
            &self.integration_parameters,
            &mut self.island_manager,
            &mut self.broad_phase,
            &mut self.narrow_phase,
            &mut self.rigid_body_set,
            &mut self.collider_set,
            &mut self.impulse_joint_set,
            &mut self.multibody_joint_set,
            &mut self.ccd_solver,
            &self.physics_hooks,
            &self.event_handler
        );
    }

    pub fn make_terrain_collider(&mut self, verts: &[f32], width: usize, old_handle: Option<ColliderHandle>) -> ColliderHandle {
        let mut i_copy = ozy::prims::plane_index_buffer(width, width);

        let mut vs = Vec::with_capacity(verts.len() / 15 * 3);
        for i in (0..verts.len()).step_by(15) {
            vs.push(Point::new(verts[i], verts[i + 1], verts[i + 2]));
        }

        let inds = unsafe {
            Vec::from_raw_parts(
                i_copy.as_mut_ptr() as *mut [u32; 3],
                i_copy.len() / 3,
                i_copy.capacity() / 3
            )
        };
        std::mem::forget(i_copy);

        if let Some(handle) = old_handle {
            self.collider_set.remove(handle, &mut self.island_manager, &mut self.rigid_body_set, false);
        }

        let terrain_collider = ColliderBuilder::trimesh(vs, inds);
        self.collider_set.insert(terrain_collider)
    }
}