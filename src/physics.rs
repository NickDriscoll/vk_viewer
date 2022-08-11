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
}