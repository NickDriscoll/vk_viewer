use crate::*;

//Converts any data structure to a slice of bytes
pub fn struct_to_bytes<'a, T>(structure: &'a T) -> &'a [u8] {
    let p = structure as *const _ as *const u8;
    let size = size_of::<T>();
    unsafe { std::slice::from_raw_parts(p, size) }
}

pub fn crash_with_error_dialog(message: &str) -> ! {
    crash_with_error_dialog_titled("Oops...", message);
}

pub fn crash_with_error_dialog_titled(title: &str, message: &str) -> ! {
    tfd::message_box_ok(title, &message.replace("'", ""), tfd::MessageBoxIcon::Error);
    panic!("{}", message);
}

 pub fn unwrap_result<T, E: Display>(res: Result<T, E>, msg: &str) -> T {
    match res {
        Ok(t) => { t }
        Err(_) => {
            crash_with_error_dialog(&format!("{}", msg));
        }
    }
}

pub fn unix_epoch_ms() -> u128 {
    SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis()
}

pub fn compute_terrain_vertices(spec: &mut TerrainSpec, fixed_seed: bool, scale: f32) -> Vec<f32> {
    if !fixed_seed {
        spec.seed = unix_epoch_ms();
    }
    spec.generate_vertices(scale)
}

pub fn regenerate_terrain(
    vk: &mut VulkanAPI,
    renderer: &mut Renderer,
    physics_engine: &mut PhysicsEngine,
    terrain_collider_handle: &mut ColliderHandle,
    terrain_model_idx: usize,
    terrain: &mut TerrainSpec,
    terrain_vertex_width: usize,
    terrain_fixed_seed: bool,
    terrain_generation_scale: f32
) {
    if let Some(ter) = renderer.get_model(terrain_model_idx) {
        let offset = ter.position_offset;
        let verts = compute_terrain_vertices(terrain, terrain_fixed_seed, terrain_generation_scale);
        replace_uploaded_uninterleaved_vertices(vk, renderer, &verts, offset.into());

        physics_engine.collider_set.remove(*terrain_collider_handle, &mut physics_engine.island_manager, &mut physics_engine.rigid_body_set, false);

        *terrain_collider_handle = physics_engine.make_terrain_collider(&verts, terrain_vertex_width);
    }
}