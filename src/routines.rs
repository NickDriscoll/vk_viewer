use ozy::render::PositionNormalTangentUvPrimitive;
use ozy::structs::UninterleavedVertexArrays;
use render::{PrimitiveKey};
use render::vkdevice::*;
use crate::*;

//Converts any data structure to a slice of bytes
#[inline]
pub fn struct_to_bytes<'a, T>(structure: &'a T) -> &'a [u8] {
    unsafe { std::slice::from_raw_parts(structure as *const _ as *const u8, size_of::<T>()) }
}

#[inline]
pub fn slice_to_bytes<'a, T>(in_array: &'a [T]) -> &'a [u8] {
    unsafe { core::slice::from_raw_parts(in_array.as_ptr() as *const u8, in_array.len() * size_of::<T>()) }
}

pub fn vec_to_bytes<'a, T>(vec: &Vec<T>) -> &'a [u8] {
    unsafe { core::slice::from_raw_parts(vec.as_ptr() as *const u8, vec.len() * size_of::<T>()) }
}

pub fn quaternion_to_euler(q: &Rotation<Real>) -> glm::TVec3<Real> {
    let mut angles = glm::vec3(0.0, 0.0, 0.0);
    let sinr_cosp = 2.0 * (q.w * q.i + q.j * q.k);
    let cosr_cosp = 1.0 - 2.0 * (q.i * q.i + q.j * q.j);
    angles.x = f32::atan2(sinr_cosp, cosr_cosp);

    // pitch (y-axis rotation)
    let sinp = f32::sqrt(1.0 + 2.0 * (q.w * q.i - q.j * q.k));
    let cosp = f32::sqrt(1.0 - 2.0 * (q.w * q.i - q.j * q.k));
    angles.y = 2.0 * f32::atan2(sinp, cosp) - glm::pi::<f32>() / 2.0;

    // yaw (z-axis rotation)
    let siny_cosp = 2.0 * (q.w * q.k + q.i * q.j);
    let cosy_cosp = 1.0 - 2.0 * (q.j * q.j + q.k * q.k);
    angles.z = f32::atan2(siny_cosp, cosy_cosp);

    angles
}

// pub fn euler_to_quaternion(roll: f32, pitch: f32, yaw: f32) -> Rotation<Real> {
//     let cr = f32::cos(roll * 0.5);
//     let sr = f32::sin(roll * 0.5);
//     let cp = f32::cos(pitch * 0.5);
//     let sp = f32::sin(pitch * 0.5);
//     let cy = f32::cos(yaw * 0.5);
//     let sy = f32::sin(yaw * 0.5);

//     let mut q = Rotation::identity();
    
//     q.w = cr * cp * cy + sr * sp * sy;
//     q.i = sr * cp * cy - cr * sp * sy;
//     q.j = cr * sp * cy + sr * cp * sy;
//     q.k = cr * cp * sy - sr * sp * cy;

//     q
// }

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
        Err(e) => {
            crash_with_error_dialog(&format!("{}\n{}", msg, e));
        }
    }
}

pub fn unix_epoch_ms() -> u128 {
    SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis()
}

pub fn compute_terrain_vertices(spec: &mut TerrainSpec, fixed_seed: bool) -> UninterleavedVertexArrays {
    if !fixed_seed {
        spec.seed = unix_epoch_ms();
    }
    spec.generate_vertices()
}

pub fn regenerate_terrain(
    vk: &mut VulkanGraphicsDevice,
    renderer: &mut Renderer,
    physics_engine: &mut PhysicsEngine,
    terrain_collider_handle: &mut ColliderHandle,
    primitive_key: PrimitiveKey,
    terrain: &mut TerrainSpec
) {
    if let Some(ter) = renderer.get_primitive(primitive_key) {
        let offset = ter.position_offset;
        let verts = compute_terrain_vertices(terrain, terrain.fixed_seed);
        replace_uploaded_vertices(vk, renderer, &verts, offset.into());

        physics_engine.collider_set.remove(*terrain_collider_handle, &mut physics_engine.island_manager, &mut physics_engine.rigid_body_set, false);

        *terrain_collider_handle = physics_engine.make_terrain_collider(&verts.positions, terrain.vertex_width, terrain.vertex_height);
    }
}

pub fn upload_vertex_attributes(vk: &mut VulkanGraphicsDevice, renderer: &mut Renderer, attribs: &UninterleavedVertexArrays) -> VertexFetchOffsets {
    let position_offset = renderer.append_vertex_positions(vk, &attribs.positions);
    let tangent_offset = renderer.append_vertex_tangents(vk, &attribs.tangents);
    let normal_offset = renderer.append_vertex_normals(vk, &attribs.normals);
    let uv_offset = renderer.append_vertex_uvs(vk, &attribs.uvs);

    VertexFetchOffsets {
        position_offset,
        tangent_offset,
        normal_offset,
        uv_offset
    }
}

pub fn upload_primitive_vertices<T: PositionNormalTangentUvPrimitive>(vk: &mut VulkanGraphicsDevice, renderer: &mut Renderer, prim: &T) -> VertexFetchOffsets {
    let position_offset = renderer.append_vertex_positions(vk, prim.vertex_positions());
    let tangent_offset = renderer.append_vertex_tangents(vk, prim.vertex_tangents());
    let normal_offset = renderer.append_vertex_normals(vk, prim.vertex_normals());
    let uv_offset = renderer.append_vertex_uvs(vk, prim.vertex_uvs());

    VertexFetchOffsets {
        position_offset,
        tangent_offset,
        normal_offset,
        uv_offset
    }
}

pub fn replace_uploaded_vertices(vk: &mut VulkanGraphicsDevice, renderer: &mut Renderer, attributes: &UninterleavedVertexArrays, offset: u64) {
    renderer.replace_vertex_positions(vk, &attributes.positions, offset);
    renderer.replace_vertex_tangents(vk, &attributes.tangents, offset);
    renderer.replace_vertex_normals(vk, &attributes.normals, offset);
    renderer.replace_vertex_uvs(vk, &attributes.uvs, offset);
}

pub fn reset_totoro(physics_engine: &mut PhysicsEngine, totoro: &Entity) {
    let handle = totoro.physics_component.rigid_body_handle;
    if let Some(body) = physics_engine.rigid_body_set.get_mut(handle) {
        body.set_linvel(glm::zero(), true);
        body.set_position(Isometry::from_parts(Translation::new(0.0, 0.0, 20.0), *body.rotation()), true);
    }
}

pub fn make_index_buffer(vk: &mut VulkanGraphicsDevice, indices: &[u32]) -> GPUBuffer {
    let index_buffer = GPUBuffer::allocate(
        vk,
        (indices.len() * size_of::<u32>()) as vk::DeviceSize,
        0,
        vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
        MemoryLocation::GpuOnly
    );
    index_buffer.write_buffer(vk, indices);
    index_buffer
}
