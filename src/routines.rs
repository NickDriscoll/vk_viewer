use ozy::render::PositionNormalTangentUvPrimitive;
use ozy::structs::UninterleavedVertexArrays;
use crate::render::{Model, PrimitiveKey};
use crate::vkutil::{DeferredImage, VertexFetchOffsets};
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

#[inline]
pub fn calculate_miplevels(width: u32, height: u32) -> u32 {
    (f32::floor(f32::log2(u32::max(width, height) as f32))) as u32 + 1
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
        Err(e) => {
            crash_with_error_dialog(&format!("{}\n{}", msg, e));
        }
    }
}

pub fn unix_epoch_ms() -> u128 {
    SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis()
}

pub fn compute_terrain_vertices(spec: &mut TerrainSpec, fixed_seed: bool, scale: f32) -> UninterleavedVertexArrays {
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
    primitive_key: PrimitiveKey,
    terrain: &mut TerrainSpec,
    terrain_generation_scale: f32
) {
    if let Some(ter) = renderer.get_primitive(primitive_key) {
        let offset = ter.position_offset;
        let verts = compute_terrain_vertices(terrain, terrain.fixed_seed, terrain_generation_scale);
        replace_uploaded_vertices(vk, renderer, &verts, offset.into());

        physics_engine.collider_set.remove(*terrain_collider_handle, &mut physics_engine.island_manager, &mut physics_engine.rigid_body_set, false);

        *terrain_collider_handle = physics_engine.make_terrain_collider(&verts.positions, terrain.vertex_width, terrain.vertex_height);
    }
}

pub fn upload_vertex_attributes(vk: &mut VulkanAPI, renderer: &mut Renderer, attribs: &UninterleavedVertexArrays) -> VertexFetchOffsets {
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

pub fn upload_primitive_vertices<T: PositionNormalTangentUvPrimitive>(vk: &mut VulkanAPI, renderer: &mut Renderer, prim: &T) -> VertexFetchOffsets {
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

pub fn replace_uploaded_vertices(vk: &mut VulkanAPI, renderer: &mut Renderer, attributes: &UninterleavedVertexArrays, offset: u64) {
    renderer.replace_vertex_positions(vk, &attributes.positions, offset);
    renderer.replace_vertex_tangents(vk, &attributes.tangents, offset);
    renderer.replace_vertex_normals(vk, &attributes.normals, offset);
    renderer.replace_vertex_uvs(vk, &attributes.uvs, offset);
}

pub fn reset_totoro(physics_engine: &mut PhysicsEngine, totoro: &Option<PhysicsProp>) {
    let handle = totoro.as_ref().unwrap().rigid_body_handle;
    if let Some(body) = physics_engine.rigid_body_set.get_mut(handle) {
        body.set_linvel(glm::zero(), true);
        body.set_position(Isometry::from_parts(Translation::new(0.0, 0.0, 20.0), *body.rotation()), true);
    }
}

pub fn make_index_buffer(vk: &mut VulkanAPI, indices: &[u32]) -> GPUBuffer {
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
