use ozy::structs::UninterleavedVertexArrays;
use crate::vkutil::{VertexFetchOffsets};
use crate::*;

//Converts any data structure to a slice of bytes
pub fn struct_to_bytes<'a, T>(structure: &'a T) -> &'a [u8] {
    let p = structure as *const _ as *const u8;
    let size = size_of::<T>();
    unsafe { std::slice::from_raw_parts(p, size) }
}

pub fn slice_to_bytes<'a, T>(in_array: &'a [T]) -> &'a [u8] {
    unsafe { core::slice::from_raw_parts(in_array.as_ptr() as *const u8, in_array.len() * size_of::<T>()) }
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
    terrain_model_idx: usize,
    terrain: &mut TerrainSpec,
    terrain_generation_scale: f32
) {
    if let Some(ter) = renderer.get_model(terrain_model_idx) {
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

pub fn upload_primitive_vertices(vk: &mut VulkanAPI, renderer: &mut Renderer, prim: &GLTFPrimitive) -> VertexFetchOffsets {
    let position_offset = renderer.append_vertex_positions(vk, &prim.vertex_positions);
    let tangent_offset = renderer.append_vertex_tangents(vk, &prim.vertex_tangents);
    let normal_offset = renderer.append_vertex_normals(vk, &prim.vertex_normals);
    let uv_offset = renderer.append_vertex_uvs(vk, &prim.vertex_uvs);

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

pub fn upload_gltf_primitives(vk: &mut VulkanAPI, renderer: &mut Renderer, data: &GLTFMeshData, pipeline: vk::Pipeline) -> Vec<usize> {
    let mut indices = vec![];
    let mut tex_id_map = HashMap::new();
    for prim in &data.primitives {
        let color_idx = if let Some(idx) = prim.material.color_index {
            match tex_id_map.get(&idx) {
                Some(id) => { *id }
                None => {
                    let image = GPUImage::from_png_bytes(vk, data.texture_bytes[idx].as_slice());
                    let image_info = vk::DescriptorImageInfo {
                        sampler: renderer.material_sampler,
                        image_view: image.vk_view,
                        image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
                    };
                    let global_tex_id = renderer.global_textures.insert(image_info) as u32;
                    tex_id_map.insert(idx, global_tex_id);
                    global_tex_id
                }
            }
        } else {
            renderer.default_diffuse_idx
        };

        let normal_idx = match prim.material.normal_index {
            Some(idx) => {
                match tex_id_map.get(&idx) {
                    Some(id) => { *id }
                    None => {
                        let image = GPUImage::from_png_bytes(vk, data.texture_bytes[idx].as_slice());
                        let image_info = vk::DescriptorImageInfo {
                            sampler: renderer.material_sampler,
                            image_view: image.vk_view,
                            image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
                        };
                        let global_tex_id = renderer.global_textures.insert(image_info) as u32;
                        tex_id_map.insert(idx, global_tex_id);
                        global_tex_id
                    }
                }
            }
            None => { renderer.default_normal_idx }
        };

        let metal_roughness_idx = match prim.material.metallic_roughness_index {
            Some(idx) => {
                match tex_id_map.get(&idx) {
                    Some(id) => { *id }
                    None => {
                        let image = GPUImage::from_png_bytes(vk, data.texture_bytes[idx].as_slice());
                        let image_info = vk::DescriptorImageInfo {
                            sampler: renderer.material_sampler,
                            image_view: image.vk_view,
                            image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
                        };
                        let global_tex_id = renderer.global_textures.insert(image_info) as u32;
                        tex_id_map.insert(idx, global_tex_id);
                        global_tex_id
                    }
                }

            }
            None => {
                renderer.default_metal_roughness_idx
            }
        };

        let emissive_idx = match prim.material.emissive_index {
            Some(idx) => {
                match tex_id_map.get(&idx) {
                    Some(id) => { *id }
                    None => {
                        let image = GPUImage::from_png_bytes(vk, data.texture_bytes[idx].as_slice());
                        let image_info = vk::DescriptorImageInfo {
                            sampler: renderer.material_sampler,
                            image_view: image.vk_view,
                            image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
                        };
                        let global_tex_id = renderer.global_textures.insert(image_info) as u32;
                        tex_id_map.insert(idx, global_tex_id);
                        global_tex_id
                    }
                }

            }
            None => {
                renderer.default_emissive_idx
            }
        };

        let material = Material {
            pipeline,
            base_color: prim.material.base_color,
            base_roughness: prim.material.base_roughness,
            color_idx,
            normal_idx,
            metal_roughness_idx,
            emissive_idx
        };
        let material_idx = renderer.global_materials.insert(material) as u32;

        let offsets = upload_primitive_vertices(vk, renderer, &prim);

        let index_buffer = vkutil::make_index_buffer(vk, &prim.indices);
        let model_idx = renderer.register_model(Primitive {
            shadow_type: ShadowType::OpaqueCaster,
            index_buffer,
            index_count: prim.indices.len().try_into().unwrap(),
            position_offset: offsets.position_offset,
            tangent_offset: offsets.tangent_offset,
            normal_offset: offsets.normal_offset,
            uv_offset: offsets.uv_offset,
            material_idx
        });
        indices.push(model_idx);
    }
    indices
}

pub fn reset_totoro(physics_engine: &mut PhysicsEngine, totoro: &Option<PhysicsProp>) {
    let handle = totoro.as_ref().unwrap().rigid_body_handle;
    if let Some(body) = physics_engine.rigid_body_set.get_mut(handle) {
        body.set_linvel(glm::zero(), true);
        body.set_position(Isometry::from_parts(Translation::new(0.0, 0.0, 20.0), *body.rotation()), true);
    }
}