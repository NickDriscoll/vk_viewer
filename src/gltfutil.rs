use gltf::Gltf;
use std::ptr;

use crate::crash_with_error_dialog;

pub fn gltf_meshdata(path: &str) -> (Vec<f32>, Vec<u32>) {
    let glb = Gltf::open(path).unwrap();
    let mut vertex_buffer = Vec::new();
    let mut index_buffer = Vec::new();
    for scene in glb.scenes() {
        for node in scene.nodes() {
            println!("Processing node");
            if let Some(mesh) = node.mesh() {
                use gltf::Semantic;
                let semantics = [Semantic::Positions, Semantic::Normals, Semantic::Tangents, Semantic::TexCoords(0)];
                let mut vecs: [Vec<f32>; 4] = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
                for prim in mesh.primitives() {
                    println!("Primitive mode: {:?}", prim.mode());
                    for s_idx in 0..semantics.len() {
                        let sem = &semantics[s_idx];
                        let acc = prim.get(sem).unwrap();
                        let view = acc.view().unwrap();

                        println!("Buffer length: {}\nOffset: {}", view.length(), view.offset());
                        println!("Data type: {:?}", acc.data_type());

                        let byte_stride = match view.stride() {
                            Some(s) => { s }
                            None => { 0 }
                        };
                        println!("Byte stride: {}", byte_stride);

                        unsafe {
                            if let Some(blob) = &glb.blob {
                                vecs[s_idx] = vec![0.0; view.length()/4];
                                let src_ptr = blob.as_ptr() as *const u8;
                                let src_ptr = src_ptr.offset(view.offset() as isize);
                                ptr::copy_nonoverlapping(src_ptr, vecs[s_idx].as_mut_ptr() as *mut u8, view.length());
                            }
                        }
                    }

                    let acc = prim.indices().unwrap();
                    println!("Index data type: {:?}", acc.data_type());
                    let view = acc.view().unwrap();

                    let index_count = view.length() / 2;
                    index_buffer = vec![0u32; index_count];
                    if let Some(blob) = &glb.blob {
                        for i in 0..index_count {
                            let current_idx = 2 * i + view.offset();
                            let bytes = [blob[current_idx], blob[current_idx + 1], 0, 0];
                            index_buffer[i] = u32::from_le_bytes(bytes);
                        }
                    }

                    //Handle material data
                    use gltf::image::Source;
                    let mat = prim.material();
                    let tex_data_source = mat.pbr_metallic_roughness().base_color_texture().unwrap().texture().source().source();
                    match tex_data_source {
                        Source::View {view, mime_type} => {
                            if mime_type.ne("image/png") {
                                crash_with_error_dialog(&format!("Error loading image from glb\nUnsupported image type: {}", mime_type));
                            }
                            
                        }
                        Source::Uri {uri, mime_type} => {
                            crash_with_error_dialog("Uri not supported");
                        }
                    }
                }

                //Now, interleave the mesh data
                let position_vec = &vecs[0];
                let normal_vec = &vecs[1];
                let tangent_vec = &vecs[2];
                let texcoord_vec = &vecs[3];
                vertex_buffer = vec![0.0f32; position_vec.len() + normal_vec.len() + 6 * tangent_vec.len() / 4 + texcoord_vec.len()];
                
                for i in 0..(vertex_buffer.len() / 14) {
                    let current_idx = i * 14;

                    let normal = glm::vec3(normal_vec[3 * i], normal_vec[3 * i + 1], normal_vec[3 * i + 2]);
                    let tangent = glm::vec3(tangent_vec[4 * i], tangent_vec[4 * i + 1], tangent_vec[4 * i + 2]);
                    let bitangent = glm::cross(&normal, &tangent);

                    vertex_buffer[current_idx] = position_vec[3 * i];
                    vertex_buffer[current_idx + 1] = position_vec[3 * i + 1];
                    vertex_buffer[current_idx + 2] = position_vec[3 * i + 2];
                    vertex_buffer[current_idx + 3] = tangent_vec[4 * i];
                    vertex_buffer[current_idx + 4] = tangent_vec[4 * i + 1];
                    vertex_buffer[current_idx + 5] = tangent_vec[4 * i + 2];
                    vertex_buffer[current_idx + 6] = bitangent.x;
                    vertex_buffer[current_idx + 7] = bitangent.y;
                    vertex_buffer[current_idx + 8] = bitangent.z;
                    vertex_buffer[current_idx + 9] = normal_vec[3 * i];
                    vertex_buffer[current_idx + 10] = normal_vec[3 * i + 1];
                    vertex_buffer[current_idx + 11] = normal_vec[3 * i + 2];
                    vertex_buffer[current_idx + 12] = texcoord_vec[2 * i];
                    vertex_buffer[current_idx + 13] = texcoord_vec[2 * i + 1];
                }
            }
        }
    }
    (vertex_buffer, index_buffer)
}