use gltf::Gltf;
use gltf::accessor::DataType;
use std::ptr;

use crate::crash_with_error_dialog;

pub enum GLTFImageType {
    PNG,
    BC7
}

pub struct GLTFMaterial {
    pub base_color: [f32; 4],
    pub color_bytes: Vec<u8>,
    pub color_imagetype: GLTFImageType,
    pub normal_bytes: Option<Vec<u8>>,
    pub normal_imagetype: GLTFImageType,
    pub metallic_roughness_bytes: Option<Vec<u8>>,
    pub metallic_roughness_imagetype: GLTFImageType,
}

pub struct GLTFPrimitive {
    pub vertex_positions: Vec<f32>,
    pub vertex_normals: Vec<f32>,
    pub vertex_tangents: Vec<f32>,
    pub vertex_uvs: Vec<f32>,
    pub indices: Vec<u32>,
    pub material: GLTFMaterial
}

pub struct GLTFData {
    pub primitives: Vec<GLTFPrimitive>
}

fn get_f32_semantic(glb: &Gltf, prim: &gltf::Primitive, semantic: gltf::Semantic) -> Option<Vec<f32>> {
    let acc = match prim.get(&semantic) {
        Some(a) => { a }
        None => { return None; }
    };
    let view = acc.view().unwrap();

    let byte_stride = match view.stride() {
        Some(s) => { s }
        None => { 0 }
    };

    unsafe {
        match &glb.blob {
            Some(blob) => {
                let mut data = vec![0.0; view.length()/4];
                let src_ptr = blob.as_ptr() as *const u8;
                let src_ptr = src_ptr.offset(view.offset() as isize);
                ptr::copy_nonoverlapping(src_ptr, data.as_mut_ptr() as *mut u8, view.length());
                Some(data)
            }
            None => {
                crash_with_error_dialog("Invalid glb. No blob.");
            }
        }
    }
}

pub fn gltf_meshdata(path: &str) -> GLTFData {
    let glb = Gltf::open(path).unwrap();
    let mut primitives = vec![];
    for mesh in glb.meshes() {
        for prim in mesh.primitives() {
            //We always expect an index buffer to be present
            let acc = prim.indices().unwrap();
            let index_buffer = unsafe {
                let view = acc.view().unwrap();
                match acc.data_type() {
                    DataType::U16 => {
                        let index_count = view.length() / 2;
                        let mut index_buffer = vec![0u32; index_count];
                        if let Some(blob) = &glb.blob {
                            for i in 0..index_count {
                                let current_idx = 2 * i + view.offset();
                                let bytes = [blob[current_idx], blob[current_idx + 1], 0, 0];
                                index_buffer[i] = u32::from_le_bytes(bytes);
                            }
                        }
                        index_buffer
                    }
                    DataType::U32 => {
                        let index_count = view.length() / 4;
                        let mut index_buffer = vec![0u32; index_count];
                        if let Some(blob) = &glb.blob {
                            let src_ptr = blob.as_ptr() as *const u8;
                            let src_ptr = src_ptr.offset(view.offset() as isize);
                            ptr::copy_nonoverlapping(src_ptr, index_buffer.as_mut_ptr() as *mut u8, view.length());
                        }
                        index_buffer
                    }
                    _ => { crash_with_error_dialog(&format!("Unsupported index type: {:?}", acc.data_type())); }
                }
            };

            //We always expect position data to be present
            use gltf::Semantic;
            let position_vec = match get_f32_semantic(&glb, &prim, Semantic::Positions) {
                Some(v) => {
                    let mut out = vec![0.0; v.len() * 4 / 3];
                    for i in (0..v.len()).step_by(3) {
                        out[4 * i / 3] =     v[i];
                        out[4 * i / 3 + 1] = v[i + 1];
                        out[4 * i / 3 + 2] = v[i + 2];
                        out[4 * i / 3 + 3] = 1.0;
                    }
                    out
                }
                None => {
                    crash_with_error_dialog("GLTF primitive was missing position attribute.");
                }
            };

            let normal_vec = match get_f32_semantic(&glb, &prim, Semantic::Normals) {
                Some(v) => {
                    //Align to float4
                    let mut out = vec![0.0; v.len() * 4 / 3];
                    for i in (0..v.len()).step_by(3) {
                        out[4 * i / 3] =     v[i];
                        out[4 * i / 3 + 1] = v[i + 1];
                        out[4 * i / 3 + 2] = v[i + 2];
                        out[4 * i / 3 + 3] = 0.0;
                    }
                    out
                }
                None => { vec![0.0; position_vec.len()] }
            };
            let tangent_vec = match get_f32_semantic(&glb, &prim, Semantic::Tangents) {
                Some(v) => { v }
                None => { vec![0.0; position_vec.len() / 3 * 4] }
            };
            let texcoord_vec = match get_f32_semantic(&glb, &prim, Semantic::TexCoords(0)) {
                Some(v) => { v }
                None => { vec![0.0; position_vec.len() / 3 * 2] }
            };

            //Handle material data
            use gltf::image::Source;
            fn png_bytes_from_source(glb: &Gltf, source: Source) -> Vec<u8> {
                let mut bytes = vec![];
                match source {
                    Source::View {view, mime_type} => unsafe {
                        if mime_type.ne("image/png") {
                            crash_with_error_dialog(&format!("Error loading image from glb\nUnsupported image type: {}", mime_type));
                        }
                        if let Some(blob) = &glb.blob {
                            bytes = vec![0u8; view.length()];
                            let src_ptr = blob.as_ptr() as *const u8;
                            let src_ptr = src_ptr.offset(view.offset() as isize);
                            ptr::copy_nonoverlapping(src_ptr, bytes.as_mut_ptr(), view.length());
                        }
                    }
                    Source::Uri {..} => {
                        crash_with_error_dialog("Uri not supported");
                    }
                }
                bytes
            }

            let mat = prim.material();
            let pbr_model = mat.pbr_metallic_roughness();

            let color_bytes = match pbr_model.base_color_texture() {
                Some(t) => {
                    let source = t.texture().source().source();
                    png_bytes_from_source(&glb, source)
                }
                None => {
                    vec![]
                }
            };

            let normal_bytes = match mat.normal_texture() {
                Some(texture) => {
                    let normal_source = texture.texture().source().source();
                    Some(png_bytes_from_source(&glb, normal_source))
                }
                None => { None }
            };

            let metallic_roughness_bytes = match pbr_model.metallic_roughness_texture() {
                Some(texture) => {
                    let source = texture.texture().source().source();
                    Some(png_bytes_from_source(&glb, source))
                }
                None => { None }
            };

            let mat = GLTFMaterial {
                base_color: pbr_model.base_color_factor(),
                color_bytes,
                color_imagetype: GLTFImageType::PNG,
                normal_bytes,
                normal_imagetype: GLTFImageType::PNG,
                metallic_roughness_bytes,
                metallic_roughness_imagetype: GLTFImageType::PNG
            };

            let p = GLTFPrimitive {
                vertex_positions: position_vec,
                vertex_normals: normal_vec,
                vertex_tangents: tangent_vec,
                vertex_uvs: texcoord_vec,
                indices: index_buffer,
                material: mat
            };
            primitives.push(p);
        }
    }

    GLTFData {
        primitives
    }
}