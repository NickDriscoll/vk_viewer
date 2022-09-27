use std::{process::Command, fs::OpenOptions};
use std::io::Write;

const SHADER_SRC_DIR: &str = "./src/shaders";
const SHADER_OUTPUT_DIR: &str = "./data/shaders";

fn compile_slang_shader(stage: &str, src_file: &str, out_file: &str) -> String {
    let out = Command::new("slangc").args([
        "-stage", stage,
        "-entry", &format!("{}_main", stage),
        "-o",
        &format!("{}/{}", SHADER_OUTPUT_DIR, out_file),
        &format!("{}/{}", SHADER_SRC_DIR, src_file)]
    ).output().unwrap();
    unsafe { format!("{}\n{}\n", String::from_utf8_unchecked(out.stdout), String::from_utf8_unchecked(out.stderr)) }
}

fn main() {
    let mut build_log = OpenOptions::new().write(true).truncate(true).create(true).open("./build_output.log").unwrap();
    write!(build_log, "Starting compilation...\n").unwrap();
    
    if let Err(e) = std::fs::remove_dir_all(SHADER_OUTPUT_DIR) {
        println!("{}", e);
    }
    if let Err(e) = std::fs::create_dir(SHADER_OUTPUT_DIR) {
        println!("{}", e);
    }

    //HLSL shader compilation
    let out = Command::new("dxc").args([
            "-Fo", &format!("{}/shadow_vert.spv", SHADER_OUTPUT_DIR),
            "-E", "main",
            "-T", "vs_6_6",
            "-spirv", &format!("{}/shadow_vertex.hlsl", SHADER_SRC_DIR)
        ]
    ).output().unwrap();
    let out = unsafe { format!("{}\n{}\n", String::from_utf8_unchecked(out.stdout), String::from_utf8_unchecked(out.stderr)) };
    write!(build_log, "{}\n", out).unwrap();

    //Slang shader compilation
    let slang_shaders = [
        ["vertex", "model_vertex.slang", "vertex_main.spv"],
        ["vertex", "atmosphere.slang", "atmosphere_vert.spv"],
        ["fragment", "atmosphere.slang", "atmosphere_frag.spv"],
        ["fragment", "model_fragment.slang", "pbr_metallic_roughness.spv"],
        ["fragment", "terrain_fragment.slang", "terrain.spv"],
        ["vertex", "imgui.slang", "imgui_vert.spv"],
        ["fragment", "imgui.slang", "imgui_frag.spv"],
        ["fragment", "shadow.slang", "shadow_frag.spv"],
        ["vertex", "postfx.slang", "postfx_vert.spv"],
        ["fragment", "postfx.slang", "postfx_frag.spv"],
    ];
    for shader in slang_shaders {
        let out = compile_slang_shader(shader[0], shader[1], shader[2]);
        write!(build_log, "{}\n", out).unwrap();
    }

    //Copy SDL2 dlls to target directory
    let envs = ["debug", "release"];
    let files = ["SDL2.dll", "SDL2_mixer.dll", "libmpg123-0.dll"];
    for env in envs {
        for file in files {
            if let Err(e) = std::fs::copy(&format!("./redist/{}", file), &format!("./target/{}/{}", env, file)) {
                write!(build_log, "{}\n", e).unwrap();
            }
        }
    }

    write!(build_log, "Compilation finished.\n").unwrap();
}
