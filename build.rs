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
    format!("{:?}", out)
}

fn main() {
    let mut build_log = OpenOptions::new().write(true).create(true).open("./build_output.log").unwrap();
    write!(build_log, "Starting compilation...\n").unwrap();
    
    if let Err(e) = std::fs::create_dir(SHADER_OUTPUT_DIR) {
        println!("{}", e);
    }

    let path = "./src/shaders/vertex";
    for entry in std::fs::read_dir(path).unwrap() {
        let entry = entry.unwrap();
        let name = entry.file_name().into_string().unwrap();
        let out = Command::new("glslc").args(["-I ..", "-fshader-stage=vert", "-o" , &format!("{}/{}.spv", SHADER_OUTPUT_DIR, name), &format!("{}/{}", path, name)]).output().unwrap();
        write!(build_log, "{:?}\n", out).unwrap();
    }

    let path = "./src/shaders/fragment";
    for entry in std::fs::read_dir(path).unwrap() {
        let entry = entry.unwrap();
        let name = entry.file_name().into_string().unwrap();
        let out = Command::new("glslc").args(["-I ..", "-fshader-stage=frag", "-o" , &format!("{}/{}.spv", SHADER_OUTPUT_DIR, name), &format!("{}/{}", path, name)]).output().unwrap();
        write!(build_log, "{:?}\n", out).unwrap();
    }

    //Slang shader compilation
    let out = compile_slang_shader("vertex", "main.slang", "vertex_main.spv");
    write!(build_log, "{}\n", out).unwrap();
    
    let out = compile_slang_shader("vertex", "atmosphere.slang", "atmosphere_vert.spv");
    write!(build_log, "{}\n", out).unwrap();
    
    let out = compile_slang_shader("fragment", "atmosphere.slang", "atmosphere_frag.spv");
    write!(build_log, "{}\n", out).unwrap();
    
    let out = compile_slang_shader("fragment", "pbr_metallic_roughness.slang", "pbr_metallic_roughness.spv");
    write!(build_log, "{}\n", out).unwrap();

    //Copy SDL2 dlls to target directory
    if let Err(e) = std::fs::copy("./redist/SDL2_mixer.dll", "./target/release/SDL2_mixer.dll") {
        write!(build_log, "{}\n", e).unwrap();
    }
    if let Err(e) = std::fs::copy("./redist/libmpg123-0.dll", "./target/release/libmpg123-0.dll") {
        write!(build_log, "{}\n", e).unwrap();            
    }
    if let Err(e) = std::fs::copy("./redist/SDL2_mixer.dll", "./target/debug/SDL2_mixer.dll") {
        write!(build_log, "{}\n", e).unwrap();
    }
    if let Err(e) = std::fs::copy("./redist/libmpg123-0.dll", "./target/debug/libmpg123-0.dll") {
        write!(build_log, "{}\n", e).unwrap();            
    }
}
