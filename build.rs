use std::{process::Command, fs::OpenOptions};
use std::io::{Write, BufWriter};

const SHADER_SRC_DIR: &str = "./src/shaders";
const SHADER_OUTPUT_DIR: &str = "./data/shaders";

fn compile_slang_shader(stage: &str, src_file: &str, out_file: &str) -> String {
    let src_path = format!("{}/{}", SHADER_SRC_DIR, src_file);
    let out_path = format!("{}/{}", SHADER_OUTPUT_DIR, out_file);

    //#[cfg(debug_assertions)]
    //let args = ["-stage", stage, "-entry", &format!("{}_main", stage), "-g3", "-O0", "-o", &out_path, &src_path];

    //#[cfg(not(debug_assertions))]
    //let args = ["-stage", stage, "-entry", &format!("{}_main", stage), "-o", &out_path, &src_path];

    let args = ["-stage", stage, "-entry", &format!("{}_main", stage), "-g3", "-O0", "-o", &out_path, &src_path];
    let out = Command::new("slangc").args(args).output().unwrap();
    unsafe { format!("{}\n{}\n", String::from_utf8_unchecked(out.stdout), String::from_utf8_unchecked(out.stderr)) }
}

fn main() {
    let mut build_log = BufWriter::new(OpenOptions::new().write(true).truncate(true).create(true).open("./build.log").unwrap());
    write!(build_log, "Starting compilation...\n").unwrap();
    
    if let Err(e) = std::fs::remove_dir_all(SHADER_OUTPUT_DIR) {
        println!("{}", e);
    }
    if let Err(e) = std::fs::create_dir(SHADER_OUTPUT_DIR) {
        println!("{}", e);
    }

    //HLSL shader compilation
    // let out = Command::new("dxc").args([
    //         "-Fo", &format!("{}/shadow_vert.spv", SHADER_OUTPUT_DIR),
    //         "-E", "main",
    //         "-T", "vs_6_6",
    //         "-spirv", &format!("{}/shadow_vertex.hlsl", SHADER_SRC_DIR)
    //     ]
    // ).output().unwrap();
    // let out = unsafe { format!("{}\n{}\n", String::from_utf8_unchecked(out.stdout), String::from_utf8_unchecked(out.stderr)) };
    // write!(build_log, "{}\n", out).unwrap();

    //Slang shader compilation
    let slang_shaders = [
        ["vertex", "model_vertex.slang", "vertex_main.spv"],
        ["vertex", "atmosphere.slang", "atmosphere_vert.spv"],
        ["fragment", "atmosphere.slang", "atmosphere_frag.spv"],
        ["fragment", "model_fragment.slang", "pbr_metallic_roughness.spv"],
        ["fragment", "terrain_fragment.slang", "terrain.spv"],
        ["vertex", "imgui.slang", "imgui_vert.spv"],
        ["fragment", "imgui.slang", "imgui_frag.spv"],
        ["vertex", "shadow.slang", "shadow_vert.spv"],
        ["fragment", "shadow.slang", "shadow_frag.spv"],
        ["vertex", "postfx.slang", "postfx_vert.spv"],
        ["fragment", "postfx.slang", "postfx_frag.spv"],
        ["compute", "lum_binning.slang", "lum_binning.spv"],
    ];
    for shader in slang_shaders {
        let out = compile_slang_shader(shader[0], shader[1], shader[2]);
        write!(build_log, "{}\n", out).unwrap();
    }

    //Copy SDL2 dlls to target directory
    let envs = ["debug", "release", "master"];
    //let files = ["SDL2.dll", "SDL2_mixer.dll", "libmpg123-0.dll"];
    for env in envs {
        for path in std::fs::read_dir("./redist").unwrap() {
            let entry = path.unwrap().file_name();
            let filename = entry.to_str().unwrap();
            if let Err(e) = std::fs::copy(&format!("./redist/{}", filename), &format!("./target/{}/{}", env, filename)) {
                write!(build_log, "{}\n", e).unwrap();
            }
        }
    }

    write!(build_log, "Compilation finished.\n").unwrap();
}
