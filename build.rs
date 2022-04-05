use std::process::Command;

fn main() {
    println!("Starting");
    /*
    for entry in std::fs::read_dir("./shaders").unwrap() {
        let entry = entry.unwrap();
        let name = entry.file_name().into_string().unwrap();
    }
    */
        
    if let Err(e) = std::fs::create_dir("./shaders") {
        println!("{}", e);
    }

    let out = Command::new("glslangValidator").args(["-V", "-S", "vert", "-o" , "./shaders/main_vert.spv", "./src/shaders/main.vs"]).output().unwrap();
    println!("{:?}", out);
    let out = Command::new("glslangValidator").args(["-V", "-S", "frag", "-o" , "./shaders/main_frag.spv", "./src/shaders/main.fs"]).output().unwrap();
    println!("{:?}", out);
    let out = Command::new("glslangValidator").args(["-V", "-S", "vert", "-o" , "./shaders/imgui_vert.spv", "./src/shaders/imgui.vs"]).output().unwrap();
    println!("{:?}", out);
    let out = Command::new("glslangValidator").args(["-V", "-S", "frag", "-o" , "./shaders/imgui_frag.spv", "./src/shaders/imgui.fs"]).output().unwrap();
    println!("{:?}", out);

    //Copy SDL2 dlls to target directory
    if let Err(e) = std::fs::copy("./redist/SDL2_mixer.dll", "./target/release/SDL2_mixer.dll") {
        println!("{}", e);
    }
    if let Err(e) = std::fs::copy("./redist/libmpg123-0.dll", "./target/release/libmpg123-0.dll") {
        println!("{}", e);            
    }    
}
