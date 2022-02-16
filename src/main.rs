extern crate nalgebra_glm as glm;

use ash::vk;
use ash::vk::{Handle};
use sdl2::event::Event;
use std::ffi::CStr;
use std::os::raw::c_void;
use std::ptr;

//Entry point
fn main() {
    //Create the window using SDL
    let sdl_ctxt = sdl2::init().unwrap();
    let mut event_pump = sdl_ctxt.event_pump().unwrap();
    let video_subsystem = sdl_ctxt.video().unwrap();
    let window_size = glm::vec2(1024, 1024);
    let window = video_subsystem.window("Trying to use Vulkan", window_size.x, window_size.y).position_centered().vulkan().build().unwrap();

    //Initialize the Vulkan API
    let vk_entry = ash::Entry::linked();
    let vk_instance = {
        let app_info = vk::ApplicationInfo {
            api_version: vk::make_api_version(0, 1, 3, 0),
            ..Default::default()
        };

        let extension_names = [
            ash::extensions::khr::Surface::name().as_ptr(),
            ash::extensions::khr::Win32Surface::name().as_ptr()
        ];
        let layer_names = unsafe  {[
            CStr::from_bytes_with_nul_unchecked(b"VK_LAYER_KHRONOS_validation\0").as_ptr()
        ]};
        let vk_create_info = vk::InstanceCreateInfo {
            p_application_info: &app_info,
            enabled_extension_count: extension_names.len() as u32,
            pp_enabled_extension_names: &extension_names as *const *const _,
            enabled_layer_count: layer_names.len() as u32,
            pp_enabled_layer_names: &layer_names as *const *const _,
            ..Default::default()
        };

        unsafe { vk_entry.create_instance(&vk_create_info, None).unwrap() }
    };

    //Use SDL to create the Vulkan surface
    let vk_surface = {
        let raw_surf = window.vulkan_create_surface(vk_instance.handle().as_raw() as usize).unwrap();
        vk::SurfaceKHR::from_raw(raw_surf)
    };
    
    //Create surface extension object
    let vk_ext_surface = ash::extensions::khr::Surface::new(&vk_entry, &vk_instance);

    //Create the Vulkan device
    let vk_physical_device;
    let mut vk_queue_family_index = 0;
    let vk_device = unsafe {
        match vk_instance.enumerate_physical_devices() {
            Ok(phys_devices) => {
                vk_physical_device = phys_devices[0];

                let mut i = 0;
                for qfp in vk_instance.get_physical_device_queue_family_properties(vk_physical_device) {
                    if qfp.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
                        vk_queue_family_index = i;
                        break;
                    }
                    i += 1;
                }

                let priorities = [1.0];
                let queue_create_info = vk::DeviceQueueCreateInfo {
                    queue_family_index: vk_queue_family_index,
                    queue_count: 1,
                    p_queue_priorities: priorities.as_ptr(),
                    ..Default::default()
                };
                
                if !vk_ext_surface.get_physical_device_surface_support(vk_physical_device, vk_queue_family_index, vk_surface).unwrap() {
                    panic!("The physical device queue doesn't support swapchain present!");
                }

                let extension_names = [ash::extensions::khr::Swapchain::name().as_ptr()];
                let create_info = vk::DeviceCreateInfo {
                    queue_create_info_count: 1,
                    p_queue_create_infos: [queue_create_info].as_ptr(),
                    enabled_extension_count: extension_names.len() as u32,
                    pp_enabled_extension_names: extension_names.as_ptr(),
                    ..Default::default()
                };

                vk_instance.create_device(vk_physical_device, &create_info, None).unwrap()
            }
            Err(e) => {
                panic!("Unable to enumerate physical devices: {}", e);
            }
        }
    };

    //Create swapchain extension object
    let vk_ext_swapchain = ash::extensions::khr::Swapchain::new(&vk_instance, &vk_device);

    //Create the main swapchain for window present
    let vk_swapchain = unsafe {
        let present_mode = vk_ext_surface.get_physical_device_surface_present_modes(vk_physical_device, vk_surface).unwrap()[0];
        let surf_capabilities = vk_ext_surface.get_physical_device_surface_capabilities(vk_physical_device, vk_surface).unwrap();
        let surf_format = vk_ext_surface.get_physical_device_surface_formats(vk_physical_device, vk_surface).unwrap()[0];
        let create_info = vk::SwapchainCreateInfoKHR {
            surface: vk_surface,
            min_image_count: surf_capabilities.min_image_count,
            image_format: surf_format.format,
            image_color_space: surf_format.color_space,
            image_extent: surf_capabilities.current_extent,
            image_array_layers: 1,
            image_usage: vk::ImageUsageFlags::COLOR_ATTACHMENT,
            image_sharing_mode: vk::SharingMode::EXCLUSIVE,
            queue_family_index_count: 1,
            p_queue_family_indices: [vk_queue_family_index].as_ptr(),
            pre_transform: surf_capabilities.current_transform,
            composite_alpha: vk::CompositeAlphaFlagsKHR::OPAQUE,
            present_mode,
            ..Default::default()
        };

        vk_ext_swapchain.create_swapchain(&create_info, None).unwrap()
    };

    let vk_depth_buffer = unsafe {
        let present_mode = vk_ext_surface.get_physical_device_surface_present_modes(vk_physical_device, vk_surface).unwrap()[0];
        let surf_format = vk_ext_surface.get_physical_device_surface_formats(vk_physical_device, vk_surface).unwrap()[0];

        let surf_capabilities = vk_ext_surface.get_physical_device_surface_capabilities(vk_physical_device, vk_surface).unwrap();
        let extent = vk::Extent3D {
            width: surf_capabilities.current_extent.width,
            height: surf_capabilities.current_extent.height,
            depth: 1
        };
        let create_info = vk::ImageCreateInfo {
            queue_family_index_count: 1,
            p_queue_family_indices: [vk_queue_family_index].as_ptr(),
            flags: vk::ImageCreateFlags::empty(),
            image_type: vk::ImageType::TYPE_2D,
            format: vk::Format::D16_UNORM,
            extent,
            mip_levels: 1,
            array_layers: 1,
            samples: vk::SampleCountFlags::TYPE_4,
            usage: vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            ..Default::default()
        };

        let depth_image = vk_device.create_image(&create_info, None).unwrap();
        let mem_reqs = vk_device.get_image_memory_requirements(depth_image);
        
    };
    
    //Main application loop
    'running: loop {
        for event in event_pump.poll_iter() {
            match event {
                Event::Quit{..} => { break 'running; }
                _ => {}
            }
        }
    }
}
