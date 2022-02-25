extern crate nalgebra_glm as glm;
extern crate ozy_engine as ozy;

use ash::vk;
use ash::vk::{Handle};
use sdl2::event::Event;
use std::ffi::CStr;
use std::mem::size_of;
use std::os::raw::c_void;
use std::ptr;

use ozy::structs::FrameTimer;

unsafe fn get_memory_type_index(
    vk_instance: &ash::Instance,
    vk_physical_device: vk::PhysicalDevice,
    memory_requirements: vk::MemoryRequirements,
    flags: vk::MemoryPropertyFlags
) -> u32 {
    let mut i = 0;
    let mut memory_type_index = 0;
    let mut largest_heap = 0;
    let phys_device_mem_props = vk_instance.get_physical_device_memory_properties(vk_physical_device);
    for mem_type in phys_device_mem_props.memory_types {
        if memory_requirements.memory_type_bits & (1 << i) == 0 {
            continue;
        }

        if mem_type.property_flags.contains(flags) {
            let heap_size = phys_device_mem_props.memory_heaps[mem_type.heap_index as usize].size;
            if heap_size > largest_heap {
                memory_type_index = i;
                largest_heap = heap_size;
            }
        }
        i += 1;
    }

    memory_type_index
}

//Entry point
fn main() {
    //Create the window using SDL
    let sdl_ctxt = sdl2::init().unwrap();
    let mut event_pump = sdl_ctxt.event_pump().unwrap();
    let video_subsystem = sdl_ctxt.video().unwrap();
    let window_size = glm::vec2(1024, 1024);
    let window = video_subsystem.window("Vulkan't", window_size.x, window_size.y).position_centered().vulkan().build().unwrap();

    //WASAPI initialization
    let wasapi_device = wasapi::get_default_device(&wasapi::Direction::Render).unwrap();
    let mut audio_client = wasapi_device.get_iaudioclient().unwrap();
    println!("Selected audio output device: {}", wasapi_device.get_friendlyname().unwrap());
    let blockalign;
    let sample_rate;
    let audio_render_client = {
        let format = audio_client.get_mixformat().unwrap();
        sample_rate = format.get_samplespersec();
        println!("Sample rate: {}", sample_rate);
        blockalign = format.get_blockalign() as usize;
        let default_period = audio_client.get_periods().unwrap().0;
        audio_client.initialize_client(&format, 0, &wasapi::Direction::Render, &wasapi::ShareMode::Shared, false).unwrap();
        audio_client.set_get_eventhandle().unwrap();
        audio_client.get_audiorenderclient().unwrap()
    };

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

    let vk_depth_image = unsafe {
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

        //Search for the largest DEVICE_LOCAL heap the device advertises
        let memory_type_index = get_memory_type_index(&vk_instance, vk_physical_device, mem_reqs, vk::MemoryPropertyFlags::DEVICE_LOCAL);

        let allocate_info = vk::MemoryAllocateInfo {
            allocation_size: mem_reqs.size,
            memory_type_index,
            ..Default::default()
        };
        let depth_memory = vk_device.allocate_memory(&allocate_info, None).unwrap();

        //Bind the depth image to its memory
        vk_device.bind_image_memory(depth_image, depth_memory, 0).unwrap();

        depth_image
    };

    let vk_depth_buffer_view = unsafe {
        let image_subresource_range = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::DEPTH,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1
        };
        let components = vk::ComponentMapping {
            r: vk::ComponentSwizzle::R,
            g: vk::ComponentSwizzle::G,
            b: vk::ComponentSwizzle::B,
            a: vk::ComponentSwizzle::A,
        };
        let view_info = vk::ImageViewCreateInfo {
            image: vk_depth_image,
            format: vk::Format::D16_UNORM,
            view_type: vk::ImageViewType::TYPE_2D,
            components,
            subresource_range: image_subresource_range,
            ..Default::default()
        };

        vk_device.create_image_view(&view_info, None).unwrap()
    };

    let vk_uniform_buffer = unsafe {
        let buffer_create_info = vk::BufferCreateInfo {
            usage: vk::BufferUsageFlags::UNIFORM_BUFFER,
            size: 16,
            queue_family_index_count: 0,
            p_queue_family_indices: ptr::null(),
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            ..Default::default()
        };

        let vk_uniform_buffer = vk_device.create_buffer(&buffer_create_info, None).unwrap();
        
        let mem_reqs = vk_device.get_buffer_memory_requirements(vk_uniform_buffer);
        let memory_type_index = get_memory_type_index(&vk_instance, vk_physical_device, mem_reqs,vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT);

        let alloc_info = vk::MemoryAllocateInfo {
            allocation_size: mem_reqs.size,
            memory_type_index,
            ..Default::default()
        };
        let uniform_buffer_memory = vk_device.allocate_memory(&alloc_info, None).unwrap();

        vk_device.bind_buffer_memory(vk_uniform_buffer, uniform_buffer_memory, 0).unwrap();

        vk_uniform_buffer
    };

    {
        let layout_binding = vk::DescriptorSetLayoutBinding {
            binding: 0,
            descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
            stage_flags: vk::ShaderStageFlags::FRAGMENT,
            p_immutable_samplers: ptr::null(),
            ..Default::default()
        };
        
    }
    
    
    audio_client.start_stream().unwrap();
    //Main application loop
    let mut sin_t = 0.0;
    let mut timer = FrameTimer::new();
    'running: loop {
        timer.update(); //Update frame timer

        //Pump window's event loop
        for event in event_pump.poll_iter() {
            match event {
                Event::Quit{..} => { break 'running; }
                _ => {}
            }
        }

        //Update

        //Fill available part of audio buffer with sine wave
        {
            let framecount = audio_client.get_available_space_in_frames().unwrap() as usize;
            let mut data = vec![0; framecount * blockalign];
            for frame in data.chunks_exact_mut(blockalign) {
                let freq = 700.0;
                let sample = 0.2 * f32::sin(glm::two_pi::<f32>() * freq * sin_t);

                let sample_bytes = sample.to_le_bytes();
                for v in frame.chunks_exact_mut(blockalign / 2) {
                    for (bufbyte, sinbyte) in v.iter_mut().zip(sample_bytes.iter()) {
                        *bufbyte = *sinbyte;
                    }
                }

                sin_t += 1.0 / sample_rate as f32;
                let max_t = 1.0 / freq;
                if sin_t > max_t {
                    sin_t -= max_t;
                }
            }

            audio_render_client.write_to_device(framecount, blockalign, &data).unwrap();
        }
    }
}
