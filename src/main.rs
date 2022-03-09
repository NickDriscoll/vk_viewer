extern crate nalgebra_glm as glm;
extern crate ozy_engine as ozy;

use ash::vk;
use ash::vk::{Handle};
use sdl2::event::Event;
use sdl2::mixer;
use sdl2::mixer::Music;
use std::fs::File;
use std::ffi::CStr;
use std::mem::size_of;
use std::os::raw::c_void;
use std::ptr;

use ozy::structs::FrameTimer;

const COMPONENT_MAPPING_DEFAULT: vk::ComponentMapping = vk::ComponentMapping {
    r: vk::ComponentSwizzle::R,
    g: vk::ComponentSwizzle::G,
    b: vk::ComponentSwizzle::B,
    a: vk::ComponentSwizzle::A,
};

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

    //Initialize the SDL mixer
    let sdl_mixer = mixer::init(mixer::InitFlag::FLAC | mixer::InitFlag::MP3).unwrap();
    mixer::open_audio(mixer::DEFAULT_FREQUENCY, mixer::DEFAULT_FORMAT, 2, 256).unwrap();
    Music::set_volume(16);

    //Load and play bgm
    let bgm = Music::from_file("./music/bald.mp3").unwrap();
    bgm.play(-1).unwrap();

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

    //Create command buffer
    let vk_command_buffer = unsafe {
        let pool_create_info = vk::CommandPoolCreateInfo {
            queue_family_index: vk_queue_family_index,
            ..Default::default()
        };

        let command_pool = vk_device.create_command_pool(&pool_create_info, None).unwrap();

        let command_buffer_alloc_info = vk::CommandBufferAllocateInfo {
            command_pool,
            command_buffer_count: 1,
            level: vk::CommandBufferLevel::PRIMARY,
            ..Default::default()
        };
        vk_device.allocate_command_buffers(&command_buffer_alloc_info).unwrap()[0]
    };

    //Create swapchain extension object
    let vk_ext_swapchain = ash::extensions::khr::Swapchain::new(&vk_instance, &vk_device);

    //Create the main swapchain for window present
    let vk_swapchain_image_format;
    let vk_swapchain_extent;
    let vk_swapchain_image_views = unsafe {
        let present_mode = vk_ext_surface.get_physical_device_surface_present_modes(vk_physical_device, vk_surface).unwrap()[0];
        let surf_capabilities = vk_ext_surface.get_physical_device_surface_capabilities(vk_physical_device, vk_surface).unwrap();
        let surf_format = vk_ext_surface.get_physical_device_surface_formats(vk_physical_device, vk_surface).unwrap()[0];
        vk_swapchain_image_format = surf_format.format;
        vk_swapchain_extent = surf_capabilities.current_extent;
        let create_info = vk::SwapchainCreateInfoKHR {
            surface: vk_surface,
            min_image_count: surf_capabilities.min_image_count,
            image_format: vk_swapchain_image_format,
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

        let sc = vk_ext_swapchain.create_swapchain(&create_info, None).unwrap();
        let vk_swapchain_images = vk_ext_swapchain.get_swapchain_images(sc).unwrap();

        let mut image_views = Vec::with_capacity(vk_swapchain_images.len());
        for i in 0..vk_swapchain_images.len() {
            let image_subresource_range = vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1
            };
            let view_info = vk::ImageViewCreateInfo {
                image: vk_swapchain_images[i],
                format: vk_swapchain_image_format,
                view_type: vk::ImageViewType::TYPE_2D,
                components: COMPONENT_MAPPING_DEFAULT,
                subresource_range: image_subresource_range,
                ..Default::default()
            };

            image_views.push(vk_device.create_image_view(&view_info, None).unwrap());
        }

        image_views
    };

    let vk_depth_format;
    let vk_depth_image = unsafe {
        let surf_capabilities = vk_ext_surface.get_physical_device_surface_capabilities(vk_physical_device, vk_surface).unwrap();
        let extent = vk::Extent3D {
            width: surf_capabilities.current_extent.width,
            height: surf_capabilities.current_extent.height,
            depth: 1
        };

        vk_depth_format = vk::Format::D16_UNORM;
        let create_info = vk::ImageCreateInfo {
            queue_family_index_count: 1,
            p_queue_family_indices: [vk_queue_family_index].as_ptr(),
            flags: vk::ImageCreateFlags::empty(),
            image_type: vk::ImageType::TYPE_2D,
            format: vk_depth_format,
            extent,
            mip_levels: 1,
            array_layers: 1,
            samples: vk::SampleCountFlags::TYPE_1,
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

    let vk_depth_image_view = unsafe {
        let image_subresource_range = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::DEPTH,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1
        };
        let view_info = vk::ImageViewCreateInfo {
            image: vk_depth_image,
            format: vk_depth_format,
            view_type: vk::ImageViewType::TYPE_2D,
            components: COMPONENT_MAPPING_DEFAULT,
            subresource_range: image_subresource_range,
            ..Default::default()
        };

        vk_device.create_image_view(&view_info, None).unwrap()
    };

    let triangle_tint_color = [0.5f32, 0.25, 0.25, 1.0];

    let vk_descriptor_buffer_info;
    let vk_uniform_buffer = unsafe {
        let buffer_create_info = vk::BufferCreateInfo {
            usage: vk::BufferUsageFlags::UNIFORM_BUFFER,
            size: (triangle_tint_color.len() * size_of::<f32>()) as u64,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            ..Default::default()
        };

        let vk_uniform_buffer = vk_device.create_buffer(&buffer_create_info, None).unwrap();
        
        let mem_reqs = vk_device.get_buffer_memory_requirements(vk_uniform_buffer);
        let memory_type_index = get_memory_type_index(
            &vk_instance,
            vk_physical_device,
            mem_reqs,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT
        );

        let alloc_info = vk::MemoryAllocateInfo {
            allocation_size: mem_reqs.size,
            memory_type_index,
            ..Default::default()
        };
        let uniform_buffer_memory = vk_device.allocate_memory(&alloc_info, None).unwrap();

        vk_device.bind_buffer_memory(vk_uniform_buffer, uniform_buffer_memory, 0).unwrap();

        vk_descriptor_buffer_info = vk::DescriptorBufferInfo {
            buffer: vk_uniform_buffer,
            offset: 0,
            range: (triangle_tint_color.len() * size_of::<f32>()) as u64
        };

        vk_uniform_buffer
    };

    let vk_descriptor_set_layout;
    let vk_pipeline_layout = unsafe {
        let layout_binding = vk::DescriptorSetLayoutBinding {
            binding: 0,
            descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
            descriptor_count: 1,
            stage_flags: vk::ShaderStageFlags::FRAGMENT,
            p_immutable_samplers: ptr::null(),
            ..Default::default()
        };
        
        let descriptor_layout = vk::DescriptorSetLayoutCreateInfo {
            binding_count: 1,
            p_bindings: &layout_binding,
            ..Default::default()
        };

        vk_descriptor_set_layout = vk_device.create_descriptor_set_layout(&descriptor_layout, None).unwrap();

        let pipeline_layout_createinfo = vk::PipelineLayoutCreateInfo {
            push_constant_range_count: 0,
            p_push_constant_ranges: ptr::null(),
            set_layout_count: 1,
            p_set_layouts: &vk_descriptor_set_layout,
            ..Default::default()
        };
        
        vk_device.create_pipeline_layout(&pipeline_layout_createinfo, None).unwrap()
    };
    
    let vk_descriptor_sets;
    unsafe {
        let pool_size = vk::DescriptorPoolSize {
            ty: vk::DescriptorType::UNIFORM_BUFFER,
            descriptor_count: 1
        };
        let descriptor_pool_info = vk::DescriptorPoolCreateInfo {
            max_sets: 1,
            pool_size_count: 1,
            p_pool_sizes: &pool_size,
            ..Default::default()
        };
        let descriptor_pool = vk_device.create_descriptor_pool(&descriptor_pool_info, None).unwrap();

        let vk_alloc_info = vk::DescriptorSetAllocateInfo {
            descriptor_pool,
            descriptor_set_count: 1,
            p_set_layouts: &vk_descriptor_set_layout,
            ..Default::default()
        };
        vk_descriptor_sets = vk_device.allocate_descriptor_sets(&vk_alloc_info).unwrap();

        let write = vk::WriteDescriptorSet {
            dst_set: vk_descriptor_sets[0],
            descriptor_count: 1,
            descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
            p_buffer_info: &vk_descriptor_buffer_info,
            dst_array_element: 0,
            dst_binding: 0,
            ..Default::default()
        };

        vk_device.update_descriptor_sets(&[write], &[]);
    }

    let vk_render_pass = unsafe {
        let color_attachment_description = vk::AttachmentDescription {
            format: vk_swapchain_image_format,
            samples: vk::SampleCountFlags::TYPE_1,
            load_op: vk::AttachmentLoadOp::CLEAR,
            store_op: vk::AttachmentStoreOp::STORE,
            stencil_load_op: vk::AttachmentLoadOp::DONT_CARE,
            stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
            initial_layout: vk::ImageLayout::UNDEFINED,
            final_layout: vk::ImageLayout::PRESENT_SRC_KHR,
            ..Default::default()
        };

        let depth_attachment_description = vk::AttachmentDescription {
            format: vk_depth_format,
            samples: vk::SampleCountFlags::TYPE_1,
            load_op: vk::AttachmentLoadOp::CLEAR,
            store_op: vk::AttachmentStoreOp::DONT_CARE,
            stencil_load_op: vk::AttachmentLoadOp::DONT_CARE,
            stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
            initial_layout: vk::ImageLayout::UNDEFINED,
            final_layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            ..Default::default()
        };

        let color_attachment_reference = vk::AttachmentReference {
            attachment: 0,
            layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
        };
        let depth_attachment_reference = vk::AttachmentReference {
            attachment: 1,
            layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL
        };

        let subpass = vk::SubpassDescription {
            pipeline_bind_point: vk::PipelineBindPoint::GRAPHICS,
            color_attachment_count: 1,
            p_color_attachments: &color_attachment_reference,
            p_depth_stencil_attachment: &depth_attachment_reference,
            ..Default::default()
        };

        let subpass_dependency = vk::SubpassDependency {
            src_subpass: vk::SUBPASS_EXTERNAL,
            dst_subpass: 0,
            src_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            dst_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            src_access_mask: vk::AccessFlags::NONE_KHR,
            dst_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            ..Default::default()
        };

        let attachments = [color_attachment_description, depth_attachment_description];
        let renderpass_info = vk::RenderPassCreateInfo {
            attachment_count: attachments.len() as u32,
            p_attachments: &attachments as *const _,
            subpass_count: 1,
            p_subpasses: &subpass,
            dependency_count: 1,
            p_dependencies: &subpass_dependency,
            ..Default::default()
        };
        vk_device.create_render_pass(&renderpass_info, None).unwrap()
    };

    //Load shaders
    unsafe {
        let mut vert_file = File::open("./shaders/main_vert.spv").unwrap();
        let mut frag_file = File::open("./shaders/main_frag.spv").unwrap();
        let vert_spv = ash::util::read_spv(&mut vert_file).unwrap();
        let frag_spv = ash::util::read_spv(&mut frag_file).unwrap();

        let module_create_info = vk::ShaderModuleCreateInfo {
            code_size: vert_spv.len() * size_of::<u32>(),
            p_code: vert_spv.as_ptr(),
            ..Default::default()
        };
        let vert_module = vk_device.create_shader_module(&module_create_info, None).unwrap();

        let module_create_info = vk::ShaderModuleCreateInfo {
            code_size: frag_spv.len() * size_of::<u32>(),
            p_code: frag_spv.as_ptr(),
            ..Default::default()
        };
        let frag_module = vk_device.create_shader_module(&module_create_info, None).unwrap();

        let shader_stage_create_info = vk::PipelineShaderStageCreateInfo {
            stage: vk::ShaderStageFlags::VERTEX,
            p_name: "main".as_ptr() as *const i8,
            ..Default::default()
        };
    }

    //Create framebuffers
    let vk_framebuffers = unsafe {
        let mut attachments = [vk::ImageView::default(), vk_depth_image_view];
        let fb_info = vk::FramebufferCreateInfo {
            render_pass: vk_render_pass,
            attachment_count: attachments.len() as u32,
            p_attachments: attachments.as_ptr(),
            width: vk_swapchain_extent.width,
            height: vk_swapchain_extent.height,
            layers: 1,
            ..Default::default()
        };

        let mut fbs = Vec::with_capacity(vk_swapchain_image_views.len());
        for view in vk_swapchain_image_views {
            attachments[0] = view;
            fbs.push(vk_device.create_framebuffer(&fb_info, None).unwrap())
        }

        fbs
    };

    //Create vertex buffer
    unsafe {
        let triangle_vertex_data = [
            0.5f32, 0.25, 0.0, 0.0, 1.0,
            0.25, 0.75, 1.0, 0.0, 0.0,
            0.75, 0.75, 0.0, 1.0, 0.0
        ];

        let buffer_create_info = vk::BufferCreateInfo {
            usage: vk::BufferUsageFlags::VERTEX_BUFFER,
            size: (triangle_vertex_data.len() * size_of::<f32>()) as u64,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            ..Default::default()
        };
        let vertex_buffer = vk_device.create_buffer(&buffer_create_info, None).unwrap();

        let mem_reqs = vk_device.get_buffer_memory_requirements(vertex_buffer);
        let memory_type_index = get_memory_type_index(
            &vk_instance,
            vk_physical_device,
            mem_reqs,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT
        );
        let alloc_info = vk::MemoryAllocateInfo {
            allocation_size: mem_reqs.size,
            memory_type_index,
            ..Default::default()
        };
        let vertex_buffer_memory = vk_device.allocate_memory(&alloc_info, None).unwrap();

        vk_device.bind_buffer_memory(vertex_buffer, vertex_buffer_memory, 0).unwrap();
    }

    //Configure pipeline state
    {
        let mut dynamic_state_enables = [vk::DynamicState::default(); 2];
        let dynamic_state = vk::PipelineDynamicStateCreateInfo {
            p_dynamic_states: dynamic_state_enables.as_ptr(),
            dynamic_state_count: 0,
            ..Default::default()
        };

        let vertex_binding = vk::VertexInputBindingDescription {
            binding: 0,
            stride: 5 * size_of::<f32>() as u32,
            input_rate: vk::VertexInputRate::VERTEX
        };

        let position_attribute = vk::VertexInputAttributeDescription {
            location: 0,
            binding: 0,
            format: vk::Format::R32G32_SFLOAT,
            offset: 0
        };

        let color_attribute = vk::VertexInputAttributeDescription {
            location: 1,
            binding: 1,
            format: vk::Format::R32G32B32_SFLOAT,
            offset: 2 * size_of::<f32>() as u32
        };

        let attrs = [position_attribute, color_attribute];
        let vertex_input_state = vk::PipelineVertexInputStateCreateInfo {
            vertex_binding_description_count: 1,
            p_vertex_binding_descriptions: &vertex_binding,
            vertex_attribute_description_count: attrs.len() as u32,
            p_vertex_attribute_descriptions: attrs.as_ptr(),
            ..Default::default()
        };

        let input_assembly_state = vk::PipelineInputAssemblyStateCreateInfo {
            topology: vk::PrimitiveTopology::TRIANGLE_LIST,
            ..Default::default()
        };

        let rasterization_state = vk::PipelineRasterizationStateCreateInfo {
            polygon_mode: vk::PolygonMode::FILL,
            cull_mode: vk::CullModeFlags::BACK,
            front_face: vk::FrontFace::COUNTER_CLOCKWISE,
            depth_clamp_enable: vk::TRUE,
            rasterizer_discard_enable: vk::FALSE,
            depth_bias_enable: vk::FALSE,
            line_width: 1.0,
            ..Default::default()
        };

        let color_blend_attachment_state = vk::PipelineColorBlendAttachmentState {
            color_write_mask: vk::ColorComponentFlags::from_raw(0xF),   //All components
            blend_enable: vk::FALSE,
            alpha_blend_op: vk::BlendOp::ADD,
            color_blend_op: vk::BlendOp::ADD,
            src_color_blend_factor: vk::BlendFactor::ZERO,
            dst_color_blend_factor: vk::BlendFactor::ZERO,
            src_alpha_blend_factor: vk::BlendFactor::ZERO,
            dst_alpha_blend_factor: vk::BlendFactor::ZERO
        };

        let color_blend_pipeline_state = vk::PipelineColorBlendStateCreateInfo {
            attachment_count: 1,
            p_attachments: &color_blend_attachment_state,
            logic_op_enable: vk::FALSE,
            logic_op: vk::LogicOp::NO_OP,
            blend_constants: [0.0; 4],
            ..Default::default()
        };

        dynamic_state_enables[0] = vk::DynamicState::VIEWPORT;
        dynamic_state_enables[1] = vk::DynamicState::SCISSOR;
        let viewport_state = vk::PipelineViewportStateCreateInfo {
            viewport_count: 1,
            p_scissors: ptr::null(),
            p_viewports: ptr::null(),
            ..Default::default()
        };


    }

    //Main application loop
    let mut timer = FrameTimer::new();
    'running: loop {
        timer.update(); //Update frame timer

        //Pump event queue
        for event in event_pump.poll_iter() {
            match event {
                Event::Quit{..} => { break 'running; }
                _ => {}
            }
        }

        //Update

        
    }
}
