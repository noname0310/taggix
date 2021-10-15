use cgmath::prelude::*;
use rayon::prelude::*;
use wgpu::util::DeviceExt;
use winit::{
    event::*,
    event_loop::{ControlFlow, EventLoop},
    window::Window,
};

mod camera;
mod model;
mod texture;
mod instance;

use model::{DrawModel, Vertex};
use instance::*;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct CameraUniform {
    view_position: [f32; 4],
    view_proj: [[f32; 4]; 4],
}

impl CameraUniform {
    fn new() -> Self {
        Self {
            view_position: [0.0; 4],
            view_proj: cgmath::Matrix4::identity().into(),
        }
    }

    fn update_view_proj(&mut self, camera: &camera::Camera, projection: &camera::Projection) {
        self.view_position = camera.position.to_homogeneous().into();
        self.view_proj = (projection.calc_matrix() * camera.calc_matrix()).into()
    }
}

struct State {
    surface: wgpu::Surface,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    render_pipeline: wgpu::RenderPipeline,
    obj_model: model::Model,
    camera: camera::Camera,
    projection: camera::Projection,
    camera_controller: camera::CameraController,
    camera_uniform: CameraUniform,
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
    instances: Vec<Instance>,
    #[allow(dead_code)]
    instance_buffer: wgpu::Buffer,
    depth_texture: texture::Texture,
    size: winit::dpi::PhysicalSize<u32>,
    #[allow(dead_code)]
    debug_material: model::Material,
    mouse_pressed: bool,
}

fn create_render_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    color_format: wgpu::TextureFormat,
    depth_format: Option<wgpu::TextureFormat>,
    vertex_layouts: &[wgpu::VertexBufferLayout],
    shader: wgpu::ShaderModuleDescriptor,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(&shader);

    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(&format!("{:?}", shader)),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: "main",
            buffers: vertex_layouts,
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: "main",
            targets: &[wgpu::ColorTargetState {
                format: color_format,
                blend: Some(wgpu::BlendState {
                    alpha: wgpu::BlendComponent::REPLACE,
                    color: wgpu::BlendComponent::REPLACE,
                }),
                write_mask: wgpu::ColorWrites::ALL,
            }],
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: None, //Some(wgpu::Face::Back),
            // Setting this to anything other than Fill requires Features::NON_FILL_POLYGON_MODE
            polygon_mode: wgpu::PolygonMode::Fill,
            // Requires Features::DEPTH_CLAMPING
            clamp_depth: false,
            // Requires Features::CONSERVATIVE_RASTERIZATION
            conservative: false,
        },
        depth_stencil: depth_format.map(|format| wgpu::DepthStencilState {
            format,
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::Less,
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState {
            count: 1,
            mask: !0,
            alpha_to_coverage_enabled: false,
        },
    })
}

impl State {
    async fn new(window: &Window) -> Self {
        let size = window.inner_size();

        // The instance is a handle to our GPU
        // BackendBit::PRIMARY => Vulkan + Metal + DX12 + Browser WebGPU
        
        let backend = wgpu::util::backend_bits_from_env().unwrap_or_else(wgpu::Backends::all);

        let instance = wgpu::Instance::new(backend);
        let surface = unsafe { instance.create_surface(window) };
        let adapter = wgpu::util::initialize_adapter_from_env_or_default(&instance, backend, Some(&surface))
            .await
            .expect("No suitable GPU adapters found on the system!");
            
        let trace_dir = std::env::var("WGPU_TRACE");
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    features: wgpu::Features::empty(),
                    limits: wgpu::Limits::default(),
                },
                trace_dir.ok().as_ref().map(std::path::Path::new), // Trace path
            )
            .await
            .expect("Unable to find a suitable GPU adapter!");

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface.get_preferred_format(&adapter).unwrap(),
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Fifo,
        };

        surface.configure(&device, &config);

        let texture_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler {
                            comparison: false,
                            filtering: true,
                        },
                        count: None,
                    },
                ],
                label: Some("texture_bind_group_layout"),
            });

        let camera = camera::Camera::new((-1.2, 13.0, 25.0), cgmath::Deg(-90.0), cgmath::Deg(-5.0));
        let projection =
            camera::Projection::new(config.width, config.height, cgmath::Deg(45.0), 0.1, 100.0);
        let camera_controller = camera::CameraController::new(4.0, 0.4);

        let mut camera_uniform = CameraUniform::new();
        camera_uniform.update_view_proj(&camera, &projection);

        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Camera Buffer"),
            contents: bytemuck::cast_slice(&[camera_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        const SPACE_BETWEEN: f32 = 3.0;
        const NUM_INSTANCES_PER_ROW: u32 = 1;

        let instances = (0..NUM_INSTANCES_PER_ROW)
            .into_par_iter() 
            .flat_map(|z| {
                (0..NUM_INSTANCES_PER_ROW).into_par_iter().map(move |x| {
                    let x = SPACE_BETWEEN * (x as f32 - NUM_INSTANCES_PER_ROW as f32 / 2.0);
                    let z = SPACE_BETWEEN * (z as f32 - NUM_INSTANCES_PER_ROW as f32 / 2.0);

                    let position = cgmath::Vector3 { x, y: 0.0, z };

                    let rotation = if position.is_zero() {
                        cgmath::Quaternion::from_axis_angle(
                            cgmath::Vector3::unit_z(),
                            cgmath::Deg(0.0),
                        );
                        cgmath::Quaternion::from_axis_angle(position.normalize(), cgmath::Deg(0.0))
                    } else {
                        cgmath::Quaternion::from_axis_angle(position.normalize(), cgmath::Deg(45.0));
                        cgmath::Quaternion::from_axis_angle(position.normalize(), cgmath::Deg(0.0))
                    };

                    Instance { position, rotation }
                })
            })
            .collect::<Vec<_>>();

        let instance_data = instances.iter().map(Instance::to_raw).collect::<Vec<_>>();
        let instance_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Instance Buffer"),
            contents: bytemuck::cast_slice(&instance_data),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let camera_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
                label: Some("camera_bind_group_layout"),
            });

        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &camera_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
            label: Some("camera_bind_group"),
        });

        //let res_dir = std::path::Path::new(env!("OUT_DIR")).join("res");
        let obj_model = model::Model::load_mesh_buf(
            &device,
            &include_bytes!("../res/yyb_school_miku_pose/yyb bake r.obj")[..],
            //res_dir.join("yyb_school_miku/yyb school miku.obj"),
            vec![
                // model::Material::new(
                //     &device,
                //     "Hair01",//0
                //     texture::Texture::load(&device, &queue, res_dir.join("yyb_school_miku/Hair01.png"), false).unwrap(),
                //     &texture_bind_group_layout,
                // ),
                // model::Material::new(
                //     &device,
                //     "Hair02",//1
                //     texture::Texture::load(&device, &queue, res_dir.join("yyb_school_miku/Hair02.png"), false).unwrap(),
                //     &texture_bind_group_layout,
                // ),
                // model::Material::new(
                //     &device,
                //     "hairclip",//2
                //     texture::Texture::load(&device, &queue, res_dir.join("yyb_school_miku/hairclip.png"), false).unwrap(),
                //     &texture_bind_group_layout,
                // ),
                // model::Material::new(
                //     &device,
                //     "head",//3
                //     texture::Texture::load(&device, &queue, res_dir.join("yyb_school_miku/head.png"), false).unwrap(),
                //     &texture_bind_group_layout,
                // ),
                // model::Material::new(
                //     &device,
                //     "Material 4_UVP",//4
                //     texture::Texture::load(&device, &queue, res_dir.join("yyb_school_miku/Material 4_UVP.png"), false).unwrap(),
                //     &texture_bind_group_layout,
                // ),
                // model::Material::new(
                //     &device,
                //     "skin",//5
                //     texture::Texture::load(&device, &queue, res_dir.join("yyb_school_miku/skin.png"), false).unwrap(),
                //     &texture_bind_group_layout,
                // ),
                // model::Material::new(
                //     &device,
                //     "socks",//6
                //     texture::Texture::load(&device, &queue, res_dir.join("yyb_school_miku/socks.png"), false).unwrap(),
                //     &texture_bind_group_layout,
                // ),
                
                // model::Material::new(
                //     &device,
                //     "Hair01",//0
                //     texture::Texture::from_bytes(&device, &queue, include_bytes!("../res/yyb_school_miku/Hair01.png"), "Hair01", false).unwrap(),
                //     &texture_bind_group_layout,
                // ),
                // model::Material::new(
                //     &device,
                //     "Hair02",//1
                //     texture::Texture::from_bytes(&device, &queue, include_bytes!("../res/yyb_school_miku/Hair02.png"), "Hair02", false).unwrap(),
                //     &texture_bind_group_layout,
                // ),
                // model::Material::new(
                //     &device,
                //     "hairclip",//2
                //     texture::Texture::from_bytes(&device, &queue, include_bytes!("../res/yyb_school_miku/hairclip.png"), "hairclip", false).unwrap(),
                //     &texture_bind_group_layout,
                // ),
                // model::Material::new(
                //     &device,
                //     "head",//3
                //     texture::Texture::from_bytes(&device, &queue, include_bytes!("../res/yyb_school_miku/head.png"), "head", false).unwrap(),
                //     &texture_bind_group_layout,
                // ),
                // model::Material::new(
                //     &device,
                //     "Material 4_UVP",//4
                //     texture::Texture::from_bytes(&device, &queue, include_bytes!("../res/yyb_school_miku/Material 4_UVP.png"), "Material", false).unwrap(),
                //     &texture_bind_group_layout,
                // ),
                // model::Material::new(
                //     &device,
                //     "skin",//5
                //     texture::Texture::from_bytes(&device, &queue, include_bytes!("../res/yyb_school_miku/skin.png"), "skin", false).unwrap(),
                //     &texture_bind_group_layout,
                // ),
                // model::Material::new(
                //     &device,
                //     "socks",//6
                //     texture::Texture::from_bytes(&device, &queue, include_bytes!("../res/yyb_school_miku/socks.png"), "socks", false).unwrap(),
                //     &texture_bind_group_layout,
                // ),
                model::Material::new(
                    &device,
                    "hairpin",
                    texture::Texture::from_bytes(&device, &queue, include_bytes!("../res/yyb_school_miku_pose/hairpin bake.png"), "socks", false).unwrap(),
                    &texture_bind_group_layout,
                ),
                model::Material::new(
                    &device,
                    "head",
                    texture::Texture::from_bytes(&device, &queue, include_bytes!("../res/yyb_school_miku/head.png"), "socks", false).unwrap(),
                    &texture_bind_group_layout,
                ),
                model::Material::new(
                    &device,
                    "shoes",
                    texture::Texture::from_bytes(&device, &queue, include_bytes!("../res/yyb_school_miku_pose/shoes bake.png"), "socks", false).unwrap(),
                    &texture_bind_group_layout,
                ),
                model::Material::new(
                    &device,
                    "hair01",
                    texture::Texture::from_bytes(&device, &queue, include_bytes!("../res/yyb_school_miku/Hair01.png"), "socks", false).unwrap(),
                    &texture_bind_group_layout,
                ),
                model::Material::new(
                    &device,
                    "hair02",
                    texture::Texture::from_bytes(&device, &queue, include_bytes!("../res/yyb_school_miku/Hair02.png"), "socks", false).unwrap(),
                    &texture_bind_group_layout,
                ),
                model::Material::new(
                    &device,
                    "socks",
                    texture::Texture::from_bytes(&device, &queue, include_bytes!("../res/yyb_school_miku/socks.png"), "socks", false).unwrap(),
                    &texture_bind_group_layout,
                ),
                model::Material::new(
                    &device,
                    "dress white",
                    texture::Texture::from_bytes(&device, &queue, include_bytes!("../res/yyb_school_miku_pose/Material 4_UVP1.png"), "socks", false).unwrap(),
                    &texture_bind_group_layout,
                ),
                model::Material::new(
                    &device,
                    "bow",
                    texture::Texture::from_bytes(&device, &queue, include_bytes!("../res/yyb_school_miku_pose/bow bake.png"), "socks", false).unwrap(),
                    &texture_bind_group_layout,
                ),
                model::Material::new(
                    &device,
                    "dress",
                    texture::Texture::from_bytes(&device, &queue, include_bytes!("../res/yyb_school_miku_pose/dress bake.png"), "socks", false).unwrap(),
                    &texture_bind_group_layout,
                ),
            ],
            vec![
                0, //Hairclip
                1, //Body
                2, //Shoes
                3, //Hair01
                3, //Hairshadow
                4, //Hair03
                5, //Socls
                6, //Dress_White
                7, //Bow
                8, //Dress
            ],
            "yyb school miku",
        )
        .unwrap();

        let depth_texture =
            texture::Texture::create_depth_texture(&device, &config, "depth_texture");

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[
                    &texture_bind_group_layout,
                    &camera_bind_group_layout,
                ],
                push_constant_ranges: &[],
            });

        let render_pipeline = {
            let shader = wgpu::ShaderModuleDescriptor {
                label: Some("Normal Shader"),
                source: wgpu::ShaderSource::Wgsl(include_str!("shader/shader.wgsl").into()),
            };
            create_render_pipeline(
                &device,
                &render_pipeline_layout,
                config.format,
                Some(texture::Texture::DEPTH_FORMAT),
                &[model::ModelVertex::desc(), InstanceRaw::desc()],
                shader,
            )
        };

        let debug_material = {
            let diffuse_bytes = include_bytes!("../res/head.png");

            let diffuse_texture = texture::Texture::from_bytes(
                &device,
                &queue,
                diffuse_bytes,
                "res/alt-diffuse.png",
                false,
            )
            .unwrap();

            model::Material::new(
                &device,
                "alt-material",
                diffuse_texture,
                &texture_bind_group_layout,
            )
        };

        Self {
            surface,
            device,
            queue,
            config,
            render_pipeline,
            obj_model,
            camera,
            projection,
            camera_controller,
            camera_buffer,
            camera_bind_group,
            camera_uniform,
            instances,
            instance_buffer,
            depth_texture,
            size,
            #[allow(dead_code)]
            debug_material,
            mouse_pressed: false,
        }
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.projection.resize(new_size.width, new_size.height);
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
            self.depth_texture =
                texture::Texture::create_depth_texture(&self.device, &self.config, "depth_texture");
        }
    }

    fn input(&mut self, event: &DeviceEvent) -> bool {
        match event {
            DeviceEvent::Key(KeyboardInput {
                virtual_keycode: Some(key),
                state,
                ..
            }) => self.camera_controller.process_keyboard(*key, *state),
            DeviceEvent::MouseWheel { delta, .. } => {
                self.camera_controller.process_scroll(delta);
                true
            }
            DeviceEvent::Button {
                button: 1, // Left Mouse Button
                state,
            } => {
                self.mouse_pressed = *state == ElementState::Pressed;
                true
            }
            DeviceEvent::MouseMotion { delta } => {
                if self.mouse_pressed {
                    self.camera_controller.process_mouse(delta.0, delta.1);
                }
                true
            }
            _ => false,
        }
    }

    fn update(&mut self, dt: std::time::Duration) {
        self.camera_controller.update_camera(&mut self.camera, dt);
        self.camera_uniform
            .update_view_proj(&self.camera, &self.projection);
        self.queue.write_buffer(
            &self.camera_buffer,
            0,
            bytemuck::cast_slice(&[self.camera_uniform]),
        );
    }

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let frame = match self.surface.get_current_texture() {
            Ok(frame) => frame,
            Err(_) => {
               self.surface.configure(&self.device, &self.config);
               self.surface
                    .get_current_texture()
                    .expect("Failed to acquire next surface texture!")
            }
        };
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.1,
                            g: 0.2,
                            b: 0.3,
                            a: 1.0,
                        }),
                        store: true,
                    },
                }],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_texture.view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: true,
                    }),
                    stencil_ops: None,
                }),
            });

            render_pass.set_vertex_buffer(1, self.instance_buffer.slice(..));
            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.draw_model_instanced(
                &self.obj_model,
                0..self.instances.len() as u32,
                &self.camera_bind_group,
            );
        }
        self.queue.submit(Some(encoder.finish()));

        frame.present();
        Ok(())
    }
}

fn main() {
    env_logger::init();
    let event_loop = EventLoop::new();
    let title = env!("CARGO_PKG_NAME");
    let window = winit::window::WindowBuilder::new()
        .with_title(title)
        .build(&event_loop)
        .unwrap();
    let mut state = pollster::block_on(State::new(&window)); // NEW!
    let mut last_render_time = std::time::Instant::now();
    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Poll;
        match event {
            Event::MainEventsCleared => window.request_redraw(),
            Event::DeviceEvent {
                ref event,
                .. // We're not using device_id currently
            } => {
                state.input(event);
            }
            Event::WindowEvent {
                ref event,
                window_id,
            } if window_id == window.id() => {
                match event {
                    WindowEvent::CloseRequested
                    | WindowEvent::KeyboardInput {
                        input:
                            KeyboardInput {
                                state: ElementState::Pressed,
                                virtual_keycode: Some(VirtualKeyCode::Escape),
                                ..
                            },
                        ..
                    } => *control_flow = ControlFlow::Exit,
                    WindowEvent::Resized(physical_size) => {
                        state.resize(*physical_size);
                    }
                    WindowEvent::ScaleFactorChanged { new_inner_size, .. } => {
                        state.resize(**new_inner_size);
                    }
                    _ => {}
                }
            }
            Event::RedrawRequested(_) => {
                let now = std::time::Instant::now();
                let dt = now - last_render_time;
                last_render_time = now;
                state.update(dt);
                match state.render() {
                    Ok(_) => {}
                    // Reconfigure the surface if lost
                    Err(wgpu::SurfaceError::Lost) => state.resize(state.size),
                    // The system is out of memory, we should probably quit
                    Err(wgpu::SurfaceError::OutOfMemory) => *control_flow = ControlFlow::Exit,
                    // All other errors (Outdated, Timeout) should be resolved by the next frame
                    Err(e) => eprintln!("{:?}", e),
                }
            }
            _ => {}
        }
    });
}
