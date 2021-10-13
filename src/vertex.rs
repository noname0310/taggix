use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct Vertex {
    _pos: [f32; 4],
    _tex_coord: [f32; 2],
}

impl Vertex {
    pub fn new(pos: [f32; 3], tc: [f32; 2]) -> Self {
        Self {
            _pos: [pos[0], pos[1], pos[2], 1.0],
            _tex_coord: tc,
        }
    }
}