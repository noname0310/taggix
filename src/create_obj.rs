use std::{fs::File, path::Path};
use std::io::BufReader;
use obj::{Obj, TexturedVertex, load_obj};
use crate::vertex::Vertex;

pub fn obj_from_flie<P: AsRef<Path>>(path: P) -> Result<(Vec<Vertex>, Vec<u16>), String> {
    let file = match File::open(path) {
        Ok(file) => file,
        Err(err) => return Err(err.to_string())
    };
    let input = BufReader::new(file);

    let obj: Obj<TexturedVertex> = match load_obj(input) {
        Ok(obj) => obj,
        Err(err) => return Err(err.to_string())
    };
    
    let vertex_data = obj.vertices.iter()
        .map(|v| Vertex::new(v.position, [v.texture[0], v.texture[1]]))
        .collect();

    let index_data = obj.indices.iter()
        .map(|i| *i as u16)
        .collect();

    Ok((vertex_data, index_data))
}
