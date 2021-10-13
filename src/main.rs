mod framework;
mod texture;
mod renderer;
mod vertex;
mod create_obj;

fn main() {
    framework::run::<renderer::Renderer>("taggix");
}
