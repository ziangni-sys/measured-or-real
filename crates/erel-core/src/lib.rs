pub mod dgp;
pub mod disatt;
pub mod pcg;
pub mod reliability;

pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}
