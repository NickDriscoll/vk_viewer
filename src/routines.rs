use crate::*;

//Converts any data structure to a slice of bytes
pub fn struct_to_bytes<'a, T>(structure: &'a T) -> &'a [u8] {
    let p = structure as *const _ as *const u8;
    let size = size_of::<T>();
    unsafe { std::slice::from_raw_parts(p, size) }
}