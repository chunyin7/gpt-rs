use std::{
    fs::File,
    io::{Read, Write},
};

pub(super) fn write_u32(writer: &mut dyn Write, value: u32) -> Result<(), String> {
    writer
        .write_all(&(value).to_le_bytes())
        .map_err(|e| e.to_string())?;
    Ok(())
}

pub(super) fn read_u32(reader: &mut dyn Read) -> Result<u32, String> {
    let mut buf = [0_u8; 4];
    reader.read_exact(&mut buf).map_err(|e| e.to_string())?;
    Ok(u32::from_le_bytes(buf))
}

pub(super) fn write_u64(writer: &mut dyn Write, value: u64) -> Result<(), String> {
    writer
        .write_all(&(value).to_le_bytes())
        .map_err(|e| e.to_string())?;
    Ok(())
}

pub(super) fn read_u64(reader: &mut dyn Read) -> Result<u64, String> {
    let mut bytes = [0u8; 8];
    reader.read_exact(&mut bytes).map_err(|e| e.to_string())?;
    Ok(u64::from_le_bytes(bytes))
}
