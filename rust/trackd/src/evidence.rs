//! Bounded service evidence: current file plus a fixed number of rotated files.
use anyhow::{Result, ensure};
use std::{fs::File, io::Write, path::PathBuf};
pub struct RollingEvidence {
    path: PathBuf,
    file: File,
    bytes: u64,
    limit: u64,
    files: usize,
}
impl RollingEvidence {
    pub fn new(path: PathBuf, limit: u64, files: usize) -> Result<Self> {
        ensure!(
            limit >= 1024 && (1..=32).contains(&files),
            "invalid evidence bounds"
        );
        let file = File::create_new(&path)?;
        Ok(Self {
            path,
            file,
            bytes: 0,
            limit,
            files,
        })
    }
    fn older(&self, index: usize) -> PathBuf {
        self.path.with_file_name(format!(
            "{}.{}",
            self.path.file_name().unwrap().to_string_lossy(),
            index
        ))
    }
    pub fn write_record(&mut self, bytes: &[u8]) -> Result<()> {
        let size = bytes.len() as u64 + 1;
        ensure!(size <= self.limit, "evidence record exceeds file limit");
        if self.bytes + size > self.limit {
            self.file.sync_all()?;
            if self.files > 1 {
                let oldest = self.older(self.files - 1);
                if oldest.exists() {
                    std::fs::remove_file(oldest)?;
                }
                for index in (1..self.files - 1).rev() {
                    let old = self.older(index);
                    if old.exists() {
                        std::fs::rename(old, self.older(index + 1))?;
                    }
                }
                std::fs::rename(&self.path, self.older(1))?;
            } else {
                std::fs::remove_file(&self.path)?;
            }
            self.file = File::create_new(&self.path)?;
            self.bytes = 0;
        }
        self.file.write_all(bytes)?;
        self.file.write_all(b"\n")?;
        self.bytes += size;
        Ok(())
    }
    pub fn finish(&self) -> Result<()> {
        self.file.sync_all()?;
        Ok(())
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rotation_keeps_complete_records_with_a_fixed_disk_bound() {
        let dir = std::env::temp_dir().join(format!(
            "trackd-evidence-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir(&dir).unwrap();
        let path = dir.join("estimates.jsonl");
        let mut output = RollingEvidence::new(path.clone(), 1024, 3).unwrap();
        for index in 0..30 {
            output
                .write_record(
                    format!("{{\"index\":{index},\"padding\":\"{}\"}}", "x".repeat(400)).as_bytes(),
                )
                .unwrap();
        }
        output.finish().unwrap();
        let files = std::fs::read_dir(&dir)
            .unwrap()
            .collect::<std::io::Result<Vec<_>>>()
            .unwrap();
        assert_eq!(files.len(), 3);
        for entry in files {
            assert!(entry.metadata().unwrap().len() <= 1024);
            for line in std::fs::read_to_string(entry.path()).unwrap().lines() {
                serde_json::from_str::<serde_json::Value>(line).unwrap();
            }
        }
        assert!(
            std::fs::read_to_string(path)
                .unwrap()
                .contains("\"index\":29")
        );
        assert!(output.write_record(&vec![0; 1024]).is_err());
        drop(output);
        std::fs::remove_dir_all(dir).unwrap();
    }
}
