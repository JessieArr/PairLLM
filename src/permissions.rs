use serde::{Deserialize, Serialize};
use std::path::{Component, Path, PathBuf};
use std::sync::{Arc, Mutex};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FileAccess {
    Read,
    Write,
}

impl FileAccess {
    pub fn label(&self) -> &'static str {
        match self {
            FileAccess::Read => "read",
            FileAccess::Write => "write",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PathPermission {
    AllowDirectory,
    AllowRecursive,
    Deny,
}

impl PathPermission {
    pub fn label(&self) -> &'static str {
        match self {
            PathPermission::AllowDirectory => "Allow directory",
            PathPermission::AllowRecursive => "Allow recursively",
            PathPermission::Deny => "Deny",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PathPermissionRule {
    pub path: String,
    pub permission: PathPermission,
}

#[derive(Clone, Default)]
pub struct PathPermissionState {
    pub persistent: Vec<PathPermissionRule>,
    pub session: Vec<PathPermissionRule>,
}

impl PathPermissionState {
    pub fn all_rules(&self) -> impl Iterator<Item = &PathPermissionRule> {
        self.persistent.iter().chain(self.session.iter())
    }

    pub fn add_session_rule(&mut self, rule: PathPermissionRule) {
        self.session.push(rule);
    }

    pub fn add_persistent_rule(&mut self, rule: PathPermissionRule) {
        self.persistent.push(rule);
    }

    pub fn sync_persistent(&mut self, rules: &[PathPermissionRule]) {
        self.persistent = rules.to_vec();
    }
}

pub type SharedPathPermissions = Arc<Mutex<PathPermissionState>>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PermissionCheck {
    Allowed,
    Denied,
    NeedsPrompt,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FilePermissionChoice {
    AllowDirectory,
    AllowRecursive,
    Reject,
}

impl FilePermissionChoice {
    pub fn to_rule(self, directory: &Path) -> PathPermissionRule {
        PathPermissionRule {
            path: directory.display().to_string(),
            permission: match self {
                FilePermissionChoice::AllowDirectory => PathPermission::AllowDirectory,
                FilePermissionChoice::AllowRecursive => PathPermission::AllowRecursive,
                FilePermissionChoice::Reject => PathPermission::Deny,
            },
        }
    }

    pub fn session_status(&self) -> &'static str {
        match self {
            FilePermissionChoice::AllowDirectory => {
                "Allowed for this directory (this session only)"
            }
            FilePermissionChoice::AllowRecursive => {
                "Allowed recursively (this session only)"
            }
            FilePermissionChoice::Reject => "Rejected (this session only)",
        }
    }
}

pub fn file_access_for_tool(tool_name: &str) -> Option<FileAccess> {
    match tool_name {
        "ls" | "cat" => Some(FileAccess::Read),
        "sed" => Some(FileAccess::Write),
        _ => None,
    }
}

pub fn is_file_permission_tool(tool_name: &str) -> bool {
    file_access_for_tool(tool_name).is_some()
}

pub fn permission_directory_for_target(target: &Path) -> PathBuf {
    if target.is_dir() {
        target.to_path_buf()
    } else {
        target
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| target.to_path_buf())
    }
}

pub fn check_path_permission(target: &Path, state: &PathPermissionState) -> PermissionCheck {
    let target = normalize_path(target);
    let mut best: Option<(usize, PathPermission)> = None;

    for rule in state.all_rules() {
        let Ok(rule_path) = normalize_path_str(&rule.path) else {
            continue;
        };

        if !rule_matches_target(&rule_path, &target, &rule.permission) {
            continue;
        }

        let specificity = path_depth(&rule_path);
        if best.as_ref().is_none_or(|(depth, _)| specificity > *depth) {
            best = Some((specificity, rule.permission.clone()));
        }
    }

    match best.map(|(_, permission)| permission) {
        Some(PathPermission::Deny) => PermissionCheck::Denied,
        Some(PathPermission::AllowDirectory) | Some(PathPermission::AllowRecursive) => {
            PermissionCheck::Allowed
        }
        None => PermissionCheck::NeedsPrompt,
    }
}

fn rule_matches_target(rule_path: &Path, target: &Path, permission: &PathPermission) -> bool {
    match permission {
        PathPermission::Deny | PathPermission::AllowRecursive => is_under_or_equal(target, rule_path),
        PathPermission::AllowDirectory => {
            target == rule_path
                || (target.parent() == Some(rule_path) && !target.is_dir())
        }
    }
}

pub fn normalize_path(path: &Path) -> PathBuf {
    if let Ok(canonical) = path.canonicalize() {
        return canonical;
    }

    let mut normalized = PathBuf::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                normalized.pop();
            }
            other => normalized.push(other.as_os_str()),
        }
    }
    normalized
}

fn normalize_path_str(path: &str) -> Result<PathBuf, String> {
    let expanded = expand_tilde(path)?;
    Ok(normalize_path(Path::new(&expanded)))
}

pub fn expand_tilde(path: &str) -> Result<PathBuf, String> {
    if path == "~" {
        return home_dir().ok_or_else(|| "Could not resolve home directory.".to_string());
    }

    if let Some(rest) = path.strip_prefix("~/") {
        let home = home_dir().ok_or_else(|| "Could not resolve home directory.".to_string())?;
        return Ok(home.join(rest));
    }

    Ok(PathBuf::from(path))
}

fn home_dir() -> Option<PathBuf> {
    std::env::var("HOME")
        .ok()
        .map(PathBuf::from)
        .or_else(|| std::env::var("USERPROFILE").ok().map(PathBuf::from))
}

fn path_depth(path: &Path) -> usize {
    path.components().count()
}

fn is_under_or_equal(path: &Path, base: &Path) -> bool {
    if path == base {
        return true;
    }

    path_depth(path) > path_depth(base) && path_starts_with(path, base)
}

fn path_starts_with(path: &Path, prefix: &Path) -> bool {
    let mut path_components = path.components();
    for prefix_component in prefix.components() {
        match path_components.next() {
            Some(component) if component == prefix_component => {}
            _ => return false,
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn rule(path: &str, permission: PathPermission) -> PathPermissionRule {
        PathPermissionRule {
            path: path.to_string(),
            permission,
        }
    }

    #[test]
    fn deny_in_subdirectory_overrides_parent_allow() {
        let home = home_dir().expect("home");
        let state = PathPermissionState {
            persistent: vec![
                rule(&home.display().to_string(), PathPermission::AllowRecursive),
                rule(
                    &home.join(".ssh").display().to_string(),
                    PathPermission::Deny,
                ),
            ],
            session: vec![],
        };

        let target = home.join(".ssh/id_rsa");
        assert_eq!(
            check_path_permission(&target, &state),
            PermissionCheck::Denied
        );
    }

    #[test]
    fn allow_directory_does_not_cover_subdirectories() {
        let dir = std::env::temp_dir().join(format!("pairllm-perm-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(dir.join("nested")).expect("create nested");

        let state = PathPermissionState {
            persistent: vec![rule(
                &dir.display().to_string(),
                PathPermission::AllowDirectory,
            )],
            session: vec![],
        };

        assert_eq!(
            check_path_permission(&dir, &state),
            PermissionCheck::Allowed
        );
        assert_eq!(
            check_path_permission(&dir.join("file.txt"), &state),
            PermissionCheck::Allowed
        );
        assert_eq!(
            check_path_permission(&dir.join("nested"), &state),
            PermissionCheck::NeedsPrompt
        );
        assert_eq!(
            check_path_permission(&dir.join("nested/file.txt"), &state),
            PermissionCheck::NeedsPrompt
        );

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn allow_recursive_covers_descendants() {
        let dir = std::env::temp_dir().join(format!("pairllm-perm-rec-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(dir.join("nested")).expect("create nested");

        let state = PathPermissionState {
            persistent: vec![rule(
                &dir.display().to_string(),
                PathPermission::AllowRecursive,
            )],
            session: vec![],
        };

        assert_eq!(
            check_path_permission(&dir.join("nested/file.txt"), &state),
            PermissionCheck::Allowed
        );

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn session_rule_can_allow_after_prompt() {
        let dir = std::env::temp_dir().join(format!("pairllm-perm-session-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("create dir");

        let mut state = PathPermissionState::default();
        assert_eq!(
            check_path_permission(&dir.join("file.txt"), &state),
            PermissionCheck::NeedsPrompt
        );

        state.add_session_rule(rule(
            &dir.display().to_string(),
            PathPermission::AllowDirectory,
        ));
        assert_eq!(
            check_path_permission(&dir.join("file.txt"), &state),
            PermissionCheck::Allowed
        );

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn permission_directory_uses_parent_for_files() {
        let path = PathBuf::from("/tmp/example/file.txt");
        assert_eq!(
            permission_directory_for_target(&path),
            PathBuf::from("/tmp/example")
        );
    }
}
