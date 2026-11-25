"""
engine/plugins_manager.py - Dynamic User Plugin Architecture
Discovers and loads custom Python plugins strictly from the per-user configuration directory.

SECURITY NOTICE:
Plugins execute arbitrary Python code with full user privileges.
They are NOT sandboxed and have unrestricted access to local files, network,
processes, and environment variables.
"""

import ast
import os
import re
import hashlib
import importlib.util
import json
import urllib.request
import platformdirs

SECURITY_WARNING = (
    "SECURITY WARNING: Potatostics plugins execute arbitrary Python code with full user "
    "privileges. Enabled plugins have unrestricted access to your filesystem, network, "
    "environment variables, and local processes. Only enable plugins from authors you trust."
)

MAX_PLUGIN_DOWNLOAD_SIZE = 10 * 1024 * 1024  # 10 MB maximum remote plugin download limit

COMMUNITY_CATALOG = {}


class PluginsManager:
    SECURITY_WARNING = SECURITY_WARNING

    def __init__(self, interpreter, plugins_dir=None, enabled=True):
        self.interpreter = interpreter
        # Opt-out via constructor flag or environment variable
        env_disabled = os.environ.get("POTATOSTICS_NO_PLUGINS", "0").lower() in ("1", "true", "yes")
        self.enabled = bool(enabled and not env_disabled)

        if plugins_dir is None:
            self.plugins_dir = os.path.join(platformdirs.user_config_dir("potatostics"), "plugins")
        else:
            self.plugins_dir = plugins_dir

        self.state_file = os.path.join(self.plugins_dir, ".plugin_state.json")
        self.loaded_plugins = []
        self._enabled_plugins, self._allow_script_plugins = self._load_state()
        self._plugin_info = {}
        self._ensure_plugins_dir()

    def _ensure_plugins_dir(self):
        if not os.path.exists(self.plugins_dir):
            try:
                os.makedirs(self.plugins_dir, exist_ok=True)
            except Exception:
                pass

    def _load_state(self):
        """Loads persistent set of explicitly enabled plugins and script permissions from state file."""
        enabled = set()
        allow_scripts = False
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        enabled = set(data)
                    elif isinstance(data, dict):
                        enabled = set(data.get("enabled", []))
                        allow_scripts = bool(data.get("allow_script_plugins", False))
            except Exception:
                pass
        return enabled, allow_scripts

    def _save_state(self):
        """Persists set of enabled plugins and script permissions to config."""
        self._ensure_plugins_dir()
        try:
            with open(self.state_file, "w", encoding="utf-8") as f:
                json.dump({
                    "enabled": sorted(list(self._enabled_plugins)),
                    "allow_script_plugins": bool(self._allow_script_plugins)
                }, f, indent=2)
        except Exception:
            pass

    def get_allow_script_plugins(self) -> bool:
        """Returns True if scripts/do-files are permitted to install and enable plugins."""
        return getattr(self, "_allow_script_plugins", False)

    def set_allow_script_plugins(self, allow: bool):
        """Updates and persists the allow_script_plugins configuration."""
        self._allow_script_plugins = bool(allow)
        self._save_state()

    def _save_enabled_state(self):
        self._save_state()

    def validate_plugin_name(self, name):
        """
        Sanitizes plugin name. Allows only [a-z0-9_]+ and ensures resolved path stays
        strictly inside the user plugins directory to prevent path traversal attacks.
        """
        if not name or not isinstance(name, str):
            raise ValueError("Plugin name must be a non-empty string.")
        clean_name = name.strip().lower()
        if not re.match(r'^[a-z0-9_]+$', clean_name):
            raise ValueError(f"Invalid plugin name '{name}': only lowercase alphanumeric characters and underscores [a-z0-9_]+ are allowed.")

        target_path = os.path.abspath(os.path.join(self.plugins_dir, f"{clean_name}.py"))
        expected_dir = os.path.abspath(self.plugins_dir)
        try:
            common = os.path.commonpath([expected_dir, target_path])
        except ValueError:
            raise ValueError(f"Directory traversal attack detected for plugin name '{name}'.")

        if common != expected_dir or not target_path.startswith(expected_dir + os.sep):
            raise ValueError(f"Directory traversal attack detected for plugin name '{name}'.")

        return clean_name

    def enable_plugin(self, name):
        """Explicitly enables a plugin by name."""
        name = self.validate_plugin_name(name)
        self._enabled_plugins.add(name)
        self._save_enabled_state()

    def disable_plugin(self, name):
        """Disables a plugin by name."""
        name = self.validate_plugin_name(name)
        if name in self._enabled_plugins:
            self._enabled_plugins.remove(name)
            self._save_enabled_state()

    def is_enabled(self, name):
        """Checks if a plugin is currently enabled."""
        name = self.validate_plugin_name(name)
        return name in self._enabled_plugins

    def discover_plugins(self):
        """
        Scans plugins_dir in deterministic sorted order.
        Returns a dict mapping plugin names to metadata dictionaries.
        """
        discovered = {}
        if not os.path.exists(self.plugins_dir):
            return discovered

        for f in sorted(os.listdir(self.plugins_dir)):
            if f.endswith(".py") and not f.startswith("__") and not f.startswith("."):
                name = os.path.splitext(f)[0]
                path = os.path.join(self.plugins_dir, f)
                discovered[name] = {
                    "name": name,
                    "path": path,
                    "version": "1.0",
                    "enabled": name in self._enabled_plugins,
                    "loaded": name in self.loaded_plugins,
                    "status": "Loaded" if name in self.loaded_plugins else ("Enabled" if name in self._enabled_plugins else "Disabled")
                }
        self._plugin_info = discovered
        return discovered

    def load_all_plugins(self):
        """
        Loads all explicitly enabled plugins in deterministic order.
        Returns list of successfully loaded plugin names.
        """
        if not self.enabled:
            self.loaded_plugins = []
            return []

        pdir = self.plugins_dir
        if not os.path.exists(pdir):
            return []

        self.discover_plugins()
        loaded = []

        # Deterministic alphabetical iteration
        for f in sorted(os.listdir(pdir)):
            if f.endswith(".py") and not f.startswith("__") and not f.startswith("."):
                plugin_path = os.path.join(pdir, f)
                plugin_name = os.path.splitext(f)[0]
                
                # Check explicit enablement
                if plugin_name not in self._enabled_plugins:
                    continue

                if plugin_name in loaded:
                    continue

                try:
                    spec = importlib.util.spec_from_file_location(plugin_name, plugin_path)
                    if spec is None or spec.loader is None:
                        continue
                    
                    mod = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(mod)

                    if hasattr(mod, "register_plugin"):
                        mod.register_plugin(self.interpreter)
                        loaded.append(plugin_name)
                        
                        version = getattr(mod, "__version__", "1.0")
                        if plugin_name in self._plugin_info:
                            self._plugin_info[plugin_name]["version"] = str(version)
                            self._plugin_info[plugin_name]["loaded"] = True
                            self._plugin_info[plugin_name]["status"] = "Loaded"
                except Exception as e:
                    print(f"Failed to load plugin {plugin_name}: {e}")
                    if plugin_name in self._plugin_info:
                        self._plugin_info[plugin_name]["status"] = f"Error: {e}"

        self.loaded_plugins = loaded
        return loaded

    def list_plugins(self):
        """Returns metadata list of all discovered plugins."""
        self.discover_plugins()
        return list(self._plugin_info.values())

    def create_template(self, name):
        """Scaffolds a new plugin template file in the user plugins directory."""
        self._ensure_plugins_dir()
        name = self.validate_plugin_name(name)

        target_path = os.path.join(self.plugins_dir, f"{name}.py")
        if os.path.exists(target_path):
            raise FileExistsError(f"Plugin '{name}' already exists at {target_path}.")

        template = f'''"""
Plugin: {name}
Author: User
Description: Custom econometric routine for Potatostics.
"""
import numpy as np
import pandas as pd
from engine.parser import CommandResult

def {name}_cmd(interp, cmd):
    """
    Handler for '{name}'.
    Access dataset via interp.dataset.df.
    Return a CommandResult(text, ok=True) or string.
    """
    tokens = cmd.body.split()
    if not tokens:
        return CommandResult("syntax: {name} varlist [, option(val)]", ok=False, return_code=198)

    var = tokens[0]
    df = interp.dataset.df
    if var not in df.columns:
        return CommandResult(f"variable '{{var}}' not found", ok=False, return_code=111)

    mean_val = float(df[var].mean())
    interp.registers.clear_r()
    interp.registers.set_r_scalar("mean", mean_val)

    return CommandResult(f"{name}: Mean of {{var}} = {{mean_val:.4f}}", ok=True)

def register_plugin(interp):
    interp.register_command("{name}", {name}_cmd)
'''
        # Validate syntax before writing
        ast.parse(template)
        with open(target_path, "w", encoding="utf-8") as f:
            f.write(template)

        return target_path

    def search_packages(self, keyword):
        """Searches community package catalog by keyword."""
        kw = keyword.strip().lower()
        results = []
        for pkg_name, info in COMMUNITY_CATALOG.items():
            if (kw in pkg_name.lower() or
                kw in info.get("description", "").lower() or
                kw in info.get("author", "").lower()):
                results.append(info)
        return results

    def install_plugin(self, source, target_name=None, enable=False, expected_sha256=None):
        """
        Installs a plugin from community catalog, local path, or remote HTTPS URL.
        Validates Python syntax statically with AST and verifies optional SHA-256 digest.
        Default is install-without-enabling (enable=False). Code is never executed on install.
        """
        self._ensure_plugins_dir()
        code = None
        source = str(source).strip().strip('"\'')
        pkg_name = str(target_name).strip().strip('"\'') if target_name else None

        if source.startswith("http://"):
            raise ValueError(f"Insecure URL scheme '{source}': only https:// URLs are permitted for plugin installation.")
        elif source.startswith("https://"):
            req = urllib.request.Request(source, headers={"User-Agent": "Potatostics-PkgMgr/1.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                chunks = []
                total_bytes = 0
                while True:
                    chunk = resp.read(65536)
                    if not chunk:
                        break
                    total_bytes += len(chunk)
                    if total_bytes > MAX_PLUGIN_DOWNLOAD_SIZE:
                        raise ValueError(
                            f"Plugin download from '{source}' exceeded maximum size limit "
                            f"of {MAX_PLUGIN_DOWNLOAD_SIZE} bytes (received {total_bytes} bytes)."
                        )
                    chunks.append(chunk)
                code = b"".join(chunks).decode("utf-8")
            if not pkg_name:
                url_base = source.rstrip("/").split("/")[-1]
                pkg_name = os.path.splitext(url_base)[0]
        elif source in COMMUNITY_CATALOG:
            cat_entry = COMMUNITY_CATALOG[source]
            code = cat_entry["code"]
            pkg_name = pkg_name or source
        elif os.path.exists(source):
            with open(source, "r", encoding="utf-8") as f:
                code = f.read()
            if not pkg_name:
                pkg_name = os.path.splitext(os.path.basename(source))[0]
        else:
            raise ValueError(f"Package '{source}' not found in community catalog, local filesystem, or URL.")

        if not pkg_name:
            raise ValueError("Could not determine package name.")

        pkg_name = self.validate_plugin_name(pkg_name)

        code_bytes = code.encode("utf-8")
        actual_sha256 = hashlib.sha256(code_bytes).hexdigest()
        file_size = len(code_bytes)

        if expected_sha256:
            clean_expected = expected_sha256.strip().lower().strip('"\'')
            if actual_sha256.lower() != clean_expected:
                raise ValueError(
                    f"SHA-256 verification failed for package '{pkg_name}': "
                    f"expected {expected_sha256}, got {actual_sha256}"
                )

        # Static syntax validation without execution
        try:
            ast.parse(code)
        except Exception as e:
            raise ValueError(f"Syntax validation failed for plugin '{pkg_name}': {e}")

        target_file = os.path.join(self.plugins_dir, f"{pkg_name}.py")
        with open(target_file, "w", encoding="utf-8") as f:
            f.write(code)

        if enable:
            self.enable_plugin(pkg_name)
            self.load_all_plugins()

        return {
            "name": pkg_name,
            "path": target_file,
            "size": file_size,
            "sha256": actual_sha256,
            "enabled": bool(enable),
            "status": "Installed and enabled" if enable else "Installed (disabled)"
        }

    def uninstall_plugin(self, name):
        """Uninstalls a plugin, removing file and unregistering command."""
        name = self.validate_plugin_name(name)
        self.disable_plugin(name)
        target_file = os.path.join(self.plugins_dir, f"{name}.py")
        if os.path.exists(target_file):
            try:
                os.remove(target_file)
            except Exception:
                pass

        if hasattr(self.interpreter, "custom_commands") and name in self.interpreter.custom_commands:
            del self.interpreter.custom_commands[name]

        if name in self.loaded_plugins:
            self.loaded_plugins.remove(name)

        return True

