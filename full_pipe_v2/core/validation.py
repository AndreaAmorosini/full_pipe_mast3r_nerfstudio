import toml
import logging
import subprocess
from pathlib import Path
from typing import Dict, Any, Tuple
import jinja2
import typer


class Validator:
    """Carica, elenca e valida i metodi dai manifest .toml."""

    def __init__(self, methods_dir: Path):
        self.methods_dir = methods_dir
        self.project_root = methods_dir.parent
        self.registry = self._load_method_registry()
        self.jinja_env = jinja2.Environment(loader=jinja2.BaseLoader())

    def _load_method_registry(self) -> Dict[str, Any]:
        """Scansiona 'methods/' e carica tutti i .toml."""
        registry = {}
        if not self.methods_dir.is_dir():
            raise FileNotFoundError(f"Directory metodi non trovata: {self.methods_dir}")

        for toml_file in self.methods_dir.glob("**/*.toml"):
            try:
                config = toml.load(toml_file)
                name = config["name"]
                config["__path__"] = (
                    toml_file  # Salva il percorso per riferimenti futuri
                )
                registry[name] = config
            except Exception as e:
                logging.warning(f"Impossibile caricare {toml_file}: {e}")
        return registry

    def find_method_manifest(self, method_name: str) -> Tuple[Path, Dict[str, Any]]:
        """Trova il percorso e la config di un metodo dal suo nome."""
        if method_name not in self.registry:
            raise FileNotFoundError(f"Metodo '{method_name}' non trovato nel registro.")
        config = self.registry[method_name]
        return config["__path__"], config

    def list_methods(self):
        """Stampa un elenco di tutti i metodi trovati."""
        if not self.registry:
            typer.echo("Nessun metodo registrato.")
            return

        for name, config in self.registry.items():
            typer.secho(f"\n{name} (tipo: {config.get('step_type', 'N/A')})", bold=True)
            typer.echo(f"  Desc: {config.get('description', 'N/A')}")
            typer.echo(
                f"  Manifest: {config['__path__'].relative_to(self.project_root)}"
            )

    def validate_method(self, name: str, config: dict, verbose: bool) -> bool:
        """Esegue il 'validation_command' per un singolo metodo."""
        cmd_template = config.get("validation", {}).get("validation_command")
        if not cmd_template:
            logging.warning(f"Nessun 'validation_command' per {name}. Assunto valido.")
            return True

        try:
            # Prepara le variabili per il template (solo env_path)
            vars = {}
            env_name = config.get("installation", {}).get("conda_env_name")
            if env_name:
                env_path = (self.project_root / ".envs" / env_name).resolve()
                if not env_path.exists():
                    logging.warning(
                        f"Ambiente locale non trovato in {env_path}. Validazione fallirà."
                    )
                    return False
                vars["env_path"] = str(env_path)

            cmd = self.jinja_env.from_string(cmd_template).render(vars)

            logging.debug(f"Validazione {name} con: {cmd}")
            subprocess.run(
                cmd, shell=True, check=True, capture_output=not verbose, text=True
            )
            return True

        except Exception as e:
            logging.error(f"Validazione fallita per {name}: {e}")
            if not verbose and hasattr(e, "stderr"):
                logging.error(f"Errore: {e.stderr.strip()}")
            return False

    def validate_all(self, verbose: bool) -> bool:
        """Valida tutti i metodi nel registro."""
        overall_success = True
        typer.echo("Validazione installazione tool (negli ambienti ./.envs/)...")

        if not self.registry:
            typer.echo("Nessun metodo da validare.")
            return True

        for name, config in self.registry.items():
            typer.echo(f"  Validando [{name}]...", nl=False)
            success = self.validate_method(name, config, verbose)
            if success:
                typer.secho(" OK", fg=typer.colors.GREEN, bold=True)
            else:
                typer.secho(" FALLITO", fg=typer.colors.RED, bold=True)
                overall_success = False

        return overall_success
