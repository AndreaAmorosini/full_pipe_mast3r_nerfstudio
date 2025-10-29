import logging
from pathlib import Path
import jinja2
from core.utils import run_command
from core.context import PipelineContext


class CommandRunnerStep:
    """Esegue uno step leggendo il suo manifest .toml."""

    def __init__(
        self, context: PipelineContext, step_config: dict, method_config: dict
    ):
        self.context = context
        self.step_config = step_config  # Config da pipeline.toml
        self.method_config = method_config  # Config da method.toml
        self.name = self.method_config["name"]
        self.logger = logging.getLogger(self.name)
        self.verbose = context.get("verbose", False)
        self.jinja_env = jinja2.Environment(loader=jinja2.BaseLoader())

        # Prende la root del progetto dal context (iniettata da main.py)
        self.project_root = Path(self.context.get("project_root", "."))

    def run(self):
        try:
            template_vars = self._prepare_template_vars()
            command_template = self.method_config["execution"]["command"]
            command_to_run = self._render_template(command_template, template_vars)

            self.logger.info(f"Avvio esecuzione per {self.name}...")
            run_command(
                command_to_run,
                log_name=self.logger.name,
                verbose=self.verbose,
                shell=True,
            )

            self._check_and_register_outputs(template_vars)
        except Exception as e:
            self.logger.error(f"Esecuzione fallita: {e}", exc_info=self.verbose)
            raise

    def _render_template(self, template_str: str, vars: dict) -> str:
        template = self.jinja_env.from_string(template_str)
        return template.render(vars)

    def _prepare_template_vars(self) -> dict:
        """Raccoglie tutte le variabili per Jinja2."""
        
        method_vendor_dir = (self.project_root / "full_pipe_v2" / "vendor" / self.name).resolve()

        vars = {
            "context": self.context.data,
            "config": self.step_config,
            "method": self.method_config,
            "step_output_dir": str(self.context.get_step_output_dir(self.name)),
            "project_root": str(self.project_root.resolve()),
            "method_vendor_dir": str(method_vendor_dir),
        }

        # Calcola e aggiungi 'env_path'
        env_name = self.method_config.get("installation", {}).get("conda_env_name")
        if env_name:
            env_path = (
                self.project_root / "full_pipe_v2" / ".envs" / env_name
            ).resolve()
            vars["env_path"] = str(env_path)

            if not env_path.exists():
                pass  # La logica di installazione gestirà questo

        # Risolvi gli input
        inputs = {}
        if "inputs" in self.method_config["execution"]:
            for key in self.method_config["execution"]["inputs"]:
                input_value = self.context.get_required(key)
                inputs[key] = input_value
        vars["inputs"] = inputs

        # Risolvi gli output
        outputs = {}
        if "outputs" in self.method_config["execution"]:
            for key, path_template in self.method_config["execution"][
                "outputs"
            ].items():
                rendered_path = self._render_template(path_template, vars)
                outputs[key] = rendered_path
        vars["outputs"] = outputs

        # Prepara i kwargs per il comando
        kwargs = {}
        if "template_vars" in self.method_config["execution"]:
            for key, template_str in self.method_config["execution"][
                "template_vars"
            ].items():
                rendered_value = self._render_template(template_str, vars)
                # Prova a convertire in tipi numerici o booleani
                if isinstance(rendered_value, str):
                    if rendered_value.lower() == "true":
                        kwargs[key] = True
                    elif rendered_value.lower() == "false":
                        kwargs[key] = False
                    elif rendered_value.isdigit():
                        kwargs[key] = int(rendered_value)
                    else:
                        try:
                            kwargs[key] = float(rendered_value)
                        except ValueError:
                            kwargs[key] = rendered_value
                else:
                    kwargs[key] = rendered_value
        vars["kwargs"] = kwargs

        return vars
    
    def _check_and_register_outputs(self, template_vars: dict):
        """Verifica e salva l'output primario nel context."""
        self.logger.info("Verifica degli output...")
        primary_output_name = self.method_config["execution"].get("primary_output")
        if not primary_output_name:
            self.logger.debug("Nessun 'primary_output' definito. Step completato.")
            return

        output_key_in_context = self.step_config.get("output_key")
        if not output_key_in_context:
            self.logger.warning(
                f"'output_key' non specificato. L'output non sarà passato."
            )
            return

        output_path_str = template_vars["outputs"].get(primary_output_name)
        if not output_path_str:
            raise ValueError(f"primary_output '{primary_output_name}' non trovato.")

        output_path = Path(output_path_str)
        if not output_path.exists():
            raise FileNotFoundError(
                f"Output primario atteso non trovato in {output_path}"
            )

        self.context.set(output_key_in_context, str(output_path))
        self.logger.info(f"Output salvato in context['{output_key_in_context}']")
