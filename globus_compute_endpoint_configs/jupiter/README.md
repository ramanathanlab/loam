# Jupiter managed endpoint templates

The combined-allocation scaling route does not need Globus. These templates
support the optional managed endpoint route. They contain placeholders, not a
particular user's environment or authentication material.

Render into a **new** endpoint configuration directory:

```bash
PYTHONPATH=. .venv/bin/python scripts/render_globus_endpoint_config.py \
  /path/to/new-endpoint-config --software-root /shared/software --name loam
```

Inspect the rendered files and use the site's Globus Compute deployment process
to register/start that endpoint. Rendering performs no registration,
authentication, scheduler submission, or changes to existing endpoints.
